// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "AdeptIntegration.h"

#include <VecGeom/management/BVHManager.h>
#include "VecGeom/management/GeoManager.h"

#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4Proton.hh>
#include <G4Region.hh>
#include <G4SDManager.hh>
#include <G4VFastSimSensitiveDetector.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4ProductionCutsTable.hh>
#include <G4TransportationManager.hh>

#include <G4HepEmData.hh>
#include <G4HepEmState.hh>
#include <G4HepEmStateInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmMatCutData.hh>
#include <G4LogicalVolumeStore.hh>

#include "SensitiveDetector.hh"
#include "EventAction.hh"

#include <iomanip>

namespace adeptint {
TrackBuffer::TrackHandle adeptint::TrackBuffer::createToDeviceSlot()
{
  bool warningIssued = false;
  while (true) {
    auto &toDevice = getActiveBuffer();
    std::shared_lock lock{toDevice.mutex};
    const auto slot = toDevice.nTrack.fetch_add(1, std::memory_order_relaxed);

    if (slot < toDevice.maxTracks)
      return TrackHandle{toDevice.tracks[slot], std::move(lock)};
    else {
      if (!warningIssued) {
        std::cerr << __FILE__ << ':' << __LINE__ << " Contention in to-device queue; thread sleeping" << std::endl;
        warningIssued = true;
      }
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(1ms);
    }
  }
}
} // namespace adeptint

namespace {
template <typename T>
std::size_t countTracks(int pdgToSelect, T const &container)
{
  return std::count_if(container.begin(), container.end(),
                       [pdgToSelect](adeptint::TrackData const &track) { return track.pdg == pdgToSelect; });
}

std::ostream &operator<<(std::ostream &stream, adeptint::TrackData const &track)
{
  const auto flags = stream.flags();
  stream << std::setw(5) << track.pdg << std::scientific << std::setw(15) << std::setprecision(6) << track.energy
         << " (" << std::setprecision(2) << std::setw(9) << track.position[0] << std::setw(9) << track.position[1]
         << std::setw(9) << track.position[2] << ")";
  stream.flags(flags);
  return stream;
}
} // namespace

void AdeptIntegration::AddTrack(G4int threadId, G4int eventId, unsigned short cycleNumber, unsigned int trackId,
                                int pdg, double energy, double x, double y, double z, double dirx, double diry,
                                double dirz)
{
  if (pdg != 11 && pdg != -11 && pdg != 22) {
    G4cerr << __FILE__ << ":" << __LINE__ << ": Only supporting EM tracks. Got pdgID=" << pdg << "\n";
    return;
  }

  adeptint::TrackData track{threadId, eventId, trackId, pdg, energy, x, y, z, dirx, diry, dirz};
  if (fDebugLevel >= 2) {
    fGPUNetEnergy[threadId] += energy;
    if (fDebugLevel >= 5) {
      G4cout << "\n[_in," << eventId << "," << cycleNumber << "," << trackId << "]: " << track << "\tGPU net energy "
             << std::setprecision(6) << fGPUNetEnergy[threadId] << G4endl;
    }
  }

  // Lock buffer and emplace the track
  {
    auto trackHandle  = fBuffer->createToDeviceSlot();
    trackHandle.track = std::move(track);
  }

  fEventStates[threadId].store(EventState::NewTracksFromG4, std::memory_order_release);
}

void AdeptIntegration::Initialize()
{
  fNumVolumes = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  // We set the number of sensitive volumes equal to the number of placed volumes. This is temporary
  fNumSensitive = vecgeom::GeoManager::Instance().GetPlacedVolumesCount();
  if (fNumVolumes == 0) throw std::runtime_error("AdeptIntegration::Initialize: Number of geometry volumes is zero.");

    G4cout << "=== AdeptIntegration: initializing geometry and physics\n";
    // Initialize geometry on device
    if (!vecgeom::GeoManager::Instance().IsClosed())
      throw std::runtime_error("AdeptIntegration::Initialize: VecGeom geometry not closed.");

    const vecgeom::cxx::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
    if (!InitializeGeometry(world))
      throw std::runtime_error("AdeptIntegration::Initialize: Cannot initialize geometry on GPU");

    // Initialize G4HepEm
    if (!InitializePhysics()) throw std::runtime_error("AdeptIntegration::Initialize cannot initialize physics on GPU");

    // Do the material-cut couple index mapping once
    // as well as set flags for sensitive volumes and region
    VolAuxData *auxData = CreateVolAuxData(
        G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume(),
        vecgeom::GeoManager::Instance().GetWorld(), *fg4hepem_state);

    // Initialize volume auxiliary data on device
    VolAuxArray::GetInstance().fNumVolumes = fNumVolumes;
    VolAuxArray::GetInstance().fAuxData    = auxData;
    VolAuxArray::GetInstance().InitializeOnGPU();

    G4cout << "=== AdeptIntegration: initializing transport engine for thread: " << G4Threading::G4GetThreadId()
           << G4endl;

    // Initialize user scoring data
    fScoring = std::vector<AdeptScoring>(fNThread, fNumSensitive);

    // Initialize the transport engine for the current thread
    InitializeGPU();

    fGPUWorker = std::thread{&AdeptIntegration::TransportLoop, this};
}
void AdeptIntegration::InitBVH()
{
  vecgeom::cxx::BVHManager::Init();
  vecgeom::cxx::BVHManager::DeviceInit();
}

void AdeptIntegration::Flush(G4int threadId, G4int eventId, unsigned short cycleNumber)
{
  if (fDebugLevel >= 3) {
      G4cout << "\nFlushing AdePT for event " << eventId << " wave " << cycleNumber << G4endl;
  }

  assert(static_cast<unsigned int>(threadId) < fBuffer->fromDeviceBuffers.size());
  fEventStates[threadId].store(EventState::G4Flush, std::memory_order_release);

  std::vector<adeptint::TrackData> tracks;
  if (fEventStates[threadId].load(std::memory_order_acquire) < EventState::LeakedTracksRetrieved) {
      std::unique_lock lock{fBuffer->fromDeviceMutex};
      fBuffer->cv_fromDevice.wait(lock, [this, threadId]() {
        return fEventStates[threadId].load(std::memory_order_acquire) == EventState::DeviceFlushed;
      });
      tracks = std::move(fBuffer->fromDeviceBuffers[threadId]);
  }
  fEventStates[threadId].store(EventState::LeakedTracksRetrieved, std::memory_order_release);

  // TODO: Sort tracks on device?
  assert(std::all_of(tracks.begin(), tracks.end(),
                     [threadId](adeptint::TrackData const &a) { return a.threadId == threadId; }));
  std::sort(tracks.begin(), tracks.end());

  constexpr double tolerance = 10. * vecgeom::kTolerance;

  // Build the secondaries and put them back on the Geant4 stack
  const auto oldEnergyTransferred = fGPUNetEnergy[threadId];
  unsigned int trackId            = 0;
  for (const auto &track : tracks) {
      assert(eventId == track.eventId);

      if (fDebugLevel >= 2) {
      fGPUNetEnergy[threadId] -= track.energy;
      if (fDebugLevel >= 5) {
        G4cout << "\n[out," << track.eventId << "," << cycleNumber << "," << trackId++ << "]: " << track
               << "\tGPU net energy " << std::setprecision(6) << fGPUNetEnergy[threadId] << G4endl;
      }
      }

      G4ParticleMomentum direction(track.direction[0], track.direction[1], track.direction[2]);

      G4DynamicParticle *dynamique =
          new G4DynamicParticle(G4ParticleTable::GetParticleTable()->FindParticle(track.pdg), direction, track.energy);

      G4ThreeVector posi(track.position[0], track.position[1], track.position[2]);
      // The returned track will be located by Geant4. For now we need to
      // push it to make sure it is not relocated again in the GPU region
      posi += tolerance * direction;

      G4Track *secondary = new G4Track(dynamique, 0, posi);
      secondary->SetParentID(-99);

      G4EventManager::GetEventManager()->GetStackManager()->PushOneTrack(secondary);
  }

  if (fDebugLevel >= 2) {
      std::stringstream str;
      str << "\n[" << eventId << "," << cycleNumber << "]: Pushed " << tracks.size() << " tracks to G4";
      str << "\tEnergy back to G4: " << std::setprecision(6)
          << (oldEnergyTransferred - fGPUNetEnergy[threadId]) / CLHEP::GeV << "\tGPU net energy "
          << std::setprecision(6) << fGPUNetEnergy[threadId] / CLHEP::GeV << " GeV";
      str << "\t(" << countTracks(11, tracks) << ", " << countTracks(-11, tracks) << ", " << countTracks(22, tracks)
          << ")";
      G4cout << str.str() << G4endl;
  }

  if (tracks.empty()) {
      AdeptScoring &scoring = fScoring[threadId];
      scoring.CopyHitsToHost();
      scoring.ClearGPU();
      fGPUNetEnergy[threadId] = 0.;

      if (fDebugLevel >= 2) {
      G4cout << "\n\tScoring for event " << eventId << " cycle " << cycleNumber << G4endl;
      scoring.Print();
      }

      // Create energy deposit in the detector
      auto *sd                            = G4SDManager::GetSDMpointer()->FindSensitiveDetector("AdePTDetector");
      SensitiveDetector *fastSimSensitive = static_cast<SensitiveDetector *>(sd);

      for (auto id = 0; id != fNumSensitive; id++) {
        // here I add the energy deposition to the pre-existing Geant4 hit based on id
      fastSimSensitive->ProcessHits(id, scoring.fScoringPerVolume.energyDeposit[id] / copcore::units::MeV);
      }

      EventAction *evAct = static_cast<EventAction *>(G4EventManager::GetEventManager()->GetUserEventAction());
      evAct->number_gammas += scoring.fGlobalScoring.numGammas;
      evAct->number_electrons += scoring.fGlobalScoring.numElectrons;
      evAct->number_positrons += scoring.fGlobalScoring.numPositrons;
      evAct->number_killed += scoring.fGlobalScoring.numKilled;

      fEventStates[threadId].store(EventState::ScoringRetrieved, std::memory_order_release);
  }
}

namespace {

bool isInRegion(G4LogicalVolume const *volume, G4Region const *const region,
                std::unordered_map<G4LogicalVolume const *, bool> &resultCache)
{
  if (volume->GetRegion() == region) {
      return true;
  }
  // Maybe the volume is already known:
  if (auto resultIt = resultCache.find(volume); resultIt != resultCache.end()) {
      return resultIt->second;
  }

  // Visit all parent regions:
  for (const auto parent : *G4LogicalVolumeStore::GetInstance()) {
      const auto nDaughter = parent->GetNoDaughters();
      for (unsigned int i = 0; i < nDaughter; ++i) {
      if (parent->GetDaughter(i)->GetLogicalVolume() == volume) {
        if (isInRegion(parent, region, resultCache)) {
          std::cout << "Adding volume " << volume->GetName() << " to GPU region because parent is " << parent->GetName()
                    << " in region " << parent->GetRegion()->GetName() << "\n";
          resultCache[volume] = true;
          return true;
        }
      }
      }
  }

  resultCache[volume] = false;
  return false;
};

struct VisitHelpers {
  int nphysical      = 0;
  int nlogical_sens  = 0;
  int nphysical_sens = 0;
  int ninregion      = 0;
  std::unordered_map<G4LogicalVolume const *, bool> regionCache;
};
void visitAndSetMCIndex(G4VPhysicalVolume const *g4pvol, vecgeom::VPlacedVolume const *pvol, VisitHelpers &helpers,
                        const G4HepEmState &hepEmState, adeptint::VolAuxData *auxData, G4Region const *adeptRegion,
                        std::unordered_map<std::string, int> const &sensitive_volume_index,
                        std::unordered_map<const G4VPhysicalVolume *, int> &fScoringMap)
{
  const int *g4tohepmcindex = hepEmState.fData->fTheMatCutData->fG4MCIndexToHepEmMCIndex;
  const auto nvolumes       = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  const auto g4vol          = g4pvol->GetLogicalVolume();
  const auto vol            = pvol->GetLogicalVolume();
  const int nd              = g4vol->GetNoDaughters();
  auto daughters            = vol->GetDaughters();
  if (static_cast<std::size_t>(nd) != daughters.size())
      throw std::runtime_error("Fatal: CreateVolAuxData: Mismatch in number of daughters");
  // Check if transformations are matching
  auto g4trans = g4pvol->GetTranslation();
  auto g4rot   = g4pvol->GetRotation();
  G4RotationMatrix idrot;
  auto vgtransformation  = pvol->GetTransformation();
  constexpr double epsil = 1.e-8;
  for (int i = 0; i < 3; ++i) {
      if (std::abs(g4trans[i] - vgtransformation->Translation(i)) > epsil)
        throw std::runtime_error(
            std::string("Fatal: CreateVolAuxData: Mismatch between Geant4 translation for physical volume") +
            pvol->GetName());
  }

  // check if VecGeom and Geant4 (local) transformations are matching. Not optimized, this will re-check
  // already checked placed volumes when re-visiting the same volumes in different branches
  if (!g4rot) g4rot = &idrot;
  for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        int i = row + 3 * col;
        if (std::abs((*g4rot)(row, col) - vgtransformation->Rotation(i)) > epsil)
          throw std::runtime_error(
              std::string("Fatal: CreateVolAuxData: Mismatch between Geant4 rotation for physical volume") +
              pvol->GetName());
      }
  }

  // Check the couples
  if (g4vol->GetMaterialCutsCouple() == nullptr)
      throw std::runtime_error("Fatal: CreateVolAuxData: G4LogicalVolume " + std::string(g4vol->GetName()) +
                               std::string(" has no material-cuts couple"));
  int g4mcindex    = g4vol->GetMaterialCutsCouple()->GetIndex();
  int hepemmcindex = g4tohepmcindex[g4mcindex];
  // Check consistency with G4HepEm data
  if (hepEmState.fData->fTheMatCutData->fMatCutData[hepemmcindex].fG4MatCutIndex != g4mcindex)
      throw std::runtime_error(
          "Fatal: CreateVolAuxData: Mismatch between Geant4 mcindex and corresponding G4HepEm index");
  if (vol->id() >= nvolumes)
      throw std::runtime_error("Fatal: CreateVolAuxData: Volume id larger than number of volumes");

  // All OK, now fill the MCC index in the array
  auxData[vol->id()].fMCIndex = hepemmcindex;
  helpers.nphysical++;

  // Check if the volume belongs to the interesting region
  if (isInRegion(g4vol, adeptRegion, helpers.regionCache)) {
      auxData[vol->id()].fGPUregion = 1;
      helpers.ninregion++;
  }

  // Check if the logical volume is sensitive
  bool sens = false;
  for (auto sensvol : sensitive_volume_index) {
      if (vol->GetName() == sensvol.first || std::string(vol->GetName()).rfind(sensvol.first + "0x", 0) == 0) {
        sens = true;
        if (g4vol->GetSensitiveDetector() == nullptr)
          throw std::runtime_error("Fatal: CreateVolAuxData: G4LogicalVolume " + std::string(g4vol->GetName()) +
                                   " not sensitive while VecGeom one " + std::string(vol->GetName()) + " is.");
        if (auxData[vol->id()].fSensIndex < 0) helpers.nlogical_sens++;
        auxData[vol->id()].fSensIndex = sensvol.second;
        fScoringMap.insert(std::pair<const G4VPhysicalVolume *, int>(g4pvol, pvol->id()));
        helpers.nphysical_sens++;
        break;
      }
  }

  if (!sens && g4vol->GetSensitiveDetector() != nullptr)
      throw std::runtime_error("Fatal: CreateVolAuxData: G4LogicalVolume " + std::string(g4vol->GetName()) +
                               " sensitive while VecGeom one " + std::string(vol->GetName()) + " isn't.");

  // Now do the daughters
  for (int id = 0; id < nd; ++id) {
      auto g4pvol_d = g4vol->GetDaughter(id);
      auto pvol_d   = daughters[id];

      // VecGeom does not strip pointers from logical volume names
      if (std::string(pvol_d->GetLogicalVolume()->GetName()).rfind(g4pvol_d->GetLogicalVolume()->GetName(), 0) != 0)
        throw std::runtime_error("Fatal: CreateVolAuxData: Volume names " +
                                 std::string(pvol_d->GetLogicalVolume()->GetName()) + " and " +
                                 std::string(g4pvol_d->GetLogicalVolume()->GetName()) + " mismatch");
      visitAndSetMCIndex(g4pvol_d, pvol_d, helpers, hepEmState, auxData, adeptRegion, sensitive_volume_index,
                         fScoringMap);
  }
}
} // namespace

adeptint::VolAuxData *AdeptIntegration::CreateVolAuxData(const G4VPhysicalVolume *g4world,
                                                         const vecgeom::VPlacedVolume *world,
                                                         const G4HepEmState &hepEmState)
{
  // - FIND vecgeom::LogicalVolume corresponding to each and every G4LogicalVolume
  VisitHelpers counters;

  const auto nvolumes = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  VolAuxData *auxData = new VolAuxData[nvolumes];

  // recursive geometry visitor lambda matching one by one Geant4 and VecGeom logical volumes
  // (we need to make sure we set the right MCC index to the right volume)
  visitAndSetMCIndex(g4world, world, counters, hepEmState, auxData, fRegion, sensitive_volume_index, fScoringMap);

  G4cout << "Visited " << counters.nphysical << " matching physical volumes\n";
  G4cout << "Number of logical sensitive:      " << counters.nlogical_sens << "\n";
  G4cout << "Number of physical sensitive:     " << counters.nphysical_sens << "\n";
  G4cout << "Number of physical in GPU region: " << counters.ninregion << "\n";
  return auxData;
}
