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

#include "SensitiveDetector.hh"
#include "EventAction.hh"

AdeptIntegration::~AdeptIntegration()
{
  delete fScoring;
}

void AdeptIntegration::AddTrack(int pdg, double energy, double x, double y, double z, double dirx, double diry,
                                double dirz)
{
  fBuffer.toDevice.emplace_back(pdg, energy, x, y, z, dirx, diry, dirz);
  if (pdg == 11)
    fBuffer.nelectrons++;
  else if (pdg == -11)
    fBuffer.npositrons++;
  else if (pdg == 22)
    fBuffer.ngammas++;

  if (fBuffer.toDevice.size() >= fBufferThreshold) {
    if (fDebugLevel > 0)
      G4cout << "Reached the threshold of " << fBufferThreshold << " triggering the shower" << G4endl;
    this->Shower(G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID());
  }
}

void AdeptIntegration::Initialize(bool common_data)
{
  if (fInit) return;
  if (fMaxBatch <= 0) throw std::runtime_error("AdeptIntegration::Initialize: Maximum batch size not set.");

  fNumVolumes = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  // We set the number of sensitive volumes equal to the number of placed volumes. This is temporary
  fNumSensitive = vecgeom::GeoManager::Instance().GetPlacedVolumesCount();
  if (fNumVolumes == 0) throw std::runtime_error("AdeptIntegration::Initialize: Number of geometry volumes is zero.");

  if (common_data) {
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
    int *sensitive_volumes = nullptr;

    VolAuxData *auxData = CreateVolAuxData(
        G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume(),
        vecgeom::GeoManager::Instance().GetWorld(), *fg4hepem_state);

    // Initialize volume auxiliary data on device
    VolAuxArray::GetInstance().fNumVolumes = fNumVolumes;
    VolAuxArray::GetInstance().fAuxData    = auxData;
    VolAuxArray::GetInstance().InitializeOnGPU();
    return;
  }

  G4cout << "=== AdeptIntegration: initializing transport engine for thread: " << G4Threading::G4GetThreadId()
         << G4endl;

  // Initialize user scoring data
  fScoring     = new AdeptScoring(fNumSensitive);
  fScoring_dev = fScoring->InitializeOnGPU();

  // Initialize the transport engine for the current thread
  InitializeGPU();

  fInit = true;
}
void AdeptIntegration::InitBVH()
{
  vecgeom::cxx::BVHManager::Init();
  vecgeom::cxx::BVHManager::DeviceInit();
}

void AdeptIntegration::Cleanup()
{
  if (!fInit) return;
  AdeptIntegration::FreeGPU();
}

void AdeptIntegration::Shower(int event)
{
  constexpr double tolerance = 10. * vecgeom::kTolerance;

  int tid = G4Threading::G4GetThreadId();
  if (fDebugLevel > 0 && fBuffer.toDevice.size() == 0) {
    G4cout << "[" << tid << "] AdeptIntegration::Shower: No more particles in buffer. Exiting.\n";
    return;
  }

  if (event != fBuffer.eventId) {
    fBuffer.eventId    = event;
    fBuffer.startTrack = 0;
  } else {
    fBuffer.startTrack += fBuffer.toDevice.size();
  }

  int itr   = 0;
  int nelec = 0, nposi = 0, ngamma = 0;
  if (fDebugLevel > 0) {
    G4cout << "[" << tid << "] toDevice: " << fBuffer.nelectrons << " elec, " << fBuffer.npositrons << " posi, "
           << fBuffer.ngammas << " gamma\n";
  }
  if (fDebugLevel > 1) {
    for (auto &track : fBuffer.toDevice) {
      G4cout << "[" << tid << "] toDevice[ " << itr++ << "]: pdg " << track.pdg << " energy " << track.energy
             << " position " << track.position[0] << " " << track.position[1] << " " << track.position[2]
             << " direction " << track.direction[0] << " " << track.direction[1] << " " << track.direction[2] << G4endl;
    }
  }

  AdeptIntegration::ShowerGPU(event, fBuffer);

  if (fDebugLevel > 1) fScoring->Print();
  for (int i = 0; i < fBuffer.numFromDevice; ++i) {
    const auto &track = fBuffer.fromDevice[i];
    if (track.pdg == 11)
      nelec++;
    else if (track.pdg == -11)
      nposi++;
    else if (track.pdg == 22)
      ngamma++;
  }
  if (fDebugLevel > 0) {
    G4cout << "[" << tid << "] fromDevice: " << nelec << " elec, " << nposi << " posi, " << ngamma << " gamma\n";
  }

  // Build the secondaries and put them back on the Geant4 stack
  for (int i = 0; i < fBuffer.numFromDevice; ++i) {
    const auto &track = fBuffer.fromDevice_sorted[i];
    if (fDebugLevel > 1) {
      G4cout << "[" << tid << "] fromDevice[ " << i << "]: pdg " << track.pdg << " energy " << track.energy
             << " position " << track.position[0] << " " << track.position[1] << " " << track.position[2]
             << " direction " << track.direction[0] << " " << track.direction[1] << " " << track.direction[2] << G4endl;
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

  // Create energy deposit in the detector
  auto *sd                            = G4SDManager::GetSDMpointer()->FindSensitiveDetector("AdePTDetector");
  SensitiveDetector *fastSimSensitive = dynamic_cast<SensitiveDetector *>(sd);

  for (auto id = 0; id != fNumSensitive; id++) {
    // here I add the energy deposition to the pre-existing Geant4 hit based on id
    fastSimSensitive->ProcessHits(id, fScoring->fScoringPerVolume.energyDeposit[id] / copcore::units::MeV);
  }

  EventAction *evAct      = dynamic_cast<EventAction *>(G4EventManager::GetEventManager()->GetUserEventAction());
  evAct->number_gammas    = evAct->number_gammas + fScoring->fGlobalScoring.numGammas;
  evAct->number_electrons = evAct->number_electrons + fScoring->fGlobalScoring.numElectrons;
  evAct->number_positrons = evAct->number_positrons + fScoring->fGlobalScoring.numPositrons;
  evAct->number_killed    = evAct->number_killed + fScoring->fGlobalScoring.numKilled;

  fBuffer.Clear();
  fScoring->ClearGPU();
}

adeptint::VolAuxData *AdeptIntegration::CreateVolAuxData(const G4VPhysicalVolume *g4world,
                                                         const vecgeom::VPlacedVolume *world,
                                                         const G4HepEmState &hepEmState)
{
  const int *g4tohepmcindex = hepEmState.fData->fTheMatCutData->fG4MCIndexToHepEmMCIndex;

  // - FIND vecgeom::LogicalVolume corresponding to each and every G4LogicalVolume
  int nphysical      = 0;
  int nlogical_sens  = 0;
  int nphysical_sens = 0;
  int ninregion      = 0;

  int nvolumes        = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  VolAuxData *auxData = new VolAuxData[nvolumes];

  // recursive geometry visitor lambda matching one by one Geant4 and VecGeom logical volumes
  // (we need to make sure we set the right MCC index to the right volume)
  typedef std::function<void(G4VPhysicalVolume const *, vecgeom::VPlacedVolume const *)> func_t;
  func_t visitAndSetMCindex = [&](G4VPhysicalVolume const *g4pvol, vecgeom::VPlacedVolume const *pvol) {
    const auto g4vol = g4pvol->GetLogicalVolume();
    const auto vol   = pvol->GetLogicalVolume();
    int nd           = g4vol->GetNoDaughters();
    auto daughters   = vol->GetDaughters();
    if (nd != daughters.size())
      throw std::runtime_error("Fatal: CreateVolAuxData: Mismatch in number of daughters");
    // Check if transformations are matching
    auto g4trans = g4pvol->GetTranslation();
    auto g4rot = g4pvol->GetRotation();
    G4RotationMatrix idrot;
    auto vgtransformation = pvol->GetTransformation();
    constexpr double epsil = 1.e-8;
    for (int i = 0; i<3; ++i) {
      if (std::abs(g4trans[i] - vgtransformation->Translation(i)) > epsil)
        throw std::runtime_error(std::string("Fatal: CreateVolAuxData: Mismatch between Geant4 translation for physical volume") + pvol->GetName());
    }

    // check if VecGeom and Geant4 (local) transformations are matching. Not optimized, this will re-check
    // already checked placed volumes when re-visiting the same volumes in different branches
    if (!g4rot) g4rot = &idrot;
    for (int row = 0; row<3; ++row) {
      for (int col = 0; col<3; ++col) {
        int i = row + 3*col;
        if (std::abs((*g4rot)(row,col) - vgtransformation->Rotation(i)) > epsil)
          throw std::runtime_error(std::string("Fatal: CreateVolAuxData: Mismatch between Geant4 rotation for physical volume") + pvol->GetName());
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
      throw std::runtime_error("Fatal: CreateVolAuxData: Mismatch between Geant4 mcindex and corresponding G4HepEm index");
    if (vol->id() >= nvolumes)
      throw std::runtime_error("Fatal: CreateVolAuxData: Volume id larger than number of volumes");

    // All OK, now fill the MCC index in the array
    auxData[vol->id()].fMCIndex = hepemmcindex;
    nphysical++;

    // Check if the volume belongs to the interesting region
    if (g4vol->GetRegion() == fRegion) {
      auxData[vol->id()].fGPUregion = 1;
      ninregion++;
    }

    // Check if the logical volume is sensitive
    bool sens = false;
    for (auto sensvol : (*sensitive_volume_index)) {
      if (vol->GetName() == sensvol.first ||
          std::string(vol->GetName()).rfind(sensvol.first + "0x", 0) == 0) {
        sens = true;
        if (g4vol->GetSensitiveDetector() == nullptr)
          throw std::runtime_error("Fatal: CreateVolAuxData: G4LogicalVolume " + std::string(g4vol->GetName()) +
                                   " not sensitive while VecGeom one " + std::string(vol->GetName()) + " is.");
        if (auxData[vol->id()].fSensIndex < 0) nlogical_sens++;
        auxData[vol->id()].fSensIndex = sensvol.second;
        fScoringMap->insert(std::pair<const G4VPhysicalVolume *, int>(g4pvol, pvol->id()));
        nphysical_sens++;
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
        throw std::runtime_error("Fatal: CreateVolAuxData: Volume names " + std::string(pvol_d->GetLogicalVolume()->GetName()) +
                                 " and " + std::string(g4pvol_d->GetLogicalVolume()->GetName()) + " mismatch");
      visitAndSetMCindex(g4pvol_d, pvol_d);
    }
  };

  visitAndSetMCindex(g4world, world);

  G4cout << "Visited " << nphysical << " matching physical volumes\n";
  G4cout << "Number of logical sensitive:      " << nlogical_sens << "\n";
  G4cout << "Number of physical sensitive:     " << nphysical_sens << "\n";
  G4cout << "Number of physical in GPU region: " << ninregion << "\n";
  return auxData;
}
