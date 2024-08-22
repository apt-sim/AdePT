// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "AdeptIntegration.h"

#include "TrackTransfer.h"
#include "Histograms.h"

#include <AdePT/integration/AdePTGeant4Integration.hh>

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

#include <iomanip>

std::shared_ptr<AdePTTransportInterface> AdePTTransportFactory(unsigned int nThread, unsigned int nTrackSlot,
                                                               unsigned int nHitSlot, int verbosity,
                                                               std::vector<std::string> const *GPURegionNames,
                                                               bool trackInAllRegions)
{
  static std::shared_ptr<AsyncAdePT::AdeptIntegration> adePT{
      new AsyncAdePT::AdeptIntegration(nThread, nTrackSlot, nHitSlot, verbosity, GPURegionNames, trackInAllRegions)};
  return adePT;
}

namespace AsyncAdePT {

TrackBuffer::TrackHandle TrackBuffer::createToDeviceSlot()
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

namespace {
template <typename T>
std::size_t countTracks(int pdgToSelect, T const &container)
{
  return std::count_if(container.begin(), container.end(),
                       [pdgToSelect](TrackDataWithIDs const &track) { return track.pdg == pdgToSelect; });
}

std::ostream &operator<<(std::ostream &stream, TrackDataWithIDs const &track)
{
  const auto flags = stream.flags();
  stream << std::setw(5) << track.pdg << std::scientific << std::setw(15) << std::setprecision(6) << track.eKin << " ("
         << std::setprecision(2) << std::setw(9) << track.position[0] << std::setw(9) << track.position[1]
         << std::setw(9) << track.position[2] << ")";
  stream.flags(flags);
  return stream;
}
} // namespace

void AdeptIntegration::AddTrack(int pdg, int parentID, double energy, double x, double y, double z, double dirx,
                                double diry, double dirz, double globalTime, double localTime, double properTime,
                                int threadId, unsigned int eventId, unsigned int trackId)
{
  if (pdg != 11 && pdg != -11 && pdg != 22) {
    G4cerr << __FILE__ << ":" << __LINE__ << ": Only supporting EM tracks. Got pdgID=" << pdg << "\n";
    return;
  }

  TrackDataWithIDs track{pdg,       parentID,   energy,  x,       y,
                         z,         dirx,       diry,    dirz,    globalTime,
                         localTime, properTime, eventId, trackId, static_cast<short>(threadId)};
  if (fDebugLevel >= 2) {
    fGPUNetEnergy[threadId] += energy;
    if (fDebugLevel >= 6) {
      G4cout << "\n[_in," << eventId << "," << trackId << "]: " << track << "\tGPU net energy " << std::setprecision(6)
             << fGPUNetEnergy[threadId] << G4endl;
    }
  }

  // Lock buffer and emplace the track
  {
    auto trackHandle  = fBuffer->createToDeviceSlot();
    trackHandle.track = std::move(track);
  }

  fEventStates[threadId].store(EventState::NewTracksFromG4, std::memory_order_release);
}

void AdeptIntegration::FullInit()
{
  const auto numVolumes = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  if (numVolumes == 0) throw std::runtime_error("AdeptIntegration::Initialize: Number of geometry volumes is zero.");

  G4cout << "=== AdeptIntegration: initializing geometry and physics\n";
  // Initialize geometry on device
  if (!vecgeom::GeoManager::Instance().IsClosed())
    throw std::runtime_error("AdeptIntegration::Initialize: VecGeom geometry not closed.");

  const vecgeom::cxx::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (!InitializeGeometry(world))
    throw std::runtime_error("AdeptIntegration::Initialize: Cannot initialize geometry on GPU");

  // Initialize G4HepEm
  const double bz = fG4Integrations.front().GetUniformFieldZ();
  if (!InitializePhysics(bz)) throw std::runtime_error("AdeptIntegration::Initialize cannot initialize physics on GPU");

  // Check VecGeom geometry matches Geant4. Initialize auxiliary per-LV data. Initialize scoring map.
  fG4Integrations.front().CheckGeometry(fg4hepem_state.get());
  adeptint::VolAuxData *auxData = new adeptint::VolAuxData[vecgeom::GeoManager::Instance().GetRegisteredVolumesCount()];
  fG4Integrations.front().InitVolAuxData(auxData, fg4hepem_state.get(), fTrackInAllRegions, fGPURegionNames);

  // Initialize volume auxiliary data on device
  auto &volAuxArray       = adeptint::VolAuxArray::GetInstance();
  volAuxArray.fNumVolumes = numVolumes;
  volAuxArray.fAuxData    = auxData;
  AsyncAdePT::InitVolAuxArray(volAuxArray);

  for (auto &g4int : fG4Integrations) {
    g4int.InitScoringData(volAuxArray.fAuxData);
  }

  // Allocate buffers to transport particles to/from device. Scale the size of the staging area
  // with the number of threads.
  fBuffer = std::make_unique<TrackBuffer>(8192 * fNThread, 1024 * fNThread, fNThread);

  fGPUWorker = std::thread{&AdeptIntegration::TransportLoop, this};
}

void AdeptIntegration::InitBVH()
{
  vecgeom::cxx::BVHManager::Init();
  vecgeom::cxx::BVHManager::DeviceInit();
}

void AdeptIntegration::Flush(G4int threadId, G4int eventId)
{
  if (fDebugLevel >= 3) {
    G4cout << "\nFlushing AdePT for event " << eventId << G4endl;
  }

  auto xyHisto_count     = std::make_shared<TH2L>(("Event_" + std::to_string(eventId) + "_count_xy").c_str(),
                                              "Number of hits;x;y", 500, -2500, 2500, 500, -2500, 2500);
  auto xyHisto_e         = std::make_shared<TH2D>(("Event_" + std::to_string(eventId) + "_E_xy").c_str(),
                                          "Energy deposition;x;y", 500, -2500, 2500, 500, -2500, 2500);
  auto etaPhiHisto_count = std::make_shared<TH2L>(("Event_" + std::to_string(eventId) + "_count_etaPhi").c_str(),
                                                  "Number of hits;#phi;#eta", 100, -180, 180, 300, -15, 15);
  auto etaPhiHisto_e     = std::make_shared<TH2D>(("Event_" + std::to_string(eventId) + "_E_etaPhi").c_str(),
                                              "Energy deposition;#phi;#eta", 100, -180, 180, 300, -15, 15);
  AsyncExHistos::registerHisto(xyHisto_count);
  AsyncExHistos::registerHisto(xyHisto_e);
  AsyncExHistos::registerHisto(etaPhiHisto_count);
  AsyncExHistos::registerHisto(etaPhiHisto_e);

  assert(static_cast<unsigned int>(threadId) < fBuffer->fromDeviceBuffers.size());
  fEventStates[threadId].store(EventState::G4RequestsFlush, std::memory_order_release);

  AdePTGeant4Integration &g4Integration = fG4Integrations[threadId];

  while (fEventStates[threadId].load(std::memory_order_acquire) < EventState::DeviceFlushed) {
    {
      std::unique_lock lock{fMutex_G4Workers};
      fCV_G4Workers.wait(lock);
    }

    std::shared_ptr<const std::vector<GPUHit>> gpuHits;
    while ((gpuHits = GetGPUHits(threadId)) != nullptr) {
      GPUHit dummy;
      dummy.fEventId = eventId;
      auto range     = std::equal_range(gpuHits->begin(), gpuHits->end(), dummy,
                                        [](const GPUHit &lhs, const GPUHit &rhs) { return lhs.fEventId < rhs.fEventId; });
      for (auto it = range.first; it != range.second; ++it) {
        assert(it->threadId == threadId);
        const auto pos = it->fPostStepPoint.fPosition;
        xyHisto_e->Fill(pos.x(), pos.y(), it->fTotalEnergyDeposit);
        xyHisto_count->Fill(pos.x(), pos.y());
        G4ThreeVector posg4(pos.x(), pos.y(), pos.z());
        const auto phi = posg4.getPhi() / 2. / (2. * acos(0)) * 360;
        etaPhiHisto_e->Fill(phi, posg4.getEta(), it->fTotalEnergyDeposit);
        etaPhiHisto_count->Fill(phi, posg4.getEta());
        g4Integration.ProcessGPUHit(*it);
      }
    }
  }

  // Now device should be flushed, so retrieve the tracks:
  std::vector<TrackDataWithIDs> tracks;
  {
    auto handle = fBuffer->getTracksFromDevice(threadId);
    tracks.swap(handle.tracks);
    fEventStates[threadId].store(EventState::LeakedTracksRetrieved, std::memory_order_release);
  }

  // TODO: Sort tracks on device?
#ifndef NDEBUG
  for (auto const &track : tracks) {
    bool error = false;
    if (track.threadId != threadId || track.eventId != static_cast<unsigned int>(eventId)) error = true;
    if (!(track.pdg == -11 || track.pdg == 11 || track.pdg == 22)) error = true;
    if (error)
      std::cerr << "Error in returning track: threadId=" << track.threadId << " eventId=" << track.eventId
                << " pdg=" << track.pdg << "\n";
    assert(!error);
  }
#endif
  std::sort(tracks.begin(), tracks.end());

  const auto oldEnergyTransferred = fGPUNetEnergy[threadId];
  if (fDebugLevel >= 2) {
    unsigned int trackId = 0;
    for (const auto &track : tracks) {

      fGPUNetEnergy[threadId] -= track.eKin;
      if (fDebugLevel >= 5) {
        G4cout << "\n[out," << track.eventId << "," << trackId++ << "]: " << track << "\tGPU net energy "
               << std::setprecision(6) << fGPUNetEnergy[threadId] << G4endl;
      }
    }
  }

  if (fDebugLevel >= 2) {
    std::stringstream str;
    str << "\n[" << eventId << "]: Pushed " << tracks.size() << " tracks to G4";
    str << "\tEnergy back to G4: " << std::setprecision(6)
        << (oldEnergyTransferred - fGPUNetEnergy[threadId]) / CLHEP::GeV << "\tGPU net energy " << std::setprecision(6)
        << fGPUNetEnergy[threadId] / CLHEP::GeV << " GeV";
    str << "\t(" << countTracks(11, tracks) << ", " << countTracks(-11, tracks) << ", " << countTracks(22, tracks)
        << ")";
    G4cout << str.str() << G4endl;
  }

  if (tracks.empty()) {
    AsyncAdePT::PerEventScoring &scoring = fScoring[threadId];
    scoring.CopyToHost();
    scoring.ClearGPU();
    fGPUNetEnergy[threadId] = 0.;

    if (fDebugLevel >= 2) {
      G4cout << "\n\tScoring for event " << eventId << G4endl;
      scoring.Print();
    }

    fEventStates[threadId].store(EventState::ScoringRetrieved, std::memory_order_release);
  }

  g4Integration.ReturnTracks(tracks.begin(), tracks.end(), fDebugLevel);
}

} // namespace AsyncAdePT
