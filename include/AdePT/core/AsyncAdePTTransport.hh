// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

/// The The Geant4 AdePT integration service. This provides the interfaces for:
/// - initializing the geometry and physics on the AdePT size
/// - filling the buffer with tracks to be transported on the GPU
/// - Calling the Shower method transporting a buffer on the GPU

#ifndef ASYNC_ADEPT_TRANSPORT_HH
#define ASYNC_ADEPT_TRANSPORT_HH

#include <AdePT/core/AdePTConfiguration.hh>
#include <AdePT/core/AsyncAdePTTransportStruct.hh>
#include <AdePT/core/CommonStruct.h>
#include <AdePT/integration/AdePTGeant4Integration.hh>
#include <AdePT/integration/G4HepEmTrackingManagerSpecialized.hh>

#include <VecGeom/base/Config.h>
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume

#include <condition_variable>
#include <mutex>
#include <memory>
#include <thread>
#include <unordered_map>
#include <optional>

class G4Region;
class G4VPhysicalVolume;
struct G4HepEmState;
namespace AsyncAdePT {
struct TrackBuffer;
struct GPUstate;

void InitVolAuxArray(adeptint::VolAuxArray &array);

class AsyncAdePTTransport {
public:
  uint64_t fAdePTSeed = 1234567;

private:
  unsigned short fNThread{0};             ///< Number of G4 workers
  unsigned int fTrackCapacity{0};         ///< Number of track slots to allocate on device
  unsigned int fLeakCapacity{0};          ///< Number of leak slots to allocate on device
  unsigned int fScoringCapacity{0};       ///< Number of hit slots to allocate on device
  int fDebugLevel{0};                     ///< Debug level
  int fCUDAStackLimit{0};                 ///< CUDA device stack limit
  int fCUDAHeapLimit{0};                  ///< CUDA device heap limit
  unsigned short fLastNParticlesOnCPU{0}; ///< Number N of last N particles that are finished on CPU
  unsigned short fMaxWDTIter{5};          ///< Maximum number of Woodcock tracking iterations per step
  std::unique_ptr<GPUstate, GPUstateDeleter> fGPUstate{nullptr}; ///< CUDA state placeholder
  std::unique_ptr<TrackBuffer> fBuffer{nullptr};     ///< Buffers for transferring tracks between host and device
  std::unique_ptr<G4HepEmState> fg4hepem_state;      ///< The HepEm state singleton
  adeptint::WDTDeviceBuffers fWDTDev{};              ///< device buffers for Woodcock tracking data
  std::thread fGPUWorker;                            ///< Thread to manage GPU
  std::condition_variable fCV_G4Workers;             ///< Communicate with G4 workers
  std::mutex fMutex_G4Workers;                       ///< Mutex associated to the condition variable
  std::vector<std::atomic<EventState>> fEventStates; ///< State machine for each G4 worker
  std::vector<double> fGPUNetEnergy;
  bool fTrackInAllRegions = false;
  bool fHasWDTRegions     = false;
  std::vector<std::string> const *fGPURegionNames;
  std::vector<std::string> const *fCPURegionNames;
  // Flags for the kernels to return the last or all steps, needed for PostUserTrackingAction or UserSteppingAction
  bool fReturnAllSteps         = false;
  bool fReturnFirstAndLastStep = false;
  std::string fBfieldFile{""}; ///< Path to magnetic field file (in the covfie format)
  double fCPUCapacityFactor{
      2.5}; ///< Factor by which the ScoringCapacity on Host is larger than on Device. Must be at least 2
  ///< Filling fraction of the ScoringCapacity on host when the hits are copied out and not taken directly by the
  ///< G4workers
  double fCPUCopyFraction{0.5};
  ///< Needed to stall the GPU, in case the nPartInFlight * fHitBufferSafetyFactor > available HitSlots
  double fHitBufferSafetyFactor{1.5};

  void Initialize(G4HepEmTrackingManagerSpecialized *hepEmTM, AdePTGeant4Integration &g4Integration,
                  const std::vector<float> &uniformFieldValues);
  void InitBVH();
  bool InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world);
  bool InitializePhysics(G4HepEmConfig *hepEmConfig);
  void InitWDTOnDevice(const adeptint::WDTHostPacked &src, adeptint::WDTDeviceBuffers &dev, unsigned short maxIter);

public:
  AsyncAdePTTransport(AdePTConfiguration &configuration, G4HepEmTrackingManagerSpecialized *hepEmTM,
                      AdePTGeant4Integration &g4Integration, const std::vector<float> &uniformFieldValues);
  AsyncAdePTTransport(const AsyncAdePTTransport &other) = delete;
  ~AsyncAdePTTransport();

  /// @brief Adds a track to the buffer
  void AddTrack(int pdg, uint64_t trackId, uint64_t parentId, double energy, double x, double y, double z, double dirx,
                double diry, double dirz, double globalTime, double localTime, double properTime, float weight,
                unsigned short stepCounter, int threadId, unsigned int eventId, vecgeom::NavigationState &&state);
  bool GetTrackInAllRegions() const { return fTrackInAllRegions; }
  bool GetCallUserActions() const { return fReturnFirstAndLastStep; }
  std::vector<std::string> const *GetGPURegionNames() { return fGPURegionNames; }
  std::vector<std::string> const *GetCPURegionNames() { return fCPURegionNames; }
  /// Block until transport of the given event is done.
  void Flush(int threadId, int eventId, AdePTGeant4Integration &g4Integration);
  void ProcessGPUSteps(int threadId, int eventId, AdePTGeant4Integration &g4Integration);
};

} // namespace AsyncAdePT

#include "AsyncAdePTTransport.icc"

#endif
