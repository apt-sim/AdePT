// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

/// The AdePT transport service. This provides the interfaces for:
/// - initializing geometry and physics on the AdePT side
/// - filling the buffer with tracks to be transported on the GPU
/// - driving the GPU transport worker

#ifndef ASYNC_ADEPT_TRANSPORT_HH
#define ASYNC_ADEPT_TRANSPORT_HH

#include <AdePT/transport/g4hepem/AdePTG4HepEmState.hh>
#include <AdePT/transport/AsyncAdePTTransportStruct.hh>
#include <AdePT/transport/config/AdePTTransportConfig.hh>
#include <AdePT/transport/geometry/GeometryAuxData.hh>
#include <AdePT/transport/steps/GPUStep.hh>

#include <VecGeom/base/Config.h>
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume

#include <condition_variable>
#include <mutex>
#include <memory>
#include <span>
#include <thread>
#include <unordered_map>
#include <optional>
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
  unsigned int fStepCapacity{0};          ///< Number of step slots to allocate on device
  int fDebugLevel{0};                     ///< Debug level
  int fCUDAStackLimit{0};                 ///< CUDA device stack limit
  int fCUDAHeapLimit{0};                  ///< CUDA device heap limit
  unsigned short fLastNParticlesOnCPU{0}; ///< Number N of last N particles that are finished on CPU
  unsigned short fMaxWDTIter{5};          ///< Maximum number of Woodcock tracking iterations per step
  std::unique_ptr<GPUstate, GPUstateDeleter> fGPUstate{nullptr}; ///< CUDA state placeholder
  std::unique_ptr<TrackBuffer> fBuffer{nullptr}; ///< Buffers for transferring tracks between host and device
  std::unique_ptr<AdePTG4HepEmState>
      fAdePTG4HepEmState;               ///< Transport-owned wrapper around `G4HepEmData` and copied `G4HepEmParameters`
  adeptint::WDTDeviceBuffers fWDTDev{}; ///< device buffers for Woodcock tracking data
  std::thread fGPUWorker;               ///< Thread to manage GPU
  std::condition_variable fCV_G4Workers;             ///< Communicate with G4 workers
  std::mutex fMutex_G4Workers;                       ///< Mutex associated to the condition variable
  std::vector<std::atomic<EventState>> fEventStates; ///< State machine for each G4 worker
  bool fHasWDTRegions = false;
  // Flags for the kernels to return the last or all steps, needed for PostUserTrackingAction or UserSteppingAction
  bool fReturnAllSteps         = false;
  bool fReturnFirstAndLastStep = false;
  std::string fBfieldFile{""}; ///< Path to magnetic field file (in the covfie format)
  double fCPUCapacityFactor{
      2.5}; ///< Factor by which the host step buffer is larger than the device step buffer. Must be at least 2
  ///< Filling fraction of the host step buffer when the steps are copied out and not taken directly by the G4 workers
  double fCPUCopyFraction{0.5};
  ///< Needed to stall the GPU, in case the nPartInFlight * fStepBufferSafetyFactor > available StepSlots
  double fStepBufferSafetyFactor{1.5};

  void Initialize(adeptint::VolAuxData *auxData, const adeptint::WDTHostPacked &wdtPacked,
                  const std::vector<float> &uniformFieldValues);
  void InitBVH();
  bool InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world);
  bool InitializePhysics();

public:
  AsyncAdePTTransport(const AdePTTransportConfig &configuration, std::unique_ptr<AdePTG4HepEmState> adeptG4HepEmState,
                      adeptint::VolAuxData *auxData, const adeptint::WDTHostPacked &wdtPacked,
                      const std::vector<float> &uniformFieldValues);
  AsyncAdePTTransport(const AsyncAdePTTransport &other) = delete;
  ~AsyncAdePTTransport();

  /// @brief Adds a track to the buffer
  void AddTrack(int pdg, uint64_t trackId, uint64_t parentId, double energy, double x, double y, double z, double dirx,
                double diry, double dirz, double globalTime, double localTime, double properTime, float weight,
                unsigned short stepCounter, int threadId, unsigned int eventId, vecgeom::NavigationState &&state);
  int GetDebugLevel() const { return fDebugLevel; }
  /// @brief Handle the currently available returned GPU-step batches for one thread and event.
  /// @details
  /// Transport retains ownership of the step-buffer lifetime. For each available
  /// batch, `callback` is invoked with a `std::span<const GPUStep>` view and the
  /// batch is released again when the callback returns.
  ///
  /// In this code path, the callback is the `AdePTTrackingManager` logic that
  /// reconstructs Geant4 steps from the returned GPU steps.
  template <typename Callback>
  void HandleGPUStepBatchesWith(int threadId, int eventId, Callback &&callback);
  /// @brief Request that the device flush all pending work for the given worker.
  void RequestFlush(int threadId);
  /// @brief Wait until the transport threads make further flush progress.
  void WaitForFlushProgress();
  /// @brief Check whether all returned GPU-step batches have been made available on the host.
  bool AreReturnedStepsFlushed(int threadId) const;
  /// @brief Mark the worker event as fully handled on the host side after replaying the returned steps.
  void MarkHostFlushed(int threadId);
};

} // namespace AsyncAdePT

#include "AsyncAdePTTransport.icc"

#endif
