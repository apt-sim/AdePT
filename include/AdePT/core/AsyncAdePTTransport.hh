// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

/// The The Geant4 AdePT integration service. This provides the interfaces for:
/// - initializing the geometry and physics on the AdePT size
/// - filling the buffer with tracks to be transported on the GPU
/// - Calling the Shower method transporting a buffer on the GPU

#ifndef ASYNC_ADEPT_TRANSPORT_HH
#define ASYNC_ADEPT_TRANSPORT_HH

#define ADEPT_SAVE_IDs
// #include "TrackTransfer.h"
// #include "BasicScoring.h"

#include <AdePT/core/AdePTTransportInterface.hh>
#include <AdePT/core/CommonStruct.h>
#include <AdePT/core/AdePTConfiguration.hh>
// #include <AdePT/core/PerEventScoringImpl.cuh>

#include <VecGeom/base/Config.h>
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume

#include <condition_variable>
#include <mutex>
#include <memory>
#include <thread>
#include <unordered_map>

struct AdePTScoring;

class G4Region;
class G4VPhysicalVolume;
struct G4HepEmState;
// class AdePTGeant4Integration;
namespace AsyncAdePT {
struct TrackBuffer;
struct GPUstate;
struct HitProcessingContext;
void InitVolAuxArray(adeptint::VolAuxArray &array);

template <typename IntegrationLayer>
class AsyncAdePTTransport : public AdePTTransportInterface {
public:
  static constexpr int kMaxThreads = 256;
  static inline uint64_t fAdePTSeed = 1234567;

  enum class EventState : unsigned char {
    NewTracksFromG4,
    G4RequestsFlush,
    Inject,
    InjectionCompleted,
    Transporting,
    WaitingForTransportToFinish,
    RequestHitFlush,
    FlushingHits,
    HitsFlushed,
    FlushingTracks,
    DeviceFlushed,
    LeakedTracksRetrieved,
    ScoringRetrieved
  };

private:
  unsigned short fNThread{0};       ///< Number of G4 workers
  unsigned int fTrackCapacity{0};   ///< Number of track slots to allocate on device
  unsigned int fScoringCapacity{0}; ///< Number of hit slots to allocate on device
  int fDebugLevel{1};               ///< Debug level
  int fCUDAStackLimit{0};                              ///< CUDA device stack limit
  std::vector<IntegrationLayer> fIntegrationLayerObjects;
  // We can't use a unique_ptr because we can't have the definition of GPUState
  // in a file compiled by gcc
  GPUstate *fGPUstate;               ///< CUDA state placeholder
  // We need to use pointers to the scoring objects because we can't have the complete definition 
  // in a file compiled by gcc
  std::vector<AdePTScoring*> fScoring;               ///< User scoring objects per G4 worker
  std::unique_ptr<TrackBuffer> fBuffer{nullptr};     ///< Buffers for transferring tracks between host and device
  std::unique_ptr<G4HepEmState> fg4hepem_state;      ///< The HepEm state singleton
  std::thread fGPUWorker;                            ///< Thread to manage GPU
  std::condition_variable fCV_G4Workers;             ///< Communicate with G4 workers
  std::mutex fMutex_G4Workers;                       ///< Mutex associated to the condition variable
  std::vector<std::atomic<EventState>> fEventStates; ///< State machine for each G4 worker
  std::vector<double> fGPUNetEnergy;
  bool fTrackInAllRegions = false;
  std::vector<std::string> const *fGPURegionNames;

  void Initialize();
  void InitBVH();
  bool InitializeField(double bz);
  bool InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world);
  bool InitializePhysics();
  // void InitializeGPU();
  // void FreeGPU();
  // /// @brief Asynchronous loop for transporting particles on GPU.
  // void TransportLoop();
  // void HitProcessingLoop(HitProcessingContext *const);
  // void ReturnTracksToG4();
  // void AdvanceEventStates(EventState oldState, EventState newState);
  // std::shared_ptr<const std::vector<GPUHit>> GetGPUHits(unsigned int threadId) const;

public:
  AsyncAdePTTransport(AdePTConfiguration &configuration);
  AsyncAdePTTransport(const AsyncAdePTTransport &other) = delete;
  ~AsyncAdePTTransport();

  /// @brief Adds a track to the buffer
  void AddTrack(int pdg, int parentID, double energy, double x, double y, double z, double dirx, double diry,
                double dirz, double globalTime, double localTime, double properTime, int threadId, unsigned int eventId,
                unsigned int trackIndex) override;
  /// @brief Set track capacity on GPU
  void SetTrackCapacity(size_t capacity) override { fTrackCapacity = capacity; }
  /// @brief Set Hit buffer capacity on GPU and Host
  virtual void SetHitBufferCapacity(size_t capacity) override { fScoringCapacity = capacity; }
  /// No effect
  void SetBufferThreshold(int) override {}
  /// No effect
  void SetMaxBatch(int) override {}
  /// @brief Set debug level for transport
  void SetDebugLevel(int level) override { fDebugLevel = level; }
  void SetTrackInAllRegions(bool trackInAllRegions) override { fTrackInAllRegions = trackInAllRegions; }
  bool GetTrackInAllRegions() const override { return fTrackInAllRegions; }
  void SetGPURegionNames(std::vector<std::string> const *regionNames) override { fGPURegionNames = regionNames; }
  void SetCUDAStackLimit(int limit) override{};
  std::vector<std::string> const *GetGPURegionNames() override { return fGPURegionNames; }
  /// No effect
  void Initialize(bool) override {}
  /// @brief Finish GPU transport, bring hits and tracks to host
  /// @details The shower call exists to maintain the same interface as the 
  /// synchronous AdePT mode, since in this case the transport loop is always 
  /// running. The only call to Shower() from G4 is done when the tracking 
  /// manager needs to flush an event.
  void Shower(int event, int threadId) override { Flush(threadId, event); }
  /// Block until transport of the given event is done.
  void Flush(int threadId, int eventId);
  void Cleanup() override {}
};

} // namespace AsyncAdePT

#include "AsyncAdePTTransport.icc"

#endif