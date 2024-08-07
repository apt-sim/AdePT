// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

/// The The Geant4 AdePT integration service. This provides the interfaces for:
/// - initializing the geometry and physics on the AdePT size
/// - filling the buffer with tracks to be transported on the GPU
/// - Calling the Shower method transporting a buffer on the GPU

#ifndef ADEPT_INTEGRATION_H
#define ADEPT_INTEGRATION_H

#define ADEPT_SAVE_IDs
#include "TrackTransfer.h"
#include "BasicScoring.h"

#include <AdePT/core/AdePTTransportInterface.hh>
#include <AdePT/core/CommonStruct.h>

#include <VecGeom/base/Config.h>
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume

#include <condition_variable>
#include <mutex>
#include <memory>
#include <thread>
#include <unordered_map>

class G4Region;
class G4VPhysicalVolume;
class G4HepEmState;
class AdePTGeant4Integration;
namespace AsyncAdePT {
struct TrackBuffer;
struct GPUstate;
struct HitProcessingContext;
void InitVolAuxArray(adeptint::VolAuxArray &array);

class AdeptIntegration : public AdePTTransportInterface {
public:
  static constexpr int kMaxThreads = 256;

private:
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

  unsigned short fNThread{0};       ///< Number of G4 workers
  unsigned int fTrackCapacity{0};   ///< Number of track slots to allocate on device
  unsigned int fScoringCapacity{0}; ///< Number of hit slots to allocate on device
  int fDebugLevel{1};               ///< Debug level
  uint64_t fAdePTSeed{1234567};     ///< Seed multiplier for tracks going to GPU
  std::vector<AdePTGeant4Integration> fG4Integrations;
  std::unique_ptr<GPUstate> fGPUstate;               ///< CUDA state placeholder
  std::vector<PerEventScoring> fScoring;             ///< User scoring objects per G4 worker
  std::unique_ptr<TrackBuffer> fBuffer{nullptr};     ///< Buffers for transferring tracks between host and device
  std::unique_ptr<G4HepEmState> fg4hepem_state;      ///< The HepEm state singleton
  std::thread fGPUWorker;                            ///< Thread to manage GPU
  std::condition_variable fCV_G4Workers;             ///< Communicate with G4 workers
  std::mutex fMutex_G4Workers;                       ///< Mutex associated to the condition variable
  std::vector<std::atomic<EventState>> fEventStates; ///< State machine for each G4 worker
  std::vector<double> fGPUNetEnergy;
  bool fTrackInAllRegions = false;
  std::vector<std::string> const *fGPURegionNames;

  void FullInit();
  void InitBVH();
  bool InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world);
  bool InitializePhysics();
  void InitializeGPU();
  void FreeGPU();
  /// @brief Asynchronous loop for transporting particles on GPU.
  void TransportLoop();
  void HitProcessingLoop(HitProcessingContext *const);
  void ReturnTracksToG4();
  void AdvanceEventStates(EventState oldState, EventState newState);
  std::shared_ptr<const std::vector<GPUHit>> GetGPUHits(unsigned int threadId) const;

public:
  AdeptIntegration(unsigned short nThread, unsigned int trackCapacity, unsigned int hitBufferCapacity, int debugLevel,
                   std::vector<std::string> const *GPURegionNames, bool trackInAllRegions, uint64_t seed = 1234567);
  AdeptIntegration(const AdeptIntegration &other) = delete;
  ~AdeptIntegration();

  /// @brief Adds a track to the buffer
  void AddTrack(int pdg, double energy, double x, double y, double z, double dirx, double diry, double dirz,
                double globalTime, double localTime, double properTime, int threadId, unsigned int eventId,
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
  std::vector<std::string> const *GetGPURegionNames() override { return fGPURegionNames; }
  /// No effect
  void Initialize(bool) override {}
  /// @brief Finish GPU transport, bring hits and tracks to host
  void Shower(int event, int threadId) override { Flush(threadId, event); }
  /// Block until transport of the given event is done.
  void Flush(int threadId, int eventId);
  void Cleanup() override {}
};

} // namespace AsyncAdePT

#endif
