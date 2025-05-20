// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

/// The The Geant4 AdePT integration service. This provides the interfaces for:
/// - initializing the geometry and physics on the AdePT size
/// - filling the buffer with tracks to be transported on the GPU
/// - Calling the Shower method transporting a buffer on the GPU

#ifndef ASYNC_ADEPT_TRANSPORT_HH
#define ASYNC_ADEPT_TRANSPORT_HH

#define ADEPT_SAVE_IDs

#include <AdePT/core/AdePTTransportInterface.hh>
#include <AdePT/core/CommonStruct.h>
#include <AdePT/core/AdePTConfiguration.hh>
#include <AdePT/core/AsyncAdePTTransportStruct.hh>
#include <AdePT/core/PerEventScoringStruct.cuh>
#include <AdePT/magneticfield/GeneralMagneticField.h>
#include <AdePT/magneticfield/UniformMagneticField.h>
#include <AdePT/integration/G4HepEmTrackingManagerSpecialized.hh>

#include <VecGeom/base/Config.h>
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume

#include <condition_variable>
#include <mutex>
#include <memory>
#include <thread>
#include <unordered_map>

class G4Region;
class G4VPhysicalVolume;
struct G4HepEmState;
namespace AsyncAdePT {
struct TrackBuffer;
struct GPUstate;

void InitVolAuxArray(adeptint::VolAuxArray &array);

template <typename IntegrationLayer>
class AsyncAdePTTransport : public AdePTTransportInterface {
public:
  static inline uint64_t fAdePTSeed = 1234567;

private:
  unsigned short fNThread{0};             ///< Number of G4 workers
  unsigned int fTrackCapacity{0};         ///< Number of track slots to allocate on device
  unsigned int fLeakCapacity{0};          ///< Number of leak slots to allocate on device
  unsigned int fScoringCapacity{0};       ///< Number of hit slots to allocate on device
  int fDebugLevel{0};                     ///< Debug level
  int fCUDAStackLimit{0};                 ///< CUDA device stack limit
  int fCUDAHeapLimit{0};                  ///< CUDA device heap limit
  unsigned short fLastNParticlesOnCPU{0}; ///< Number N of last N particles that are finished on CPU
  // note: std::optional is used here as the AdePTGeant4Integration has no default constructor and we need to
  // resize the vector to the number of threads, and then each worker has to construct its entry at its given slot
  std::vector<std::optional<IntegrationLayer>> fIntegrationLayerObjects; //< vector of integration layers per thread
  std::unique_ptr<GPUstate, GPUstateDeleter> fGPUstate{nullptr};         ///< CUDA state placeholder
  std::vector<AdePTScoring> fScoring;                                    ///< User scoring objects per G4 worker
  std::unique_ptr<TrackBuffer> fBuffer{nullptr};     ///< Buffers for transferring tracks between host and device
  std::unique_ptr<G4HepEmState> fg4hepem_state;      ///< The HepEm state singleton
  std::thread fGPUWorker;                            ///< Thread to manage GPU
  std::condition_variable fCV_G4Workers;             ///< Communicate with G4 workers
  std::mutex fMutex_G4Workers;                       ///< Mutex associated to the condition variable
  std::vector<std::atomic<EventState>> fEventStates; ///< State machine for each G4 worker
  std::vector<double> fGPUNetEnergy;
  bool fTrackInAllRegions = false;
  std::vector<std::string> const *fGPURegionNames;
  std::vector<std::string> const *fCPURegionNames;
  // Flags for the kernels to return the last or all steps, needed for PostUserTrackingAction or UserSteppingAction
  bool fReturnAllSteps         = false;
  bool fReturnFirstAndLastStep = false;
  std::string fBfieldFile{""};         ///< Path to magnetic field file (in the covfie format)
  GeneralMagneticField fMagneticField; ///< arbitrary magnetic field
  double fCPUCapacityFactor{
      2.5}; ///< Factor by which the ScoringCapacity on Host is larger than on Device. Must be at least 2
  ///< Filling fraction of the ScoringCapacity on host when the hits are copied out and not taken directly by the
  ///< G4workers
  double fCPUCopyFraction{0.5};

  void InitBVH();
  bool InitializeBField();
  bool InitializeBField(UniformMagneticField &Bfield);
  bool InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world);
  bool InitializePhysics(G4HepEmConfig *hepEmConfig);

public:
  AsyncAdePTTransport(AdePTConfiguration &configuration, G4HepEmTrackingManagerSpecialized *hepEmTM);
  AsyncAdePTTransport(const AsyncAdePTTransport &other) = delete;
  ~AsyncAdePTTransport();

  /// @brief Adds a track to the buffer
  void AddTrack(int pdg, int parentId, double energy, double vertexEnergy, double x, double y, double z, double dirx,
                double diry, double dirz, double vertexX, double vertexY, double vertexZ, double vertexDirx,
                double vertexDiry, double vertexDirz, double globalTime, double localTime, double properTime,
                float weight, int threadId, unsigned int eventId, unsigned int trackIndex,
                vecgeom::NavigationState &&state, vecgeom::NavigationState &&originState) override;
  /// @brief Set track capacity on GPU
  void SetTrackCapacity(size_t capacity) override { fTrackCapacity = capacity; }
  /// @brief Set leak capacity on GPU
  void SetLeakCapacity(size_t capacity) override { fLeakCapacity = capacity; }
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
  void SetCPURegionNames(std::vector<std::string> const *regionNames) override { fCPURegionNames = regionNames; }
  /// @brief Set path to covfie Bfield file
  void SetBfieldFileName(const std::string &fileName) override { fBfieldFile = fileName; }
  void SetCUDAStackLimit(int limit) override {};
  void SetCUDAHeapLimit(int limit) override {};
  std::vector<std::string> const *GetGPURegionNames() override { return fGPURegionNames; }
  std::vector<std::string> const *GetCPURegionNames() override { return fCPURegionNames; }
  /// No effect
  void Initialize(G4HepEmConfig *hepEmConfig, bool) override {}
  void Initialize(G4HepEmTrackingManagerSpecialized *hepEmTM, int threadId);
  /// @brief Finish GPU transport, bring hits and tracks to host
  /// @details The shower call exists to maintain the same interface as the
  /// synchronous AdePT mode, since in this case the transport loop is always
  /// running. The only call to Shower() from G4 is done when the tracking
  /// manager needs to flush an event.
  void Shower(int event, int threadId) override { Flush(threadId, event); }
  /// Block until transport of the given event is done.
  void Flush(int threadId, int eventId);
  void ProcessGPUSteps(int threadId, int eventId) override;
  void Cleanup() override {}
  /// @brief Setup function used only in async AdePT
  /// @param threadId thread Id
  /// @param hepEmTM specialized G4HepEmTrackingManager
  void SetIntegrationLayerForThread(int threadId, G4HepEmTrackingManagerSpecialized *hepEmTM) override;
};

} // namespace AsyncAdePT

#include "AsyncAdePTTransport.icc"

#endif
