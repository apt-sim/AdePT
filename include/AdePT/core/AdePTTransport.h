// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

///   The AdePT core class, providing:
///   - filling the buffer with tracks to be transported on the GPU
///   - Calling the Shower method transporting a buffer on the GPU

#ifndef ADEPT_TRANSPORT_H
#define ADEPT_TRANSPORT_H

#include <AdePT/core/CommonStruct.h>
#include <AdePT/core/AdePTTransportInterface.hh>
#include <AdePT/core/AdePTScoringTemplate.cuh>
#include <AdePT/core/HostScoringStruct.cuh>
#include <AdePT/core/AdePTConfiguration.hh>
#include <AdePT/magneticfield/GeneralMagneticField.h>
#include <AdePT/magneticfield/UniformMagneticField.h>
#include <AdePT/integration/G4HepEmTrackingManagerSpecialized.hh>

#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif

class G4Region;
struct GPUstate;
class G4VPhysicalVolume;
struct G4HepEmState;

template <typename IntegrationLayer>
class AdePTTransport : public AdePTTransportInterface {
public:
  static constexpr int kMaxThreads = 256;
  using TrackBuffer                = adeptint::TrackBuffer;
  using VolAuxArray                = adeptint::VolAuxArray;

  AdePTTransport(AdePTConfiguration &configuration, G4HepEmTrackingManagerSpecialized *hepEmTM);

  ~AdePTTransport() { delete fScoring; }

  int GetNtoDevice() const { return fBuffer.toDevice.size(); }

  int GetNfromDevice() const { return fBuffer.fromDevice.size(); }

  /// @brief Adds a track to the buffer
  void AddTrack(int pdg, int parentId, double energy, double vertexEnergy, double x, double y, double z, double dirx,
                double diry, double dirz, double vertexX, double vertexY, double vertexZ, double vertexDirx,
                double vertexDiry, double vertexDirz, double globalTime, double localTime, double properTime,
                float weight, int threadId, unsigned int eventId, unsigned int trackIndex,
                vecgeom::NavigationState &&state, vecgeom::NavigationState &&originState);

  void SetTrackCapacity(size_t capacity) { fCapacity = capacity; }
  /// @brief Get the track capacity on GPU
  int GetTrackCapacity() const { return fCapacity; }
  /// @brief  Set capacity of the leak buffer on GPU
  void SetLeakCapacity(size_t capacity) { fLeakCapacity = capacity; }
  /// @brief Get the leak buffer size
  int GetLeakCapacity(size_t capacity) { return fLeakCapacity; }
  /// @brief Set Hit buffer capacity on GPU and Host
  void SetHitBufferCapacity(size_t capacity) { fHitBufferCapacity = capacity; }
  /// @brief Get the hit buffer size (host/device)
  int GetHitBufferCapacity() const { return fHitBufferCapacity; }
  /// @brief Set maximum batch size
  void SetMaxBatch(int npart) { fMaxBatch = npart; }
  /// @brief Set buffer threshold
  void SetBufferThreshold(int limit) { fBufferThreshold = limit; }
  /// @brief Set debug level for transport
  void SetDebugLevel(int level) { fDebugLevel = level; }
  /// @brief Set whether AdePT should transport particles across the whole geometry
  void SetTrackInAllRegions(bool trackInAllRegions) { fTrackInAllRegions = trackInAllRegions; }
  bool GetTrackInAllRegions() const { return fTrackInAllRegions; }
  /// @brief Set Geant4 region to which it applies
  void SetGPURegionNames(std::vector<std::string> const *regionNames) { fGPURegionNames = regionNames; }
  void SetCPURegionNames(std::vector<std::string> const *regionNames) { fCPURegionNames = regionNames; }
  /// @brief Set path to covfie Bfield file
  void SetBfieldFileName(const std::string &fileName) override { fBfieldFile = fileName; }
  /// @brief Set CUDA device stack limit
  void SetCUDAStackLimit(int limit) { fCUDAStackLimit = limit; }
  void SetCUDAHeapLimit(int limit) { fCUDAHeapLimit = limit; }
  std::vector<std::string> const *GetGPURegionNames() { return fGPURegionNames; }
  std::vector<std::string> const *GetCPURegionNames() { return fCPURegionNames; }
  /// @brief Create material-cut couple index array
  /// @brief Initialize service and copy geometry & physics data on device
  void Initialize(G4HepEmConfig *hepEmConfig, bool common_data = false);
  /// @brief Final cleanup
  void Cleanup();
  /// @brief Interface for transporting a buffer of tracks in AdePT.
  void Shower(int event, int threadId);
  void ProcessGPUSteps(int, int) {};
  /// @brief Setup function used only in async AdePT
  /// @param threadId thread Id
  /// @param hepEmTM specialized G4HepEmTrackingManager
  void SetIntegrationLayerForThread(int threadId, G4HepEmTrackingManagerSpecialized *hepEmTM) override {};

private:
  static inline G4HepEmState *fg4hepem_state{nullptr}; ///< The HepEm state singleton
  int fCapacity{1024 * 1024};                          ///< Track container capacity on GPU
  int fLeakCapacity{1024 * 1024};                      ///< Track container capacity on GPU
  int fHitBufferCapacity{1024 * 1024};                 ///< Capacity of hit buffers
  int fNthreads{0};                                    ///< Number of cpu threads
  int fMaxBatch{0};                                    ///< Max batch size for allocating GPU memory
  int fNumVolumes{0};                                  ///< Total number of active logical volumes
  int fNumSensitive{0};                                ///< Total number of sensitive volumes
  size_t fBufferThreshold{20};                         ///< Buffer threshold for flushing AdePT transport buffer
  int fDebugLevel{1};                                  ///< Debug level
  int fCUDAStackLimit{0};                              ///< CUDA device stack limit
  int fCUDAHeapLimit{0};                               ///< CUDA device heap limit
  GPUstate *fGPUstate{nullptr};                        ///< CUDA state placeholder
  AdeptScoring *fScoring{nullptr};                     ///< User scoring object
  AdeptScoring *fScoring_dev{nullptr};                 ///< Device ptr for scoring data
  TrackBuffer fBuffer;                                 ///< Vector of buffers of tracks to/from device (per thread)
  std::vector<std::string> const *fGPURegionNames{};   ///< Regions that are run on GPU
  std::vector<std::string> const *fCPURegionNames{};   ///< Regions that are run on CPU
  IntegrationLayer fIntegrationLayer;  ///< Provides functionality needed for integration with the simulation toolkit
  bool fInit{false};                   ///< Service initialized flag
  bool fTrackInAllRegions;             ///< Whether the whole geometry is a GPU region
  std::string fBfieldFile{""};         ///< Path to magnetic field file (in the covfie format)
  GeneralMagneticField fMagneticField; ///< arbitrary magnetic field

  /// @brief Used to map VecGeom to Geant4 volumes for scoring
  void InitializeSensitiveVolumeMapping(const G4VPhysicalVolume *g4world, const vecgeom::VPlacedVolume *world);
  void InitBVH();
  bool InitializeBField();
  bool InitializeBField(UniformMagneticField &Bfield);
  bool InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world);
  bool InitializePhysics(G4HepEmConfig *hepEmConfig);
};

#include "AdePTTransport.icc"

#endif
