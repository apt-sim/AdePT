// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

///   The AdePT core class, providing:
///   - filling the buffer with tracks to be transported on the GPU
///   - Calling the Shower method transporting a buffer on the GPU

#ifndef ADEPT_INTEGRATION_H
#define ADEPT_INTEGRATION_H

#include <unordered_map>
#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif

#include <G4HepEmState.hh>

#include "CommonStruct.h"
#include <AdePT/core/AdePTScoringTemplate.cuh>
#include <AdePT/core/HostScoringStruct.cuh>

class G4Region;
struct GPUstate;
class G4VPhysicalVolume;

template <class TTag>
class TestManager;

template <typename IntegrationLayer>
class AdePTTransport {
public:
  static constexpr int kMaxThreads = 256;
  using TrackBuffer                = adeptint::TrackBuffer;
  using VolAuxArray                = adeptint::VolAuxArray;

  AdePTTransport() = default;

  ~AdePTTransport() { delete fScoring; }

  int GetNtoDevice() const { return fBuffer.toDevice.size(); }

  int GetNfromDevice() const { return fBuffer.fromDevice.size(); }

  /// @brief Adds a track to the buffer
  void AddTrack(int pdg, int parentID, double energy, double x, double y, double z, double dirx, double diry, double dirz,
                double globalTime, double localTime, double properTime);

  void SetTrackCapacity(size_t capacity) { fCapacity = capacity; }
  /// @brief Get the track capacity on GPU
  int GetTrackCapacity() const { return fCapacity; }
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
  bool GetTrackInAllRegions() { return fTrackInAllRegions; }
  /// @brief Set Geant4 region to which it applies
  void SetGPURegionNames(std::vector<std::string> *regionNames) { fGPURegionNames = regionNames; }
  std::vector<std::string> *GetGPURegionNames() { return fGPURegionNames; }
  /// @brief Create material-cut couple index array
  /// @brief Initialize service and copy geometry & physics data on device
  void Initialize(bool common_data = false);
  /// @brief Final cleanup
  void Cleanup();
  /// @brief Interface for transporting a buffer of tracks in AdePT.
  void Shower(int event);

private:
  static inline G4HepEmState *fg4hepem_state{nullptr}; ///< The HepEm state singleton
  static inline int fCapacity{1024 * 1024};            ///< Track container capacity on GPU
  static inline int fHitBufferCapacity{1024 * 1024};   ///< Capacity of hit buffers
  int fNthreads{0};                                    ///< Number of cpu threads
  int fMaxBatch{0};                                    ///< Max batch size for allocating GPU memory
  int fNumVolumes{0};                                  ///< Total number of active logical volumes
  int fNumSensitive{0};                                ///< Total number of sensitive volumes
  int fBufferThreshold{20};                            ///< Buffer threshold for flushing AdePT transport buffer
  int fDebugLevel{1};                                  ///< Debug level
  GPUstate *fGPUstate{nullptr};                        ///< CUDA state placeholder
  AdeptScoring *fScoring{nullptr};                     ///< User scoring object
  AdeptScoring *fScoring_dev{nullptr};                 ///< Device ptr for scoring data
  TrackBuffer fBuffer;                                 ///< Vector of buffers of tracks to/from device (per thread)
  std::vector<std::string> *fGPURegionNames{};         ///< Region to which applies
  IntegrationLayer fIntegrationLayer; ///< Provides functionality needed for integration with the simulation toolkit
  bool fInit{false};                  ///< Service initialized flag
  bool fTrackInAllRegions;            ///< Whether the whole geometry is a GPU region

  /// @brief Used to map VecGeom to Geant4 volumes for scoring
  void InitializeSensitiveVolumeMapping(const G4VPhysicalVolume *g4world, const vecgeom::VPlacedVolume *world);
  void InitBVH();
  bool InitializeField(double bz);
  bool InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world);
  bool InitializePhysics();
  void ProcessGPUHits();
};

#include "AdePTTransport.icc"

#endif
