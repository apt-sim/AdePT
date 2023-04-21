// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

/// The The Geant4 AdePT integration service. This provides the interfaces for:
/// - initializing the geometry and physics on the AdePT size
/// - filling the buffer with tracks to be transported on the GPU
/// - Calling the Shower method transporting a buffer on the GPU

#ifndef ADEPT_INTEGRATION_H
#define ADEPT_INTEGRATION_H

#include <unordered_map>
#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif

#include <G4HepEmState.hh>
// #include <G4HepEmData.hh>
// #include <G4HepEmParameters.hh>

// For the moment the scoring type will be determined by what we include here
#include "CommonStruct.h"
#include "BasicScoring.h"
#include "G4FastSimHitMaker.hh"

class G4Region;
struct GPUstate;
class G4VPhysicalVolume;

class AdeptIntegration {
public:
  static constexpr int kMaxThreads = 256;
  // Track capacity
  static int kCapacity;

  using TrackBuffer = adeptint::TrackBuffer;
  using VolAuxData  = adeptint::VolAuxData;

  /// @brief Structure holding the arrays of auxiliary volume data on host and device
  struct VolAuxArray {
    int fNumVolumes{0};
    VolAuxData *fAuxData{nullptr};     ///< array of auxiliary volume data on host
    VolAuxData *fAuxData_dev{nullptr}; ///< array of auxiliary volume data on device

    static VolAuxArray &GetInstance()
    {
      static VolAuxArray theAuxArray;
      return theAuxArray;
    }

    ~VolAuxArray() { FreeGPU(); }

    void InitializeOnGPU();
    void FreeGPU();
  };

private:
  bool fInit{false};                   ///< Service initialized flag
  int fNthreads{0};                    ///< Number of cpu threads
  int fMaxBatch{0};                    ///< Max batch size for allocating GPU memory
  int fNumVolumes{0};                  ///< Total number of active logical volumes
  int fNumSensitive{0};                ///< Total number of sensitive volumes
  int fBufferThreshold{20};            ///< Buffer threshold for flushing AdePT transport buffer
  int fDebugLevel{1};                  ///< Debug level
  GPUstate *fGPUstate{nullptr};        ///< CUDA state placeholder
  AdeptScoring *fScoring{nullptr};     ///< User scoring object
  AdeptScoring *fScoring_dev{nullptr}; ///< Device ptr for scoring data
  static G4HepEmState *fg4hepem_state; ///< The HepEm state singleton
  TrackBuffer fBuffer;                 ///< Vector of buffers of tracks to/from device (per thread)
  G4Region *fRegion{nullptr};          ///< Region to which applies
  std::unordered_map<std::string, int> *sensitive_volume_index;    ///< Map of sensitive volumes
  std::unordered_map<const G4VPhysicalVolume *, int> *fScoringMap; ///< Map used by G4 for scoring

  VolAuxData *CreateVolAuxData(const G4VPhysicalVolume *g4world, const vecgeom::VPlacedVolume *world,
                               const G4HepEmState &hepEmState);
  void InitBVH();
  void InitializeUserData() { fScoring->InitializeOnGPU(); }
  bool InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world);
  bool InitializePhysics();
  void InitializeGPU();
  void ShowerGPU(int event, TrackBuffer &buffer); // const &buffer);
  void FreeGPU();

public:
  int GetNtoDevice() const { return fBuffer.toDevice.size(); }
  int GetNfromDevice() const { return fBuffer.fromDevice.size(); }

public:
  AdeptIntegration() = default;
  ~AdeptIntegration();

  /// @brief Adds a track to the buffer
  void AddTrack(int pdg, double energy, double x, double y, double z, double dirx, double diry, double dirz);
  /// @brief Prepare the buffers for copying leaked tracks
  /// @param numLeaked Number of tracks to be copied
  void PrepareLeakedBuffers(int numLeaked);
  /// @brief Set track capacity on GPU
  static void SetTrackCapacity(size_t capacity) { kCapacity = capacity; }
  /// @brief Set maximum batch size
  void SetMaxBatch(int npart) { fMaxBatch = npart; }
  /// @brief Set buffer threshold
  void SetBufferThreshold(int limit) { fBufferThreshold = limit; }
  /// @brief Set debug level for transport
  void SetDebugLevel(int level) { fDebugLevel = level; }
  /// @brief Set Geant4 region to which it applies
  void SetRegion(G4Region *region) { fRegion = region; }
  /// @brief Create material-cut couple index array
  /// @brief Initialize service and copy geometry & physics data on device
  void Initialize(bool common_data = false);
  /// @brief Final cleanup
  void Cleanup();
  /// @brief Interface for transporting a buffer of tracks in AdePT.
  void Shower(int event);

  /// Helper class for creation of hits within the sensitive detector
  // std::unique_ptr<G4FastSimHitMaker> fHitMaker;

  void SetSensitiveVolumes(std::unordered_map<std::string, int> *sv) { sensitive_volume_index = sv; }
  void SetScoringMap(std::unordered_map<const G4VPhysicalVolume *, int> *sm) { fScoringMap = sm; }
};

#endif
