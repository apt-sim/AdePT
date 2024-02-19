// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

/// The The Geant4 AdePT integration service. This provides the interfaces for:
/// - initializing the geometry and physics on the AdePT size
/// - filling the buffer with tracks to be transported on the GPU
/// - Calling the Shower method transporting a buffer on the GPU

#ifndef ADEPT_INTEGRATION_H
#define ADEPT_INTEGRATION_H

#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif

#include <G4HepEmState.hh>

// For the moment the scoring type will be determined by what we include here
#include "CommonStruct.h"
#include "BasicScoring.h"
#include "G4FastSimHitMaker.hh"

#include <memory>
#include <unordered_map>

class G4Region;
struct GPUstate;
class G4VPhysicalVolume;

class AdeptIntegration {
public:
  static constexpr int kMaxThreads = 256;
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
  unsigned short fNThread{0};          ///< Number of G4 workers
  unsigned int fTrackCapacity;         ///< Number of track slots to allocate on device
  int fNumVolumes{0};                  ///< Total number of active logical volumes
  int fNumSensitive{0};                ///< Total number of sensitive volumes
  unsigned int fBufferThreshold{20};   ///< Buffer threshold for starting a copy to the GPU
  int fDebugLevel{1};                  ///< Debug level
  std::unique_ptr<GPUstate> fGPUstate; ///< CUDA state placeholder
  std::vector<AdeptScoring> fScoring;  ///< User scoring objects per G4 worker
  std::unique_ptr<adeptint::TrackBuffer> fBuffer{nullptr}; ///< Buffers for transferring tracks between host and device
  AdeptScoring *fScoring_dev{nullptr};                     ///< Device array for per-worker scoring data
  static G4HepEmState *fg4hepem_state; ///< The HepEm state singleton
  G4Region const *fRegion{nullptr};    ///< Region to which applies
  std::unordered_map<std::string, int> const &sensitive_volume_index; ///< Map of sensitive volumes
  std::unordered_map<const G4VPhysicalVolume *, int> &fScoringMap;    ///< Map used by G4 for scoring
  std::thread fGPUWorker;                                             ///< Thread to manage GPU
  enum class EventState : unsigned char {
    NewTracksFromG4,
    InjectionRunning,
    TracksInjected,
    DeviceFlushed,
    LeakedTracksRetrieved,
    ScoringRetrieved
  };
  std::vector<std::atomic<EventState>> fEventStates; ///< State machine for each G4 worker
  std::vector<double> fGPUNetEnergy;

  VolAuxData *CreateVolAuxData(const G4VPhysicalVolume *g4world, const vecgeom::VPlacedVolume *world,
                               const G4HepEmState &hepEmState);
  void InitBVH();
  bool InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world);
  bool InitializePhysics();
  void InitializeGPU();
  void FreeGPU();
  /// @brief Asynchronous loop for transporting particles on GPU.
  void TransportLoop();

public:
  AdeptIntegration(unsigned short nThread, unsigned int trackCapacity, unsigned int bufferThreshold, int debugLevel,
                   G4Region *region, std::unordered_map<std::string, int> &sensVolIndex,
                   std::unordered_map<const G4VPhysicalVolume *, int> &scoringMap);
  AdeptIntegration(const AdeptIntegration &other) = delete;
  ~AdeptIntegration();

  /// @brief Adds a track to the buffer
  void AddTrack(G4int threadId, G4int eventId, unsigned short cycleNumber, unsigned int trackIndex, int pdg,
                double energy, double x, double y, double z, double dirx, double diry, double dirz);
  /// @brief Set track capacity on GPU
  void SetTrackCapacity(size_t capacity) { fTrackCapacity = capacity; }
  /// @brief Set buffer threshold
  void SetBufferThreshold(int limit) { fBufferThreshold = limit; }
  /// @brief Set debug level for transport
  void SetDebugLevel(int level) { fDebugLevel = level; }
  /// @brief Set Geant4 region to which it applies
  void SetRegion(G4Region *region) { fRegion = region; }
  /// @brief Initialize service and copy geometry & physics data on device
  void Initialize();
  /// @brief Finish GPU transport, bring hits and tracks to host
  void Flush(G4int threadId, G4int eventId, unsigned short cycleNumber);
};

#endif
