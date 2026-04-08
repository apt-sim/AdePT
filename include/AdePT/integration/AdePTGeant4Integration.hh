// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

///   The AdePT-Geant4 integration layer
///   - Initialization and checking of VecGeom geometry
///   - G4Hit reconstruction from GPU hits
///   - Processing of reconstructed hits using the G4 Sensitive Detector

#ifndef ADEPTGEANT4_INTEGRATION_H
#define ADEPTGEANT4_INTEGRATION_H

#include <AdePT/core/ScoringCommons.hh>
#include <AdePT/core/TrackData.h>
#include <AdePT/integration/G4HepEmTrackingManagerSpecialized.hh>
#include <AdePT/core/ReturnedTrackData.hh>
#include <AdePT/integration/HostTrackDataMapper.hh>

#include <G4EventManager.hh>
#include <G4Event.hh>
#include <G4Track.hh>

#include <span>
#include <vector>

namespace AdePTGeant4Integration_detail {
struct ScoringObjects;
struct Deleter {
  void operator()(ScoringObjects *ptr);
};
} // namespace AdePTGeant4Integration_detail

class AdePTGeant4Integration {
public:
  /// @brief Stored work for a returned gamma/lepton nuclear step.
  /// @details
  /// These steps are handled later in the same sorted return order as ordinary
  /// leaked tracks, so the Geant4 nuclear process always runs in a fixed order.
  struct DeferredNuclearStep {
    adeptint::TrackData returnedTrack{};
    std::vector<GPUHit> hits{};
  };

  explicit AdePTGeant4Integration() : fHostTrackDataMapper(std::make_unique<HostTrackDataMapper>()) {}
  ~AdePTGeant4Integration();

  AdePTGeant4Integration(const AdePTGeant4Integration &)            = delete;
  AdePTGeant4Integration &operator=(const AdePTGeant4Integration &) = delete;

  AdePTGeant4Integration(AdePTGeant4Integration &&)            = default;
  AdePTGeant4Integration &operator=(AdePTGeant4Integration &&) = default;

  /// @brief Reconstructs GPU hits on host and calls the user-defined sensitive detector code
  void ProcessGPUStep(std::span<const GPUHit> gpuSteps, bool const callUserSteppingAction = false,
                      bool const callUserTrackingaction = false);

  /// @brief Takes a range of tracks coming from the device and gives them back to Geant4
  template <typename Iterator>
  void ReturnTracks(Iterator begin, Iterator end, int debugLevel, bool callUserActions = false) const
  {
    if (debugLevel > 1) {
      G4cout << "Returning " << end - begin << " tracks from device" << G4endl;
    }
    for (Iterator it = begin; it != end; ++it) {
      ReturnTrack(*it, it - begin, debugLevel, callUserActions);
    }
  }

  /// @brief Returns the Z value of the user-defined uniform magnetic field
  /// @details This function can only be called when the user-defined field is a G4UniformMagField
  std::vector<float> GetUniformField() const;

  int GetEventID() const { return G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID(); }

  int GetThreadID() const { return G4Threading::G4GetThreadId(); }

  HostTrackDataMapper &GetHostTrackDataMapper() { return *fHostTrackDataMapper; }

  /// @brief Defer a returned nuclear step for later sorted replay on the host.
  void QueueDeferredNuclearStep(std::span<const GPUHit> gpuSteps);

  /// @brief Transfer ownership of the currently queued deferred nuclear steps.
  /// @details
  /// This drains the integration-local queue into a temporary vector without
  /// copying the stored GPU-hit blocks.
  std::vector<DeferredNuclearStep> TakeDeferredNuclearSteps();

  void ReturnTrack(adeptint::TrackData const &track, unsigned int trackIndex, int debugLevel,
                   bool callUserActions = false) const;

  void SetHepEmTrackingManager(G4HepEmTrackingManagerSpecialized *hepEmTrackingManager)
  {
    fHepEmTrackingManager = hepEmTrackingManager;
  }

private:
  /// @brief Reconstruct G4TouchableHistory from a VecGeom Navigation index
  void FillG4NavigationHistory(const vecgeom::NavigationState &aNavState,
                               G4NavigationHistory &aG4NavigationHistory) const;

  G4TouchableHandle MakeTouchableFromNavState(vecgeom::NavigationState const &navState) const;

  /// @brief Construct the temporary secondary track that is attached to the secondary vector of the parent step
  G4Track *ConstructSecondaryTrackInPlace(GPUHit const *secHit) const;

  void InitSecondaryHostTrackDataFromParent(GPUHit const *secHit, HostTrackData &secTData, int g4ParentID,
                                            G4TouchableHandle &preTouchable) const;

  void FillG4Track(GPUHit const *aGPUHit, G4Track *aG4Track, const HostTrackData &hostTData,
                   G4TouchableHandle &aPreG4TouchableHandle, G4TouchableHandle &aPostG4TouchableHandle) const;

  void FillG4Step(GPUHit const *aGPUHit, G4Step *aG4Step, const HostTrackData &hostTData,
                  G4TouchableHandle &aPreG4TouchableHandle, G4TouchableHandle &aPostG4TouchableHandle,
                  G4StepStatus aPreStepStatus, G4StepStatus aPostStepStatus, bool callUserTrackingAction,
                  bool callUserSteppingAction) const;

  /// @brief Build the ordering key for a deferred nuclear step.
  adeptint::TrackData MakeReturnedTrackFromGPUHit(GPUHit const &gpuHit) const;

  /// @brief Create a heap-owned track that can be pushed onto the Geant4 stack.
  /// @details
  /// This is only used as a fallback for gamma/lepton nuclear when no Geant4
  /// nuclear process is attached. In that case there is no temporary nuclear
  /// replay track to continue on the CPU, and the visible reconstructed track
  /// cannot be handed to the stack manager because it is reused integration
  /// storage.
  G4Track *MakeTrackForCPUStacking(const G4Track &track) const;

  // pointer to specialized G4HepEmTrackingManager. Owned by AdePTTrackingManager,
  // this is just a reference to handle gamma-/lepton-nuclear reactions
  G4HepEmTrackingManagerSpecialized *fHepEmTrackingManager{nullptr};

  // helper class to provide the CPU-only data for the returning GPU tracks
  std::unique_ptr<HostTrackDataMapper> fHostTrackDataMapper;

  std::unique_ptr<AdePTGeant4Integration_detail::ScoringObjects, AdePTGeant4Integration_detail::Deleter>
      fScoringObjects{nullptr};

  std::vector<DeferredNuclearStep> fDeferredNuclearSteps;
};

#endif
