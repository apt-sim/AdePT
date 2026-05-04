// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

///   The AdePT-Geant4 integration layer
///   - Initialization and checking of VecGeom geometry
///   - G4Step reconstruction from GPU steps
///   - Processing of reconstructed steps using the Geant4 sensitive detector

#ifndef ADEPTGEANT4_INTEGRATION_H
#define ADEPTGEANT4_INTEGRATION_H

#include <AdePT/transport/steps/GPUStep.hh>
#include <AdePT/g4integration/tracking_managers/G4HepEmTrackingManagerSpecialized.hh>
#include <AdePT/g4integration/returned_steps/HostTrackDataMapper.hh>

#include <G4EventManager.hh>
#include <G4Event.hh>
#include <G4Track.hh>

#include <span>
#include <vector>

namespace AdePTGeant4Integration_detail {
struct StepReconstructionObjects;
struct Deleter {
  void operator()(StepReconstructionObjects *ptr);
};
} // namespace AdePTGeant4Integration_detail

class AdePTGeant4Integration {
public:
  enum class DeferredStepType : unsigned char { ReplayStep, ReturnTrack };

  /// @brief Stored work for a returned step that is replayed later on the host.
  /// @details
  /// The host collects returned GPU-step blocks during transport and replays
  /// them later in one fixed order, so the Geant4-side work stays
  /// reproducible from run to run.
  struct DeferredStep {
    std::size_t firstGPUStep{0};
    std::size_t numGPUSteps{0};
    DeferredStepType type{DeferredStepType::ReplayStep};
  };

  /// @brief Owns the deferred returned-step data drained from the integration.
  struct DeferredStepStore {
    std::vector<GPUStep> gpuSteps{};
    std::vector<DeferredStep> steps{};
  };

  explicit AdePTGeant4Integration() : fHostTrackDataMapper(std::make_unique<HostTrackDataMapper>()) {}
  ~AdePTGeant4Integration();

  AdePTGeant4Integration(const AdePTGeant4Integration &)            = delete;
  AdePTGeant4Integration &operator=(const AdePTGeant4Integration &) = delete;

  AdePTGeant4Integration(AdePTGeant4Integration &&)            = default;
  AdePTGeant4Integration &operator=(AdePTGeant4Integration &&) = default;

  /// @brief Reconstructs GPU steps on host and calls the user-defined sensitive detector code
  void ProcessGPUStep(std::span<const GPUStep> gpuSteps, bool const callUserSteppingAction = false,
                      bool const callUserTrackingaction = false);

  /// @brief Return a deferred parent track to Geant4 without rebuilding the visible G4 step.
  /// @details
  /// This is only used for returned gamma handoff steps with one parent step,
  /// no secondaries, and zero deposited energy. In that case there is no GPU
  /// step to process on the host, so only the parent G4Track is rebuilt from the
  /// post-step state and pushed back to the Geant4 stack.
  void ReturnDeferredTrack(std::span<const GPUStep> gpuSteps, bool const callUserActions = false);

  /// @brief Returns the Z value of the user-defined uniform magnetic field
  /// @details This function can only be called when the user-defined field is a G4UniformMagField
  std::vector<float> GetUniformField() const;

  int GetEventID() const { return G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID(); }

  int GetThreadID() const { return G4Threading::G4GetThreadId(); }

  HostTrackDataMapper &GetHostTrackDataMapper() { return *fHostTrackDataMapper; }

  /// @brief Defer a returned step for later sorted replay on the host.
  void QueueDeferredStep(std::span<const GPUStep> gpuSteps, DeferredStepType type = DeferredStepType::ReplayStep);

  /// @brief Transfer ownership of the currently queued deferred steps.
  /// @details
  /// This drains the integration-local deferred-step storage without copying it.
  DeferredStepStore TakeDeferredSteps();

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
  G4Track *ConstructSecondaryTrackInPlace(GPUStep const *secStep) const;

  void InitSecondaryHostTrackDataFromParent(GPUStep const *secStep, HostTrackData &secTData, int g4ParentID,
                                            G4TouchableHandle &preTouchable) const;

  void FillG4Track(GPUStep const *aGPUStep, G4Track *aG4Track, const HostTrackData &hostTData,
                   G4TouchableHandle &aPreG4TouchableHandle, G4TouchableHandle &aPostG4TouchableHandle) const;

  void FillG4Step(GPUStep const *aGPUStep, G4Step *aG4Step, const HostTrackData &hostTData,
                  G4TouchableHandle &aPreG4TouchableHandle, G4TouchableHandle &aPostG4TouchableHandle,
                  G4StepStatus aPreStepStatus, G4StepStatus aPostStepStatus, bool callUserTrackingAction,
                  bool callUserSteppingAction) const;

  /// @brief Create a heap-owned track that can be pushed onto the Geant4 stack.
  /// @details
  /// This is only used as a fallback for gamma/lepton nuclear when no Geant4
  /// nuclear process is attached. In that case there is no temporary nuclear
  /// replay track to continue on the CPU, and the visible reconstructed track
  /// cannot be handed to the stack manager because it is reused integration
  /// storage.
  G4Track *MakeTrackForCPUStacking(const G4Track &track) const;

  /// @brief Recreate a track to from a returned parent step to be continued on CPU.
  /// @details
  /// Out-of-GPU-region and finish-on-CPU steps used to hand Geant4 a returned
  /// track built from the post-step state. The visible reconstructed step keeps
  /// the transported GPU-step data, but the continued CPU track must still
  /// match that old current-state handoff.
  G4Track *MakeReturnedTrackFromStep(GPUStep const &parentStep, const HostTrackData &hostTData,
                                     bool setStopButAlive) const;

  // pointer to specialized G4HepEmTrackingManager. Owned by AdePTTrackingManager,
  // this is just a reference to handle gamma-/lepton-nuclear reactions
  G4HepEmTrackingManagerSpecialized *fHepEmTrackingManager{nullptr};

  // helper class to provide the CPU-only data for the returning GPU tracks
  std::unique_ptr<HostTrackDataMapper> fHostTrackDataMapper;

  std::unique_ptr<AdePTGeant4Integration_detail::StepReconstructionObjects, AdePTGeant4Integration_detail::Deleter>
      fStepReconstructionObjects{nullptr};

  std::vector<GPUStep> fDeferredGPUSteps;
  std::vector<DeferredStep> fDeferredSteps;
};

#endif
