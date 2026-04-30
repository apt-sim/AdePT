// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef AdePTTrackingManager_h
#define AdePTTrackingManager_h 1

#include "G4VTrackingManager.hh"
#include "G4RegionStore.hh"

#include "globals.hh"
#include <AdePT/core/AsyncAdePTTransport.hh>
#include "AdePT/copcore/SystemOfUnits.h"
#include <AdePT/integration/AdePTGeant4Integration.hh>
#include <AdePT/core/AdePTConfiguration.hh>
#include <AdePT/integration/G4HepEmTrackingManagerSpecialized.hh>

#include <memory>
#include <vector>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class AdePTTrackingManager : public G4VTrackingManager {
public:
  using AdePTTransport = AsyncAdePT::AsyncAdePTTransport;

  explicit AdePTTrackingManager(AdePTConfiguration *config, int verbosity = 0);
  ~AdePTTrackingManager();

  void BuildPhysicsTable(const G4ParticleDefinition &) override;

  void PreparePhysicsTable(const G4ParticleDefinition &) override;

  void HandOverOneTrack(G4Track *aTrack) override;

  void FlushEvent() override;

  void InitializeAdePT();

  G4HepEmConfig *GetG4HepEmConfig() { return fHepEmTrackingManager->GetConfig(); }

private:
  /// @brief Steps a particle using the generic G4 tracking, until it dies or enters a user-defined
  /// GPU region, in which case tracking is delegated to AdePT
  /// @details In order to be able to send tracks back and forth between the custom tracking and the generic
  /// G4 tracking, this function has to provide the functionality of G4TrackingManager::ProcessOneTrack, in
  /// addition to checking whether the track is in a GPU Region
  void ProcessTrack(G4Track *aTrack);

  /// @brief Get the corresponding VecGeom NavigationState from the G4NavigationHistory. The boundary status will be set
  /// to false by default
  /// @param aG4NavigationHistory the given G4NavigationHistory
  /// @return the corresponding vecgeom::NavigationState
  const vecgeom::NavigationState GetVecGeomFromG4State(const G4NavigationHistory &aG4NavigationHistory);

  /// @brief Get the corresponding VecGeom NavigationState from the track's G4NavigationHistory, and set the boundary
  /// status based on the track's state
  /// @param aG4Track The G4Track from which to extract the NavigationState
  /// @param aG4NavigationHistory Navigation history that is used to define the navState. If not provided, it will
  /// default to aG4Track.GetNextTouchableHandle()->GetHistory()
  /// @return The corresponding vecgeom::NavigationState
  const vecgeom::NavigationState GetVecGeomFromG4State(const G4Track &aG4Track,
                                                       const G4NavigationHistory *aG4NavigationHistory = nullptr);

  /// @brief Perform the one-time shared AdePT transport initialization on the first Geant4 worker.
  /// @details
  /// The first worker prepares all host-side inputs needed by transport:
  /// - the uniform magnetic-field values
  /// - the AdePT-owned `AdePTG4HepEmState`
  /// - geometry consistency checks
  /// - `VolAuxData`
  /// - packed WDT metadata
  ///
  /// Once that host-side preparation is complete, the worker creates the
  /// shared `AsyncAdePTTransport`. The transport constructor then performs the
  /// corresponding one-time device initialization and upload.
  void InitializeSharedAdePTTransport();

  /// @brief Drain returned GPU-step batches from transport and reconstruct the
  /// corresponding Geant4 steps on the CPU.
  /// @details
  /// Transport still owns the batch lifetime and iterates over the available
  /// batches. This helper provides the Geant4-side reconstruction logic for
  /// each batch.
  void ProcessReturnedGPUSteps(int threadId, int eventId);

  std::unique_ptr<G4HepEmTrackingManagerSpecialized> fHepEmTrackingManager;
  AdePTGeant4Integration fGeant4Integration;
  static inline int fNumThreads{0};
  std::set<G4Region const *> fGPURegions{};
  std::shared_ptr<AdePTTransport> fAdeptTransport;
  AdePTConfiguration *const fAdePTConfiguration;
  int fVerbosity{0};
  unsigned int fTrackCounter{0};
  int fCurrentEventID{0};
  bool fAdePTInitialized{false};
  bool fSpeedOfLight{false};
#ifdef ENABLE_POWER_METER
  bool fPowerMeterRunning{false};
#endif
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
