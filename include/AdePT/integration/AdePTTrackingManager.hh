// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef AdePTTrackingManager_h
#define AdePTTrackingManager_h 1

#include "G4VTrackingManager.hh"
#include "G4RegionStore.hh"

#include "globals.hh"
#include <AdePT/core/AdePTTransportInterface.hh>
#ifndef ASYNC_MODE
#include <AdePT/core/AdePTTransport.h>
#else
#include <AdePT/core/AsyncAdePTTransport.hh>
#endif
#include "AdePT/copcore/SystemOfUnits.h"
#include <AdePT/integration/AdePTGeant4Integration.hh>
#include <AdePT/core/AdePTConfiguration.hh>
#include <AdePT/integration/G4HepEmTrackingManagerSpecialized.hh>

#include <memory>
#include <vector>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class AdePTTrackingManager : public G4VTrackingManager {
public:
  AdePTTrackingManager();
  ~AdePTTrackingManager();

  void BuildPhysicsTable(const G4ParticleDefinition &) override;

  void PreparePhysicsTable(const G4ParticleDefinition &) override;

  void HandOverOneTrack(G4Track *aTrack) override;

  void FlushEvent() override;

  void InitializeAdePT();

  /// Set verbosity for integration
  void SetVerbosity(int verbosity) { fVerbosity = verbosity; }

  void SetAdePTConfiguration(AdePTConfiguration *aAdePTConfiguration) { fAdePTConfiguration = aAdePTConfiguration; }

private:
  /// @brief Steps a particle using the generic G4 tracking, until it dies or enters a user-defined
  /// GPU region, in which case tracking is delegated to AdePT
  /// @details In order to be able to send tracks back and forth between the custom tracking and the generic
  /// G4 tracking, this function has to provide the functionality of G4TrackingManager::ProcessOneTrack, in
  /// addition to checking whether the track is in a GPU Region
  void ProcessTrack(G4Track *aTrack);

  /// @brief Steps a track using the Generic G4TrackingManager until it enters a GPU region or stops
  void StepInHostRegion(G4Track *aTrack);

  /// @brief Get the corresponding VecGeom NavigationState from the G4NavigationHistory. The boundary status will be set
  /// to false by default
  /// @param aG4NavigationHistory the given G4NavigationHistory
  /// @return the corresponding vecgeom::NavigationState
  const vecgeom::NavigationState GetVecGeomFromG4State(const G4NavigationHistory &aG4NavigationHistory);

  /// @brief Get the corresponding VecGeom NavigationState from the track's G4NavigationHistory, and set the boundary
  /// status based on the track's state
  /// @param aG4Track The G4Track from which to extract the NavigationState
  /// @return The corresponding vecgeom::NavigationState
  const vecgeom::NavigationState GetVecGeomFromG4State(const G4Track &aG4Track);

  std::unique_ptr<G4HepEmTrackingManagerSpecialized> fHepEmTrackingManager;
  static inline int fNumThreads{0};
  std::set<G4Region const *> fGPURegions{};
  int fVerbosity{0};
// std::shared_ptr<AdePTTransportInterface> fAdeptTransport;
#ifndef ASYNC_MODE
  std::unique_ptr<AdePTTransportInterface> fAdeptTransport;
#else
  std::shared_ptr<AdePTTransportInterface> fAdeptTransport;
#endif
  AdePTConfiguration *fAdePTConfiguration;
  unsigned int fTrackCounter{0};
  int fCurrentEventID{0};
  bool fAdePTInitialized{false};
  bool fSpeedOfLight{false};
};

#ifdef ASYNC_MODE
std::shared_ptr<AsyncAdePT::AsyncAdePTTransport<AdePTGeant4Integration>> GetAdePTInstance();
#endif

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
