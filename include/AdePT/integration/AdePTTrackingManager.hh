// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef AdePTTrackingManager_h
#define AdePTTrackingManager_h 1

#include "G4VTrackingManager.hh"
#include "globals.hh"
#include <AdePT/core/AdePTTransport.h>
#include "AdePT/copcore/SystemOfUnits.h"

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

  /// Set verbosity for integration
  void SetVerbosity(int verbosity) { fVerbosity = verbosity; }

  // Set the AdePTTransport instance
  void SetAdePTTransport(AdePTTransport *adept) { fAdept = adept; }

private:

  /// @brief Steps a particle using the generic G4 tracking, until it dies or enters a user-defined 
  /// GPU region, in which case tracking is delegated to AdePT
  /// @details In order to be able to send tracks back and forth between the custom tracking and the generic 
  /// G4 tracking, this function has to provide the functionality of G4TrackingManager::ProcessOneTrack, in 
  /// addition to checking whether the track is in a GPU Region
  void ProcessTrack(G4Track *aTrack);

  /// @brief Steps a track using the Generic G4TrackingManager until it enters a GPU region or stops
  void StepInHostRegion(G4Track *aTrack);

  G4Region *fPlaceholderRegion{nullptr};
  AdePTTransport *fAdept;
  int fVerbosity{0};
  G4double ProductionCut = 0.7 * copcore::units::mm;
  int MCIndex[100];
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
