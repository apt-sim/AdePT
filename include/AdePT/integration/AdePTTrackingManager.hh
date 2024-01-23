// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef AdePTTrackingManager_h
#define AdePTTrackingManager_h 1

#include "G4VTrackingManager.hh"
#include "globals.hh"
#include <AdePT/integration/AdePTTransport.h>
#include "CopCore/SystemOfUnits.h"

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
  /// AdePT integration
  AdePTTransport *fAdept;

  /// Verbosity
  int fVerbosity{0};

  G4double ProductionCut = 0.7 * copcore::units::mm;

  int MCIndex[100];
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
