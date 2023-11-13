// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef AdePTTrackingManager_h
#define AdePTTrackingManager_h 1

#include "G4VTrackingManager.hh"
#include "globals.hh"
#include "AdeptIntegration.h"
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

  /// Set buffer threshold for AdePT
  void SetBufferThreshold(int value) { fBufferThreshold = value; }

  void SetSensitiveVolumes(std::vector<G4LogicalVolume *> *sv) { fSensitiveLogicalVolumes = sv; }

  void SetScoringMap(std::unordered_map<size_t, size_t> *sm) { fScoringMap = sm; }

  // Set total number of track slots on GPU
  void SetTrackSlots(double value) { fTrackSlotsGPU = value; }

  // Set total number of track slots on GPU and Host
  void SetHitSlots(double value) { fHitSlots = value; }

private:
  /// AdePT integration
  AdeptIntegration *fAdept;

  /// Region where it applies
  G4Region *fRegion{nullptr};

  /// Verbosity
  int fVerbosity{0};

  /// AdePT buffer threshold
  int fBufferThreshold{20};

  bool fInitDone{false};

  G4double ProductionCut = 0.7 * copcore::units::mm;

  int MCIndex[100];
  double fTrackSlotsGPU{1}; ///< Total number of track slots allocated on GPU (in millions)
  double fHitSlots{1};      ///< Total number of hit slots allocated on GPU and Host (in millions)
  std::vector<G4LogicalVolume *> *fSensitiveLogicalVolumes;
  std::unordered_map<size_t, size_t> *fScoringMap;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
