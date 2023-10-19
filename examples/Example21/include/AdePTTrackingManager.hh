#ifndef AdePTTrackingManager_h
#define AdePTTrackingManager_h 1

#include "G4VTrackingManager.hh"
#include "globals.hh"
#include "AdeptIntegration.h"

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

  void SetSensitiveVolumes(std::unordered_map<std::string, int> *sv) { sensitive_volume_index = sv; }

  void SetScoringMap(std::unordered_map<const G4VPhysicalVolume *, int> *sm) { fScoringMap = sm; }

  // Set total number of track slots on GPU
  void SetTrackSlots(double value) { fTrackSlotsGPU = value; }


private:
  /// AdePT integration
  AdeptIntegration *fAdept;

/// Region where it applies
G4Region *fRegion{nullptr};

/// Verbosity
int fVerbosity{0};

/// AdePT buffer threshold
int fBufferThreshold{20};

G4double ProductionCut = 0.7 * copcore::units::mm;

int MCIndex[100];
double fTrackSlotsGPU{1}; ///< Total number of track slots allocated on GPU (in millions)
std::unordered_map<std::string, int> *sensitive_volume_index;
std::unordered_map<const G4VPhysicalVolume *, int> *fScoringMap;

};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
