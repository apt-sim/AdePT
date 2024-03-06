// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef AdePTTrackingManager_h
#define AdePTTrackingManager_h 1

#include "G4VTrackingManager.hh"
#include "G4RegionStore.hh"

#include "globals.hh"
#include <AdePT/core/AdePTTransport.h>
#include "AdePT/copcore/SystemOfUnits.h"
#include <AdePT/integration/AdePTGeant4Integration.hh>

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

  // Set the AdePTTransport instance, also initializes the GPU region list
  void SetAdePTTransport(AdePTTransport<AdePTGeant4Integration> *adept)
  {
    fAdept = adept;
    if (!adept->GetTrackInAllRegions()) {
      for (std::string regionName : *(adept->GetGPURegionNames())) {
        G4cout << "AdePTTrackingManager: Marking " << regionName << " as a GPU Region" << G4endl;
        G4Region *region = G4RegionStore::GetInstance()->GetRegion(regionName);
        if (region != nullptr)
          fGPURegions.push_back(region);
        else
          G4Exception("AdePTTrackingManager", "Invalid parameter", FatalErrorInArgument,
                      ("Region given to /adept/addGPURegion: " + regionName + " Not found\n").c_str());
      }
    }
  }

private:
  /// @brief Steps a particle using the generic G4 tracking, until it dies or enters a user-defined
  /// GPU region, in which case tracking is delegated to AdePT
  /// @details In order to be able to send tracks back and forth between the custom tracking and the generic
  /// G4 tracking, this function has to provide the functionality of G4TrackingManager::ProcessOneTrack, in
  /// addition to checking whether the track is in a GPU Region
  void ProcessTrack(G4Track *aTrack);

  /// @brief Steps a track using the Generic G4TrackingManager until it enters a GPU region or stops
  void StepInHostRegion(G4Track *aTrack);

  std::vector<G4Region *> fGPURegions{};
  AdePTTransport<AdePTGeant4Integration> *fAdept;
  int fVerbosity{0};
  G4double ProductionCut = 0.7 * copcore::units::mm;
  int MCIndex[100];
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
