// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

///   The AdePT-Geant4 integration layer
///   - Initialization and checking of VecGeom geometry
///   - G4Hit reconstruction from GPU hits
///   - Processing of reconstructed hits using the G4 Sensitive Detector

#ifndef ADEPTGEANT4_INTEGRATION_H
#define ADEPTGEANT4_INTEGRATION_H

#include <unordered_map>

#include <G4HepEmState.hh>

#include <AdePT/core/CommonStruct.h>
#include <AdePT/core/HostScoringStruct.cuh>

#include <G4VPhysicalVolume.hh>
#include <G4LogicalVolume.hh>
#include <G4VPhysicalVolume.hh>
#include <G4NavigationHistory.hh>
#include <G4Step.hh>
#include <G4Event.hh>
#include <G4EventManager.hh>

#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/volumes/LogicalVolume.h>

class AdePTGeant4Integration {
public:
  AdePTGeant4Integration()  = default;
  ~AdePTGeant4Integration();

  /// @brief Initializes VecGeom geometry
  /// @details Currently VecGeom geometry is initialized by loading it from a GDML file,
  /// however ideally this function will call the G4 to VecGeom geometry converter
  static void CreateVecGeomWorld(/*Temporary parameter*/ std::string filename);

  /// @brief This function compares G4 and VecGeom geometries and reports any differences
  static void CheckGeometry(G4HepEmState *hepEmState);

  /// @brief Fills the auxiliary data needed for AdePT
  static void InitVolAuxData(adeptint::VolAuxData *volAuxData, G4HepEmState *hepEmState, bool trackInAllRegions,
                             std::vector<std::string> *gpuRegionNames);

  /// @brief Initializes the mapping of VecGeom to G4 volumes for sensitive volumes and their parents
  void InitScoringData(adeptint::VolAuxData *volAuxData);

  /// @brief Reconstructs GPU hits on host and calls the user-defined sensitive detector code
  void ProcessGPUHits(HostScoring &aScoring, HostScoring::Stats &aStats);

  /// @brief Takes a buffer of tracks coming from the device and gives them back to Geant4
  void ReturnTracks(std::vector<adeptint::TrackData> *tracksFromDevice, int debugLevel);

  /// @brief Returns the Z value of the user-defined uniform magnetic field
  /// @details This function can only be called when the user-defined field is a G4UniformMagField
  double GetUniformFieldZ();

  int GetEventID() { return G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID(); }

  int GetThreadID() { return G4Threading::G4GetThreadId(); }

private:
  /// @brief Reconstruct G4TouchableHistory from a VecGeom Navigation index
  void FillG4NavigationHistory(vecgeom::NavigationState aNavState, G4NavigationHistory *aG4NavigationHistory);

  void FillG4Step(GPUHit *aGPUHit, G4Step *aG4Step, G4TouchableHandle &aPreG4TouchableHandle,
                  G4TouchableHandle &aPostG4TouchableHandle);

  std::unordered_map<size_t, const G4VPhysicalVolume *> fglobal_vecgeom_to_g4_map; ///< Maps Vecgeom PV IDs to G4 PV IDs

  bool fScoringObjectsInitialized{false};
  G4NavigationHistory *fPreG4NavigationHistory{nullptr};
  G4NavigationHistory *fPostG4NavigationHistory{nullptr};
  G4Step *fG4Step{nullptr};
  G4TouchableHandle fPreG4TouchableHistoryHandle;
  G4TouchableHandle fPostG4TouchableHistoryHandle;
  G4Track *fElectronTrack{nullptr};
  G4Track *fPositronTrack{nullptr};
  G4Track *fGammaTrack{nullptr};
};

#endif
