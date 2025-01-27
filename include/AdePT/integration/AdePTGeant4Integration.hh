// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

///   The AdePT-Geant4 integration layer
///   - Initialization and checking of VecGeom geometry
///   - G4Hit reconstruction from GPU hits
///   - Processing of reconstructed hits using the G4 Sensitive Detector

#ifndef ADEPTGEANT4_INTEGRATION_H
#define ADEPTGEANT4_INTEGRATION_H

#include <AdePT/core/CommonStruct.h>
#include <AdePT/core/HostScoringStruct.cuh>

#include <G4EventManager.hh>
#include <G4Event.hh>

#include <unordered_map>

struct G4HepEmState;

namespace AdePTGeant4Integration_detail {
struct ScoringObjects;
struct Deleter {
  void operator()(ScoringObjects *ptr);
};
} // namespace AdePTGeant4Integration_detail

class AdePTGeant4Integration {
public:
  AdePTGeant4Integration() = default;
  ~AdePTGeant4Integration();

  /// @brief Initializes VecGeom geometry
  /// @details Currently VecGeom geometry is initialized by loading it from a GDML file,
  /// however ideally this function will call the G4 to VecGeom geometry converter
  static void CreateVecGeomWorld(/*Temporary parameter*/ std::string filename);

  /// @brief This function compares G4 and VecGeom geometries and reports any differences
  static void CheckGeometry(G4HepEmState *hepEmState);

  /// @brief Fills the auxiliary data needed for AdePT
  static void InitVolAuxData(adeptint::VolAuxData *volAuxData, G4HepEmState *hepEmState, bool trackInAllRegions,
                             std::vector<std::string> const *gpuRegionNames);

  /// @brief Initializes the mapping of VecGeom to G4 volumes for sensitive volumes and their parents
  void InitScoringData();

  /// @brief Reconstructs GPU hits on host and calls the user-defined sensitive detector code
  void ProcessGPUHit(GPUHit const &hit);

  /// @brief Takes a range of tracks coming from the device and gives them back to Geant4
  template <typename Iterator>
  void ReturnTracks(Iterator begin, Iterator end, int debugLevel) const
  {
    if (debugLevel > 1) {
      G4cout << "Returning " << end - begin << " tracks from device" << G4endl;
    }
    for (Iterator it = begin; it != end; ++it) {
      ReturnTrack(*it, it - begin, debugLevel);
    }
  }

  /// @brief Returns the Z value of the user-defined uniform magnetic field
  /// @details This function can only be called when the user-defined field is a G4UniformMagField
  double GetUniformFieldZ() const;

  int GetEventID() const { return G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID(); }

  int GetThreadID() const { return G4Threading::G4GetThreadId(); }

private:
  /// @brief Reconstruct G4TouchableHistory from a VecGeom Navigation index
  void FillG4NavigationHistory(vecgeom::NavigationState aNavState, G4NavigationHistory *aG4NavigationHistory) const;

  void FillG4Step(GPUHit const *aGPUHit, G4Step *aG4Step, G4TouchableHandle &aPreG4TouchableHandle,
                  G4TouchableHandle &aPostG4TouchableHandle) const;

  void ReturnTrack(adeptint::TrackData const &track, unsigned int trackIndex, int debugLevel) const;

  std::unordered_map<size_t,
                     const G4VPhysicalVolume *> fglobal_vecgeom_to_g4_map; ///< Maps Vecgeom PV IDs to G4 PV IDs
  std::unique_ptr<AdePTGeant4Integration_detail::ScoringObjects, AdePTGeant4Integration_detail::Deleter>
      fScoringObjects{nullptr};
};

#endif
