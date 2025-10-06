// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

///   The AdePT-Geant4 integration layer
///   - Initialization and checking of VecGeom geometry
///   - G4Hit reconstruction from GPU hits
///   - Processing of reconstructed hits using the G4 Sensitive Detector

#ifndef ADEPTGEANT4_INTEGRATION_H
#define ADEPTGEANT4_INTEGRATION_H

#include <AdePT/core/CommonStruct.h>
#include <AdePT/core/ScoringCommons.hh>
#include <AdePT/integration/G4HepEmTrackingManagerSpecialized.hh>
#include <AdePT/integration/HostTrackDataMapper.hh>

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
  explicit AdePTGeant4Integration() : fHostTrackDataMapper(std::make_unique<HostTrackDataMapper>()) {}
  ~AdePTGeant4Integration();

  AdePTGeant4Integration(const AdePTGeant4Integration &)            = delete;
  AdePTGeant4Integration &operator=(const AdePTGeant4Integration &) = delete;

  AdePTGeant4Integration(AdePTGeant4Integration &&)            = default;
  AdePTGeant4Integration &operator=(AdePTGeant4Integration &&) = default;

  /// @brief Initializes VecGeom geometry
  /// @details Currently VecGeom geometry is initialized by loading it from a GDML file,
  /// however ideally this function will call the G4 to VecGeom geometry converter
  static void CreateVecGeomWorld(/*Temporary parameter*/ std::string filename);

  /// @brief Construct VecGeom geometry from Geant4 physical volume
  /// @details This calls the G4VG converter
  /// @throws std::runtime_error if input or output volumes are nullptr
  static void CreateVecGeomWorld(G4VPhysicalVolume const *physvol);

  /// @brief This function compares G4 and VecGeom geometries and reports any differences
  static void CheckGeometry(G4HepEmState *hepEmState);

  /// @brief Fills the auxiliary data needed for AdePT
  static void InitVolAuxData(adeptint::VolAuxData *volAuxData, G4HepEmState *hepEmState, bool trackInAllRegions,
                             std::vector<std::string> const *gpuRegionNames);

  /// @brief Returns a mapping of VecGeom placed volume IDs to Geant4 physical volumes and a mapping of VecGeom logical
  /// volume IDs to Geant4 logical volumes
  static void MapVecGeomToG4(std::vector<G4VPhysicalVolume const *> &vecgeomPvToG4Map,
                             std::vector<G4LogicalVolume const *> &vecgeomLvToG4Map);

  /// @brief Reconstructs GPU hits on host and calls the user-defined sensitive detector code
  void ProcessGPUStep(GPUHit const &hit, bool const callUserSteppingAction = false,
                      bool const callUserTrackingaction = false);

  /// @brief Takes a range of tracks coming from the device and gives them back to Geant4
  template <typename Iterator>
  void ReturnTracks(Iterator begin, Iterator end, int debugLevel, bool callUserActions = false) const
  {
    if (debugLevel > 1) {
      G4cout << "Returning " << end - begin << " tracks from device" << G4endl;
    }
    for (Iterator it = begin; it != end; ++it) {
      ReturnTrack(*it, it - begin, debugLevel, callUserActions);
    }
  }

  /// @brief Returns the Z value of the user-defined uniform magnetic field
  /// @details This function can only be called when the user-defined field is a G4UniformMagField
  vecgeom::Vector3D<float> GetUniformField() const;

  int GetEventID() const { return G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID(); }

  int GetThreadID() const { return G4Threading::G4GetThreadId(); }

  HostTrackDataMapper &GetHostTrackDataMapper() { return *fHostTrackDataMapper; }

  void SetHepEmTrackingManager(G4HepEmTrackingManagerSpecialized *hepEmTrackingManager)
  {
    fHepEmTrackingManager = hepEmTrackingManager;
  }

private:
  /// @brief Reconstruct G4TouchableHistory from a VecGeom Navigation index
  void FillG4NavigationHistory(vecgeom::NavigationState aNavState, G4NavigationHistory &aG4NavigationHistory) const;

  void FillG4Step(GPUHit const *aGPUHit, G4Step *aG4Step, G4TouchableHandle &aPreG4TouchableHandle,
                  G4TouchableHandle &aPostG4TouchableHandle, G4StepStatus aPreStepStatus, G4StepStatus aPostStepStatus,
                  bool callUserTrackingAction, bool callUserSteppingAction) const;

  void ReturnTrack(adeptint::TrackData const &track, unsigned int trackIndex, int debugLevel,
                   bool callUserActions = false) const;

  // pointer to specialized G4HepEmTrackingManager. Owned by AdePTTrackingManager,
  // this is just a reference to handle gamma-/lepton-nuclear reactions
  G4HepEmTrackingManagerSpecialized *fHepEmTrackingManager{nullptr};

  // helper class to provide the CPU-only data for the returning GPU tracks
  std::unique_ptr<HostTrackDataMapper> fHostTrackDataMapper;

  static std::vector<G4VPhysicalVolume const *> fglobal_vecgeom_pv_to_g4_map;
  static std::vector<G4LogicalVolume const *> fglobal_vecgeom_lv_to_g4_map;
  std::unique_ptr<AdePTGeant4Integration_detail::ScoringObjects, AdePTGeant4Integration_detail::Deleter>
      fScoringObjects{nullptr};
};

#endif
