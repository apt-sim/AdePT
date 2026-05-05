// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_GEOMETRY_BRIDGE_HH
#define ADEPT_GEOMETRY_BRIDGE_HH

#include <AdePT/transport/geometry/GeometryAuxData.hh>
#include <AdePT/g4integration/tracking_managers/G4HepEmTrackingManagerSpecialized.hh>

#include <VecGeom/base/Config.h>

#include <string>
#include <vector>

class G4LogicalVolume;
class G4VPhysicalVolume;
struct G4HepEmData;

/// @brief Global bridge between Geant4 host geometry and the VecGeom world used by AdePT.
/// @details
/// The geometry world and the VecGeom-to-Geant4 lookup tables are process-global,
/// so this bridge provides the corresponding global services needed during setup and
/// touchable reconstruction.
class AdePTGeometryBridge {
public:
#ifdef VECGEOM_GDML_SUPPORT
  /// @brief Initializes the VecGeom world from a GDML file.
  /// @details
  /// This is the temporary path used when a standalone VecGeom GDML description is
  /// provided instead of converting the Geant4 world through G4VG.
  static void CreateVecGeomWorld(std::string filename);
#endif

  /// @brief Converts the Geant4 world volume into the process-global VecGeom world.
  /// @throws std::runtime_error if the input Geant4 world or resulting VecGeom world is null.
  static void CreateVecGeomWorld(G4VPhysicalVolume const *physvol);

  /// @brief Verifies that the Geant4 and VecGeom geometries match.
  static void CheckGeometry(G4HepEmData const *hepEmData);

  /// @brief Fills the auxiliary per-volume data needed by AdePT.
  static void InitVolAuxData(adeptint::VolAuxData *volAuxData, G4HepEmData const *hepEmData,
                             G4HepEmTrackingManagerSpecialized *hepEmTM, bool trackInAllRegions,
                             std::vector<std::string> const *gpuRegionNames,
                             std::vector<std::string> const &deadRegionNames, adeptint::WDTHostRaw &wdtRaw);

  /// @brief Pack the Woodcock tracking data from the sparse host-side map into arrays that can be copied to the GPU.
  /// @param wdtRaw Raw WDT data collected during geometry traversal.
  /// @return Packed, dense WDT data ready to be handed to the transport for device upload.
  static adeptint::WDTHostPacked PackWDT(adeptint::WDTHostRaw const &wdtRaw);

  /// @brief Returns the Geant4 placed volume matching a VecGeom placed volume.
  /// @throws std::runtime_error if the VecGeom placed volume is not present in the global lookup table.
  static G4VPhysicalVolume const *GetG4PhysicalVolume(vecgeom::VPlacedVolume const *placedVolume);

private:
  /// @brief Builds the lookup tables from VecGeom placed/logical volume ids to Geant4 volumes.
  static void MapVecGeomToG4(std::vector<G4VPhysicalVolume const *> &vecgeomPvToG4Map,
                             std::vector<G4LogicalVolume const *> &vecgeomLvToG4Map);

  static std::vector<G4VPhysicalVolume const *> fGlobalVecGeomPvToG4Map;
  static std::vector<G4LogicalVolume const *> fGlobalVecGeomLvToG4Map;
};

#endif
