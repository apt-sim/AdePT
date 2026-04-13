// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/AdePTGeometryBridge.hh>

#include <VecGeom/management/GeoManager.h>
#ifdef VECGEOM_GDML_SUPPORT
#include <VecGeom/gdml/Frontend.h>
#endif
#include <VecGeom/navigation/NavigationState.h>

#include <G4HepEmData.hh>
#include <G4HepEmMatCutData.hh>
#include <G4LogicalVolume.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4PVReplica.hh>
#include <G4RegionStore.hh>
#include <G4ReplicaNavigation.hh>
#include <G4SystemOfUnits.hh>
#include <G4TransportationManager.hh>
#include <G4VG.hh>
#include <G4VSensitiveDetector.hh>
#include <G4ios.hh>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

std::vector<G4VPhysicalVolume const *> AdePTGeometryBridge::fGlobalVecGeomPvToG4Map;
std::vector<G4LogicalVolume const *> AdePTGeometryBridge::fGlobalVecGeomLvToG4Map;

void AdePTGeometryBridge::MapVecGeomToG4(std::vector<G4VPhysicalVolume const *> &vecgeomPvToG4Map,
                                         std::vector<G4LogicalVolume const *> &vecgeomLvToG4Map)
{
  const G4VPhysicalVolume *g4world =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  const vecgeom::VPlacedVolume *vecgeomWorld = vecgeom::GeoManager::Instance().GetWorld();

  // Recursive geometry visitor lambda matching one by one Geant4 and VecGeom logical volumes.
  using VisitFn         = std::function<void(G4VPhysicalVolume const *, vecgeom::VPlacedVolume const *)>;
  VisitFn visitGeometry = [&](G4VPhysicalVolume const *g4_pvol, vecgeom::VPlacedVolume const *vg_pvol) {
    const auto g4_lvol = g4_pvol->GetLogicalVolume();
    const auto vg_lvol = vg_pvol->GetLogicalVolume();

    // Initialize mapping of VecGeom placed-volume ids to Geant4 physical volumes.
    vecgeomPvToG4Map.resize(std::max<std::size_t>(vecgeomPvToG4Map.size(), vg_pvol->id() + 1), nullptr);
    vecgeomPvToG4Map[vg_pvol->id()] = g4_pvol;

    // Initialize mapping of VecGeom logical-volume ids to Geant4 logical volumes.
    vecgeomLvToG4Map.resize(std::max<std::size_t>(vecgeomLvToG4Map.size(), vg_lvol->id() + 1), nullptr);
    vecgeomLvToG4Map[vg_lvol->id()] = g4_lvol;

    // Now do the daughters.
    for (size_t id = 0; id < g4_lvol->GetNoDaughters(); ++id) {
      visitGeometry(g4_lvol->GetDaughter(id), vg_lvol->GetDaughters()[id]);
    }
  };

  visitGeometry(g4world, vecgeomWorld);
}

#ifdef VECGEOM_GDML_SUPPORT
/// @brief Initialize the process-global VecGeom world from a GDML file.
void AdePTGeometryBridge::CreateVecGeomWorld(std::string filename)
{
  // Import the GDML file into VecGeom.
  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(0);
  vgdml::Parser vgdmlParser;
  auto middleWare = vgdmlParser.Load(filename, false, mm);
  if (middleWare == nullptr) {
    std::cerr << "Failed to read geometry from GDML file '" << filename << "'" << G4endl;
    return;
  }

  // Generate the mapping of VecGeom volume ids to Geant4 physical volumes.
  MapVecGeomToG4(fGlobalVecGeomPvToG4Map, fGlobalVecGeomLvToG4Map);

  const vecgeom::VPlacedVolume *vecgeomWorld = vecgeom::GeoManager::Instance().GetWorld();
  if (vecgeomWorld == nullptr) {
    std::cerr << "GeoManager vecgeomWorld volume is nullptr" << G4endl;
    return;
  }
}
#endif

/// @brief Convert the Geant4 world into the process-global VecGeom world via G4VG.
void AdePTGeometryBridge::CreateVecGeomWorld(G4VPhysicalVolume const *physvol)
{
  // EXPECT: a non-null input volume.
  if (physvol == nullptr) {
    throw std::runtime_error("AdePTGeometryBridge::CreateVecGeomWorld: Input Geant4 physical volume is nullptr");
  }

  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(0);
  g4vg::Options options;
  options.reflection_factory = false;
  auto conversion            = g4vg::convert(physvol, options);
  vecgeom::GeoManager::Instance().SetWorldAndClose(conversion.world);

  // Get the mapping of VecGeom volume ids to Geant4 physical volumes from G4VG.
  fGlobalVecGeomPvToG4Map = conversion.physical_volumes;
  fGlobalVecGeomLvToG4Map = conversion.logical_volumes;

  // EXPECT: we finish with a non-null VecGeom host geometry.
  if (vecgeom::GeoManager::Instance().GetWorld() == nullptr) {
    throw std::runtime_error("AdePTGeometryBridge::CreateVecGeomWorld: Output VecGeom world is nullptr");
  }
}

namespace {
struct VisitContext {
  const int *g4tohepmcindex;
  std::size_t nvolumes;
  G4HepEmData const *hepEmData;
};

/// @brief Recursively compare the Geant4 and VecGeom geometry trees.
void VisitGeometryForChecks(G4VPhysicalVolume const *g4_pvol, vecgeom::VPlacedVolume const *vg_pvol,
                            const VisitContext &context)
{
  const auto g4_lvol = g4_pvol->GetLogicalVolume();
  const auto vg_lvol = vg_pvol->GetLogicalVolume();

  // Geant4 parameterised/replica volumes are represented with direct placements in VecGeom.
  // To accurately compare the number of daughters, sum multiplicity on the Geant4 side.
  const size_t nd     = g4_lvol->GetNoDaughters();
  size_t nd_converted = 0;
  for (size_t daughter_id = 0; daughter_id < nd; ++daughter_id) {
    nd_converted += g4_lvol->GetDaughter(daughter_id)->GetMultiplicity();
  }

  const auto daughters = vg_lvol->GetDaughters();
  if (nd_converted != daughters.size()) {
    throw std::runtime_error("Fatal: CheckGeometry: Mismatch in number of daughters");
  }

  // Check whether transformations are matching.
  // As above, with parameterized/replica volumes we need to compare the transforms between
  // the VecGeom direct placement and that for the parameterised/replicated volume given the
  // same copy number as that of the VecGeom physical volume.
  // NOTE:
  // 1. This needs a const_cast because the current Geant4 API computes the transform by
  //    mutating the physical volume.
  // 2. This does modify the physical volume, but navigation will reset things afterwards.
  if (G4VPVParameterisation *param = g4_pvol->GetParameterisation()) {
    param->ComputeTransformation(vg_pvol->GetCopyNo(), const_cast<G4VPhysicalVolume *>(g4_pvol));
  } else if (auto *replica = dynamic_cast<G4PVReplica *>(const_cast<G4VPhysicalVolume *>(g4_pvol))) {
    G4ReplicaNavigation nav;
    nav.ComputeTransformation(vg_pvol->GetCopyNo(), replica);
  }

  const auto g4trans            = g4_pvol->GetTranslation();
  const G4RotationMatrix *g4rot = g4_pvol->GetRotation();
  G4RotationMatrix idrot;
  const auto vgtransformation = vg_pvol->GetTransformation();
  constexpr double epsil      = 1.e-8;

  for (int i = 0; i < 3; ++i) {
    if (std::abs(g4trans[i] - vgtransformation->Translation(i)) > epsil) {
      throw std::runtime_error(
          std::string("Fatal: CheckGeometry: Mismatch between Geant4 translation for physical volume") +
          vg_pvol->GetName());
    }
  }

  // Check if VecGeom and Geant4 local transformations are matching.
  // This is not optimized and will re-check already-checked placed volumes when
  // revisiting the same volumes in different branches.
  if (!g4rot) g4rot = &idrot;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      int i = row + 3 * col;
      if (std::abs((*g4rot)(row, col) - vgtransformation->Rotation(i)) > epsil) {
        throw std::runtime_error(
            std::string("Fatal: CheckGeometry: Mismatch between Geant4 rotation for physical volume") +
            vg_pvol->GetName());
      }
    }
  }

  if (g4_lvol->GetMaterialCutsCouple() == nullptr) {
    throw std::runtime_error("Fatal: CheckGeometry: G4LogicalVolume " + std::string(g4_lvol->GetName()) +
                             std::string(" has no material-cuts couple"));
  }
  const int g4mcindex    = g4_lvol->GetMaterialCutsCouple()->GetIndex();
  const int hepemmcindex = context.g4tohepmcindex[g4mcindex];
  // Check consistency with G4HepEm data.
  if (context.hepEmData->fTheMatCutData->fMatCutData[hepemmcindex].fG4MatCutIndex != g4mcindex) {
    throw std::runtime_error("Fatal: CheckGeometry: Mismatch between Geant4 mcindex and corresponding G4HepEm index");
  }
  if (vg_lvol->id() >= context.nvolumes) {
    throw std::runtime_error("Fatal: CheckGeometry: Volume id larger than number of volumes");
  }

  // Now do the daughters.
  for (size_t id = 0; id < g4_lvol->GetNoDaughters(); ++id) {
    VisitGeometryForChecks(g4_lvol->GetDaughter(id), vg_lvol->GetDaughters()[id], context);
  }
}
} // namespace

/// @brief Compare the Geant4 and VecGeom host geometries for consistency.
void AdePTGeometryBridge::CheckGeometry(G4HepEmData const *hepEmData)
{
  const G4VPhysicalVolume *g4world =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  const vecgeom::VPlacedVolume *vecgeomWorld = vecgeom::GeoManager::Instance().GetWorld();
  const int *g4tohepmcindex                  = hepEmData->fTheMatCutData->fG4MCIndexToHepEmMCIndex;
  const auto nvolumes                        = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();

  std::cout << "Visiting geometry ...\n";
  const VisitContext context{g4tohepmcindex, nvolumes, hepEmData};
  VisitGeometryForChecks(g4world, vecgeomWorld, context);
  std::cout << "Visiting geometry done\n";
}

/// @brief Fill the auxiliary per-volume transport metadata used by AdePT.
void AdePTGeometryBridge::InitVolAuxData(adeptint::VolAuxData *volAuxData, G4HepEmData const *hepEmData,
                                         G4HepEmTrackingManagerSpecialized *hepEmTM, bool trackInAllRegions,
                                         std::vector<std::string> const *gpuRegionNames, adeptint::WDTHostRaw &wdtRaw)
{
  // Note: the hepEmTM must be passed explicitly here, since this is now a stateless
  // global bridge helper and therefore cannot reach any per-thread integration members.
  wdtRaw.ekinMin = (float)hepEmTM->GetWDTKineticEnergyLimit();

  const G4VPhysicalVolume *g4world =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  const vecgeom::VPlacedVolume *vecgeomWorld = vecgeom::GeoManager::Instance().GetWorld();
  const int *g4tohepmcindex                  = hepEmData->fTheMatCutData->fG4MCIndexToHepEmMCIndex;

  // We need to go from region names to G4Region objects.
  std::vector<G4Region *> gpuRegions{};
  if (!trackInAllRegions) {
    for (const std::string &regionName : *gpuRegionNames) {
      gpuRegions.push_back(G4RegionStore::GetInstance()->GetRegion(regionName));
    }
  }

#if defined(ADEPT_STEPACTION_TYPE) && (ADEPT_STEPACTION_TYPE == 3)
  std::vector<bool> atlasPhotonRRSeen(vecgeom::GeoManager::Instance().GetRegisteredVolumesCount(), false);
#endif

  // Recursive geometry visitor lambda matching one by one Geant4 and VecGeom logical volumes.
  using VisitFn =
      std::function<void(G4VPhysicalVolume const *, vecgeom::VPlacedVolume const *, vecgeom::NavigationState)>;
  VisitFn visitGeometry = [&](G4VPhysicalVolume const *g4_pvol, vecgeom::VPlacedVolume const *vg_pvol,
                              vecgeom::NavigationState currentNavState) {
    const auto g4_lvol = g4_pvol->GetLogicalVolume();
    const auto vg_lvol = vg_pvol->GetLogicalVolume();

    // Push this placed volume into the running navigation state.
    currentNavState.Push(vg_pvol);
    currentNavState.SetBoundaryState(false);

    // Fill the material-cuts-couple index in the auxiliary array.
    int g4mcindex                      = g4_lvol->GetMaterialCutsCouple()->GetIndex();
    int hepemmcindex                   = g4tohepmcindex[g4mcindex];
    volAuxData[vg_lvol->id()].fMCIndex = hepemmcindex;

    const int regionId = g4_lvol->GetRegion()->GetInstanceID();
    // Check if the region is a Woodcock tracking region in G4HepEm.
    if (hepEmTM->IsWDTRegion(regionId)) {
      // Check if this logical volume is one of the declared WDT root LVs for this region.
      const int rootIMC = hepEmTM->GetWDTCoupleHepEmIndex(regionId, g4_lvol->GetInstanceID());
      if (rootIMC >= 0) {
        // This placed volume belongs to a WDT root LV, so record a WDTRoot.
        int idx = (int)wdtRaw.roots.size();
        wdtRaw.roots.push_back(adeptint::WDTRoot{currentNavState, rootIMC});
        wdtRaw.regionToRootIndices[regionId].push_back(idx);
      }
    }

    // Check if the volume belongs to a GPU region.
    if (!trackInAllRegions) {
      for (G4Region *gpuRegion : gpuRegions) {
        if (g4_lvol->GetRegion() == gpuRegion) {
          volAuxData[vg_lvol->id()].fGPUregionId = g4_lvol->GetRegion()->GetInstanceID();
        }
      }
    } else {
      volAuxData[vg_lvol->id()].fGPUregionId = g4_lvol->GetRegion()->GetInstanceID();
    }

    if (g4_lvol->GetSensitiveDetector() != nullptr) {
      if (volAuxData[vg_lvol->id()].fSensIndex < 0) {
        G4cout << "VecGeom: Making " << vg_lvol->GetName() << " sensitive" << G4endl;
      }
      volAuxData[vg_lvol->id()].fSensIndex = 1;
    }

#if defined(ADEPT_STEPACTION_TYPE) && (ADEPT_STEPACTION_TYPE == 3)
    const bool atlasPhotonRR = g4_pvol->GetName().rfind("LAr", 0) == 0;
    auto &atlasPhotonRRFlag  = volAuxData[vg_lvol->id()].fAtlasPhotonRussianRoulette;
    const bool alreadySeen   = atlasPhotonRRSeen[vg_lvol->id()];
    if (!alreadySeen) {
      atlasPhotonRRFlag                = atlasPhotonRR;
      atlasPhotonRRSeen[vg_lvol->id()] = true;
    } else if (atlasPhotonRRFlag != atlasPhotonRR) {
      G4cerr << "ATLAS photon Russian Roulette flag is inconsistent for logical volume '" << g4_lvol->GetName()
             << "': saw both LAr and non-LAr placed-volume names while building VolAuxData." << G4endl;
    }
#endif

    // Now do the daughters.
    for (size_t id = 0; id < g4_lvol->GetNoDaughters(); ++id) {
      visitGeometry(g4_lvol->GetDaughter(id), vg_lvol->GetDaughters()[id], currentNavState);
    }

    // Pop the navigation state before returning.
    currentNavState.Pop();
  };

  // Initialize the root navigation state.
  vecgeom::NavigationState rootNavState;
  visitGeometry(g4world, vecgeomWorld, rootNavState);

  auto findRegionName = [](int rid) -> std::string {
    for (auto *r : *G4RegionStore::GetInstance()) {
      if (r && r->GetInstanceID() == rid) return r->GetName();
    }
    return std::string("<unknown>");
  };

  std::cout << "\n=== Woodcock tracking summary (host) ===\n";
  std::cout << "KineticEnergyLimit = " << wdtRaw.ekinMin << " [G4 units]\n";
  std::cout << "Total WDT roots found: " << wdtRaw.roots.size() << std::endl;
  std::cout << "Regions with WDT: " << wdtRaw.regionToRootIndices.size() << std::endl;

  if (wdtRaw.regionToRootIndices.empty()) {
    std::cout << "  (none)\n";
  } else {
    for (const auto &[regionId, indices] : wdtRaw.regionToRootIndices) {
      std::cout << "\nRegionID " << regionId << "  (" << findRegionName(regionId) << "): " << indices.size()
                << " root placed-volume(s)\n";

      for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto &root = wdtRaw.roots[indices[i]];
        std::cout << "  [" << i << "] hepemIMC=" << root.hepemIMC << "\n";
        std::cout << "      NavState (level=" << root.root.GetLevel() << "):\n";
        // vecgeom::NavigationState::Print() prints the full stack.
        root.root.Print();
      }
    }
  }
  std::cout << "=== End Woodcock tracking summary ===\n\n";
}

adeptint::WDTHostPacked AdePTGeometryBridge::PackWDT(adeptint::WDTHostRaw const &wdtRaw)
{
  adeptint::WDTHostPacked packed;

  int maxRegionId = -1;
  for (auto *region : *G4RegionStore::GetInstance())
    if (region) maxRegionId = std::max(maxRegionId, region->GetInstanceID());

  packed.regionToWDT.assign(maxRegionId + 1, -1);

  packed.roots.reserve(wdtRaw.roots.size());
  packed.regions.reserve(wdtRaw.regionToRootIndices.size());

  int runningOffset = 0;
  for (auto const &[regionId, rootIndices] : wdtRaw.regionToRootIndices) {
    packed.regionToWDT[regionId] = static_cast<int>(packed.regions.size());
    packed.regions.push_back(adeptint::WDTRegion{runningOffset, static_cast<int>(rootIndices.size()), wdtRaw.ekinMin});

    for (int index : rootIndices) {
      packed.roots.push_back(wdtRaw.roots[index]);
    }
    runningOffset += static_cast<int>(rootIndices.size());
  }

  return packed;
}

/// @brief Resolve the Geant4 placed volume associated with a VecGeom placed volume.
G4VPhysicalVolume const *AdePTGeometryBridge::GetG4PhysicalVolume(vecgeom::VPlacedVolume const *placedVolume)
{
  if (placedVolume == nullptr) {
    throw std::runtime_error("AdePTGeometryBridge::GetG4PhysicalVolume: Input VecGeom placed volume is nullptr");
  }
  if (placedVolume->id() >= fGlobalVecGeomPvToG4Map.size()) {
    throw std::runtime_error(
        "AdePTGeometryBridge::GetG4PhysicalVolume: VecGeom placed volume id is outside the lookup table");
  }

  auto *g4Volume = fGlobalVecGeomPvToG4Map[placedVolume->id()];
  if (g4Volume == nullptr) {
    throw std::runtime_error(
        "AdePTGeometryBridge::GetG4PhysicalVolume: VecGeom placed volume not found in Geant4 mapping");
  }
  return g4Volume;
}
