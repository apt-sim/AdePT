// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example9.h"

#include <AdePT/ArgParser.h>
#include <CopCore/SystemOfUnits.h>

#include <G4NistManager.hh>
#include <G4Material.hh>

#include <G4Box.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>

#include <G4ParticleTable.hh>
#include <G4Electron.hh>
#include <G4Positron.hh>
#include <G4Gamma.hh>
#include <G4Proton.hh>

#include <G4ProductionCuts.hh>
#include <G4Region.hh>
#include <G4ProductionCutsTable.hh>

#include <G4SystemOfUnits.hh>

#include <VecGeom/base/Config.h>
#include <VecGeom/management/GeoManager.h>
#ifdef VECGEOM_GDML
#include <VecGeom/gdml/Frontend.h>
#endif

static void InitGeant4()
{
  // --- Create materials.
  G4Material *galactic = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
  G4Material *silicon  = G4NistManager::Instance()->FindOrBuildMaterial("G4_Si");
  //
  // --- Define a world.
  G4double worldDim         = 1 * m;
  G4Box *worldBox           = new G4Box("world", worldDim, worldDim, worldDim);
  G4LogicalVolume *worldLog = new G4LogicalVolume(worldBox, galactic, "world");
  G4PVPlacement *world      = new G4PVPlacement(nullptr, {}, worldLog, "world", nullptr, false, 0);
  // --- Define a box.
  G4double boxDim             = 0.5 * m;
  G4double boxPos             = 0.5 * boxDim;
  G4Box *siliconBox           = new G4Box("silicon", boxDim, boxDim, boxDim);
  G4LogicalVolume *siliconLog = new G4LogicalVolume(siliconBox, silicon, "silicon");
  new G4PVPlacement(nullptr, {boxPos, boxPos, boxPos}, siliconLog, "silicon", worldLog, false, 0);
  //
  // --- Create particles that have secondary production threshold.
  G4Gamma::Gamma();
  G4Electron::Electron();
  G4Positron::Positron();
  G4Proton::Proton();
  G4ParticleTable *partTable = G4ParticleTable::GetParticleTable();
  partTable->SetReadiness();
  //
  // --- Create production - cuts object and set the secondary production threshold.
  G4ProductionCuts *productionCuts = new G4ProductionCuts();
  constexpr G4double ProductionCut = 1 * mm;
  productionCuts->SetProductionCut(ProductionCut);
  //
  // --- Register a region for the world.
  G4Region *reg = new G4Region("default");
  reg->AddRootLogicalVolume(worldLog);
  reg->UsedInMassGeometry(true);
  reg->SetProductionCuts(productionCuts);
  //
  // --- Update the couple tables.
  G4ProductionCutsTable *theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  theCoupleTable->UpdateCoupleTable(world);
}

int main(int argc, char *argv[])
{
#ifndef VECGEOM_GDML
  std::cout << "### VecGeom must be compiled with GDML support to run this.\n";
  return 1;
#endif
#ifndef VECGEOM_USE_NAVINDEX
  std::cout << "### VecGeom must be compiled with USE_NAVINDEX support to run this.\n";
  return 2;
#endif

  OPTION_STRING(gdml_name, "trackML.gdml");
  OPTION_INT(cache_depth, 0); // 0 = full depth
  OPTION_INT(particles, 1);
  OPTION_DOUBLE(energy, 100); // entered in GeV
  energy *= copcore::units::GeV;

  InitGeant4();

#ifdef VECGEOM_GDML
  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(cache_depth);
  // The vecgeom millimeter unit is the last parameter of vgdml::Frontend::Load
  bool load = vgdml::Frontend::Load(gdml_name.c_str(), false, copcore::units::mm);
  if (!load) return 3;
#endif

  const vecgeom::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (!world) return 4;

  example9(world, particles, energy);
}
