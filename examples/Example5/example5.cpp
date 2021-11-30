// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example5.h"

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

#include <G4UnitsTable.hh>
#include <G4SystemOfUnits.hh>

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

int main()
{
  InitGeant4();
  example5();

  return 0;
}
