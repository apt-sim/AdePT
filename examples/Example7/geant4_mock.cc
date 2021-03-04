// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "geant4_mock.h"

#include "G4Types.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"

#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Proton.hh"
#include "G4ParticleTable.hh"
#include "G4ProductionCuts.hh"
#include "G4ProductionCutsTable.hh"

G4PVPlacement* geant4_mock() {
  // This largely follows G4HepEM's testing methods
  // Want nested regions with common materials to make sure we can connect these up

  // -- Materials (Matched to TestEm3)
  G4Material* g4Galactic = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
  G4Material* g4Pb = G4NistManager::Instance()->FindOrBuildMaterial("G4_Pb");
  G4Material* g4LAr = G4NistManager::Instance()->FindOrBuildMaterial("G4_lAr");

  // -- Geometry
  const G4double worldHalfLength = 5.0*CLHEP::m;
  const G4double outerAbsHalfLength = 0.8*worldHalfLength;
  const G4double scintHalfLength = 0.8*outerAbsHalfLength;
  const G4double innerAbsHalfLength = 0.8*scintHalfLength;

  // world...
  auto* worldBox = new G4Box("World", worldHalfLength, worldHalfLength, worldHalfLength);
  auto* worldLogical = new G4LogicalVolume(worldBox, g4Galactic, "World");
  auto* worldPhysical = new G4PVPlacement(0, G4ThreeVector(), worldLogical, "World", 0, false, 0);

  // Lead "Outer Absorber"
  auto* outerAbsBox =new G4Box("OuterAbsorber", outerAbsHalfLength, outerAbsHalfLength, outerAbsHalfLength);
  auto* outerAbsLogical = new G4LogicalVolume(outerAbsBox, g4Pb, "OuterAbsorber");
  auto* outerAbsPhysical = new G4PVPlacement(0, G4ThreeVector(), outerAbsLogical, "OuterAbsorber", worldLogical, false, 0);

  // Liquid argon "scintillator"
  auto* scintBox =new G4Box("Scintillator", scintHalfLength, scintHalfLength, scintHalfLength);
  auto* scintLogical = new G4LogicalVolume(scintBox, g4LAr, "Scintillator");
  auto* scintPhysical = new G4PVPlacement(0, G4ThreeVector(), scintLogical, "Scintillator", outerAbsLogical, false, 0);

   // Lead "Inner Absorber"
  auto* innerAbsBox = new G4Box("InnerAbsorber", innerAbsHalfLength, innerAbsHalfLength, innerAbsHalfLength);
  auto* innerAbsLogical = new G4LogicalVolume(innerAbsBox, g4Pb, "InnerAbsorber");
  auto* innerAbsPhysical = new G4PVPlacement(0, G4ThreeVector(), innerAbsLogical, "InnerAbsorber", scintLogical, false, 0);

  // -- Create particles that have secondary production threshold.
  G4Gamma::Gamma();
  G4Electron::Electron();
  G4Positron::Positron();
  G4Proton::Proton();
  G4ParticleTable* partTable = G4ParticleTable::GetParticleTable();
  partTable->SetReadiness();

  // --- Create production - cuts object and set the secondary production threshold.
  G4ProductionCuts* productionCuts = new G4ProductionCuts();
  constexpr G4double ProductionCut = 1 * mm;
  productionCuts->SetProductionCut(ProductionCut);

  // --- Register a default region
  G4Region* reg = new G4Region("default");
  reg->AddRootLogicalVolume(worldLogical);
  reg->UsedInMassGeometry(true);
  reg->SetProductionCuts(productionCuts);

  // --- and one for the Inner Absorber with different production cuts
  G4Region* absReg = new G4Region("inner_absorber");
  absReg->AddRootLogicalVolume(innerAbsLogical);
  absReg->UsedInMassGeometry(true);

  auto* absProductionCuts = new G4ProductionCuts();
  absProductionCuts->SetProductionCut(0.7*CLHEP::mm);

  absReg->SetProductionCuts(absProductionCuts);

  // --- Update the couple tables.
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  theCoupleTable->UpdateCoupleTable(worldPhysical);

  return worldPhysical;
}