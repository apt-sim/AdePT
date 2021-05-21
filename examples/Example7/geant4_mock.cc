// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "geant4_mock.h"

#include "G4Types.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4PVReplica.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"

#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Proton.hh"
#include "G4ParticleTable.hh"
#include "G4ProductionCuts.hh"
#include "G4ProductionCutsTable.hh"

// This basically copies the TestEm3 example's way of constructing the test geometry
// Yes, global, but fine for a dumb example where we just need a couple of constants common
// in two programs...
const char *WorldMaterial    = "G4_Galactic";
const char *GapMaterial      = "G4_Pb";
const char *AbsorberMaterial = "G4_lAr";

constexpr double ProductionCut = 0.7 * mm;

constexpr double fCalorSizeYZ       = 40 * cm;
constexpr int fNbOfLayers           = 50;
constexpr int fNbOfAbsorbers        = 2;
constexpr double GapThickness      = 2.3 * mm;
constexpr double AbsorberThickness = 5.7 * mm;
constexpr double fAbsorThickness[fNbOfAbsorbers+1] = {0.0, GapThickness, AbsorberThickness};

constexpr double fLayerThickness = GapThickness + AbsorberThickness;
constexpr double fCalorThickness = fNbOfLayers * fLayerThickness;

constexpr double fWorldSizeX  = 1.2 * fCalorThickness;
constexpr double fWorldSizeYZ = 1.2 * fCalorSizeYZ;

G4PVPlacement* geant4_mock() {
  // - GEOMETRY
  // materials
  G4Material *fDefaultMaterial = G4NistManager::Instance()->FindOrBuildMaterial(WorldMaterial);
  G4Material *gap      = G4NistManager::Instance()->FindOrBuildMaterial(GapMaterial);
  G4Material *absorber = G4NistManager::Instance()->FindOrBuildMaterial(AbsorberMaterial);

  G4Material* fAbsorMaterial[fNbOfAbsorbers+1] = {fDefaultMaterial, gap, absorber};

  //
  // World
  //
  auto* fSolidWorld = new G4Box("World",                                                 // its name
                                fWorldSizeX / 2., fWorldSizeYZ / 2., fWorldSizeYZ / 2.); // its size

  auto* fLogicWorld = new G4LogicalVolume(fSolidWorld,      // its solid
                                          fDefaultMaterial, // its material
                                          "World");         // its name

  auto* fPhysiWorld = new G4PVPlacement(0,               // no rotation
                                        G4ThreeVector(), // at (0,0,0)
                                        fLogicWorld,     // its fLogical volume
                                        "World",         // its name
                                        0,               // its mother  volume
                                        false,           // no boolean operation
                                        0);              // copy number

  //
  // Calorimeter
  //
  auto* fSolidCalor = new G4Box("Calorimeter", fCalorThickness / 2., fCalorSizeYZ / 2., fCalorSizeYZ / 2.);

  auto* fLogicCalor = new G4LogicalVolume(fSolidCalor, fDefaultMaterial, "Calorimeter");

  auto* fPhysiCalor = new G4PVPlacement(0,               // no rotation
                                        G4ThreeVector(), // at (0,0,0)
                                        fLogicCalor,     // its fLogical volume
                                        "Calorimeter",   // its name
                                        fLogicWorld,     // its mother  volume
                                        false,           // no boolean operation
                                        0);              // copy number

  //
  // Layers
  //
  auto* fSolidLayer = new G4Box("Layer", fLayerThickness / 2, fCalorSizeYZ / 2, fCalorSizeYZ / 2);

  auto* fLogicLayer = new G4LogicalVolume(fSolidLayer, fDefaultMaterial, "Layer");

  // Layers
  // VecGeom doesn't support replica volumes, so use direct placement for now (different from original TestEm3)
  G4VPhysicalVolume* fPhysiLayer = nullptr;
  G4double xstart = -0.5 * fCalorThickness;
  for (G4int k = 1; k <= fNbOfLayers; ++k) {
    G4double xcenter = xstart + (k - 1 + 0.5) * fLayerThickness;
    fPhysiLayer = new G4PVPlacement(0,
                                    G4ThreeVector(xcenter, 0., 0.),
                                    fLogicLayer,
                                    "Layer",
                                    fLogicCalor,
                                    false,
                                    k);
  }

  //
  // Absorbers
  //
  G4double xfront = -0.5 * fLayerThickness;
  for (G4int k = 1; k <= fNbOfAbsorbers; k++) {
    auto* fSolidAbsor = new G4Box("Absorber", // its name
                                  fAbsorThickness[k] / 2, fCalorSizeYZ / 2, fCalorSizeYZ / 2);

    auto* fLogicAbsor = new G4LogicalVolume(fSolidAbsor,    // its solid
                                            fAbsorMaterial[k], // its material
                                            fAbsorMaterial[k]->GetName());

    G4double xcenter = xfront + 0.5 * fAbsorThickness[k];
    xfront += fAbsorThickness[k];
    auto* fPhysiAbsor = new G4PVPlacement(0, G4ThreeVector(xcenter, 0., 0.), fLogicAbsor, fAbsorMaterial[k]->GetName(),
                                         fLogicLayer, false,
                                         k); // copy number
  }

  // - PHYSICS
  // --- Create particles that have secondary production threshold.
  G4Gamma::Gamma();
  G4Electron::Electron();
  G4Positron::Positron();
  G4Proton::Proton();
  G4ParticleTable *partTable = G4ParticleTable::GetParticleTable();
  partTable->SetReadiness();
  //
  // --- Create production - cuts object and set the secondary production threshold.
  auto *productionCuts = new G4ProductionCuts();
  productionCuts->SetProductionCut(ProductionCut);
  //
  // --- Register a region for the world.
  auto *reg = new G4Region("default");
  reg->AddRootLogicalVolume(fLogicWorld);
  reg->UsedInMassGeometry(true);
  reg->SetProductionCuts(productionCuts);
  //
  // --- Update the couple tables.
  G4ProductionCutsTable *theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  theCoupleTable->UpdateCoupleTable(fPhysiWorld);

  // Return geometry, ...
  return fPhysiWorld;
}