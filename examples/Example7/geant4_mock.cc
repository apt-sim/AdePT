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

  // world...
  auto* worldBox = new G4Box("World", worldHalfLength, worldHalfLength, worldHalfLength);
  auto* worldLogical = new G4LogicalVolume(worldBox, g4Galactic, "World");
  auto* worldPhysical = new G4PVPlacement(0, G4ThreeVector(), worldLogical, "World", 0, false, 0);

  // -- Create particles that have secondary production threshold.
  G4Gamma::Gamma();
  G4Electron::Electron();
  G4Positron::Positron();
  G4Proton::Proton();
  G4ParticleTable *partTable = G4ParticleTable::GetParticleTable();
  partTable->SetReadiness();

  // --- Create production - cuts object and set the secondary production threshold.
  G4ProductionCuts *productionCuts = new G4ProductionCuts();
  constexpr G4double ProductionCut = 1 * mm;
  productionCuts->SetProductionCut(ProductionCut);

  // --- Register a region for the world.
  G4Region *reg = new G4Region("default");
  reg->AddRootLogicalVolume(worldLogical);
  reg->UsedInMassGeometry(true);
  reg->SetProductionCuts(productionCuts);

  // --- Update the couple tables.
  G4ProductionCutsTable *theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  theCoupleTable->UpdateCoupleTable(worldPhysical);

  return worldPhysical;
}