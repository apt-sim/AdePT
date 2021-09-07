/ SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example11.h"

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
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/management/GeoManager.h>
#ifdef VECGEOM_GDML
#include <VecGeom/gdml/Frontend.h>
#endif

#include <G4String.hh>
#include <G4GDMLParser.hh>

// Necessary state ... 
// static G4GDMLParser        gParser;
static G4VPhysicalVolume*  gWorldPhysical = nullptr;

// Configuration 'parameters'
static G4String fGDMLFileName=     "Default.gdml"; // Change it ... 
bool            cautiousNavigator= true; //  Use it (true) for first runs and tests.  False otherwise.
   
static
G4VPhysicalVolume* ConstructG4geometry() // Replacing G4VUserDetectorConstruction::Construct()
{
  std:cout << "ConstructG4 geometry: reading GDML file : " << fGDMLFileName << "\n";
  G4GDMLParser gdmlParser;   
  gdmlParser.Read(fGDMLFileName, false);
  auto worldPhysical = const_cast<G4VPhysicalVolume *>(gParser.GetWorldVolume());

  if (worldPhysical==nullptr) {
    G4ExceptionDescription ed;
    ed << "World volume not set properly check your setup selection criteria" 
       << "or GDML input!" << G4endl;
    G4Exception( "ConstructG4Geometry() failed", 
                "GeometryCreationError_01", FatalException, ed );
                
  }
  return worldPhysical;
}

static void initGeant4()
{
  //  parser.SetOverlapCheck(true);
  G4TransportationManager *trMgr = 
     G4TransportationManager::GetTransportationManager();
  assert(trMgr);

  gWorldPhysical= ConstructG4geometry();
  
  G4Navigator *nav = trMgr->GetNavigatorForTracking();
  nav->SetWorldVolume(gWorldPhysical);

  // Configure G4 Navigator - to best report errors (optional)
  assert(nav);
  if( cautiousNavigator ) { 
     nav->CheckMode(true);
     std::cout << "Enabled Check mode in G4Navigator";
     nav->SetVerboseLevel(1);
     std::cout << "Enabled Verbose 1  in G4Navigator";
  }
  
  // write back
  // gParser.Write("out.gdml", gWorldPhysical);

  // Manage the magnetic field (future extension)
  // auto fFieldMgr = trMgr->GetFieldManager();

  // Fix the visibility of the world (for Geant4 11.0-beta )
  gWorldPhysical->GetLogicalVolume()->SetVisAttributes(G4VisAttributes::GetInvisible());

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

void InitBVH()
{
  vecgeom::cxx::BVHManager::Init();
  vecgeom::cxx::BVHManager::DeviceInit();
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

  // This will convert the geometry to VecGeom and implicitely also set up
  //   the VecGeom navigator.
  //  Q: does it change the G4 Navigator to use VecGeom Navigation ?????  JA 07.09.2021
  G4VecGeomConverter::Instance().SetVerbose(1);
  G4VecGeomConverter::Instance().ConvertG4Geometry(gWorldPhysical);
  G4cout << vecgeom::GeoManager::Instance().getMaxDepth() << "\n";
#endif

  const vecgeom::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (!world) return 4;

  example11(world, particles, energy);
}
