// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
//
#include <unordered_map>

#include "PrimaryGeneratorAction.hh"
#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"
#include "SensitiveDetector.hh"

#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4PVReplica.hh"
#include "G4VisAttributes.hh"
#include "G4RunManager.hh"
#include "G4UniformMagField.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"

#include "G4SDManager.hh"

#include "G4UnitsTable.hh"

#include "G4VPhysicalVolume.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4ProductionCuts.hh"
#include <G4ProductionCutsTable.hh>

#include "G4GeometryManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"

#include <G4EmParameters.hh>
#include "G4Electron.hh"

#include "AdePTTrackingManager.hh"

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/volumes/UnplacedBox.h>
#include <VecGeom/gdml/Frontend.h>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::DetectorConstruction() : G4VUserDetectorConstruction()
{
  fDetectorMessenger = new DetectorMessenger(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::~DetectorConstruction()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume *DetectorConstruction::Construct()
{
  // Read the geometry, regions and cuts from the GDML file
  G4cout << "Reading " << fGDML_file << " transiently on CPU for Geant4 ...\n";
  // G4GDMLParser parser;
  // parser.Read(fGDML_file, false); // turn off schema checker
  // G4VPhysicalVolume *world = parser.GetWorldVolume();
  fParser.Read(fGDML_file, false); // turn off schema checker
  G4VPhysicalVolume *world = fParser.GetWorldVolume();

  if (world == nullptr) {
    std::cerr << "Example17: World volume not set properly check your setup selection criteria or GDML input!\n";
    return world;
  }

  // - REGIONS
  if (world->GetLogicalVolume()->GetRegion() == nullptr) {
    // Add default region if none available
    // constexpr double DefaultCut = 0.7 * mm;
    auto defaultRegion = G4RegionStore::GetInstance()->GetRegion("DefaultRegionForTheWorld");
    // auto pcuts = G4ProductionCutsTable::GetProductionCutsTable()->GetDefaultProductionCuts();
    // pcuts->SetProductionCut(DefaultCut, "gamma");
    // pcuts->SetProductionCut(DefaultCut, "e-");
    // pcuts->SetProductionCut(DefaultCut, "e+");
    // pcuts->SetProductionCut(DefaultCut, "proton");
    // defaultRegion->SetProductionCuts(pcuts);

    defaultRegion->AddRootLogicalVolume(world->GetLogicalVolume());
  }

  for (auto region : *G4RegionStore::GetInstance()) {
    region->UsedInMassGeometry(true); // make sure all regions are marked as used
    region->UpdateMaterialList();
  }

  // - UPDATE COUPLES
  G4cout << "Updating material-cut couples based on " << G4RegionStore::GetInstance()->size() << " regions ...\n";
  G4ProductionCutsTable *theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  theCoupleTable->UpdateCoupleTable(world);

  // --- Set MSC range factor to match G4HepEm physics lists.
  G4EmParameters *param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetMscRangeFactor(0.04);

  CreateVecGeomWorld();

  fWorld = world;

  return world;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::ConstructSDandField()
{
  if (fMagFieldVector.mag() > 0.0) {
    // Apply a global uniform magnetic field along the Z axis.
    // Notice that only if the magnetic field is not zero, the Geant4
    // transportion in field gets activated.
    auto uniformMagField     = new G4UniformMagField(fMagFieldVector);
    G4FieldManager *fieldMgr = G4TransportationManager::GetTransportationManager()->GetFieldManager();
    fieldMgr->SetDetectorField(uniformMagField);
    fieldMgr->CreateChordFinder(uniformMagField);
    G4cout << G4endl << " *** SETTING MAGNETIC FIELD : fieldValue = " << fMagFieldVector / kilogauss
           << " [kilogauss] *** " << G4endl << G4endl;

  } else {
    G4cout << G4endl << " *** NO MAGNETIC FIELD SET  *** " << G4endl << G4endl;
  }

  /*
  Set up the sensitive volumes and scoring map

  We will have one hit per Placement of a sensitive LogicalVolume
  SensitiveDetector will store the mapping of PhysicalVolumes to hits, and use it in ProcessHits
  
  fSensitive_volumes Contains the names of the LogicalVolumes we want to make sensitive, at this point 
  it has been already filled through macro commands

  In order to find all placements of these sensitive volumes, we need to walk the tree
  */
  int numSensitiveTouchables = 0;
  int numTouchables = 0;

  SensitiveDetector *caloSD = new SensitiveDetector("AdePTDetector", &fSensitivePhysicalVolumes);
  G4SDManager::GetSDMpointer()->AddNewDetector(caloSD);
  std::vector<G4LogicalVolume*> aSensitiveLogicalVolumes;

  std::function<void(G4VPhysicalVolume const *)> visitAndSetupScoring = [&](G4VPhysicalVolume const *pvol) {
    const auto lvol = pvol->GetLogicalVolume();
    int nd           = lvol->GetNoDaughters();
    numTouchables++;

    // Check if the LogicalVolume is sensitive
    auto aAuxInfoList = fParser.GetVolumeAuxiliaryInformation(lvol);
    for(auto iaux = aAuxInfoList.begin(); iaux != aAuxInfoList.end(); iaux++ )
    {
      G4String str=iaux->type;
      G4String val=iaux->value;
      G4String unit=iaux->unit;

      if(str == "SensDet")
      {
        // If it is, record the PV
        if( fSensitivePhysicalVolumes.find(pvol) == fSensitivePhysicalVolumes.end() )
        {
          fSensitivePhysicalVolumes.insert(pvol); 
        }
        // If this is the first time we see this LV
        if(std::find(caloSD->fSensitiveLogicalVolumes.begin(), 
            caloSD->fSensitiveLogicalVolumes.end(), 
            lvol) == caloSD->fSensitiveLogicalVolumes.end())
        //if( caloSD->fSensitiveLogicalVolumes.find(lvol) == caloSD->fSensitiveLogicalVolumes.end() )
        {
          //G4cout << "Making " << lvol->GetName() << " sensitive" << G4endl;
          // Make LogicalVolume sensitive by registering a SensitiveDetector for it
          SetSensitiveDetector(lvol, caloSD);
          // We keep a list of Logical sensitive volumes, used for initializing AdePTIntegration
          caloSD->fSensitiveLogicalVolumes.push_back(lvol);
        }
        numSensitiveTouchables++;
        break;
      }
    }

    // Visit the daughters
    for (int id = 0; id < nd; ++id) {
      auto daughter = lvol->GetDaughter(id);
      visitAndSetupScoring(daughter);
    }
  };

  visitAndSetupScoring(fWorld);

  // Print info about the geometry

  G4cout << "Num sensitive touchables: " << numSensitiveTouchables << G4endl;
  G4cout << "Num touchables: " << numTouchables << G4endl;
  G4cout << "Num VecGeom placements: " << vecgeom::GeoManager::Instance().GetPlacedVolumesCount() << G4endl;
  G4cout << "Num sensitive PVs: " << fSensitivePhysicalVolumes.size() << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::Print() const {}

void DetectorConstruction::CreateVecGeomWorld()
{

  // Import the gdml file into VecGeom
  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(0);
  vgdml::Parser vgdmlParser;
  auto middleWare = vgdmlParser.Load(fGDML_file.c_str(), false, copcore::units::mm);
  if (middleWare == nullptr) {
    std::cerr << "Failed to read geometry from GDML file '" << fGDML_file << "'" << G4endl;
    return;
  }

  const vecgeom::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (world == nullptr) {
    std::cerr << "GeoManager world volume is nullptr" << G4endl;
    return;
  }
}
