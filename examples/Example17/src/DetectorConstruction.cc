// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
//
#include <unordered_map>

#include "PrimaryGeneratorAction.hh"
#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"
#include "SensitiveDetector.hh"
#include "EMShowerModel.hh"

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

#include <G4GDMLParser.hh>
#include <G4EmParameters.hh>

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/volumes/UnplacedBox.h>
#include <VecGeom/gdml/Frontend.h>

static std::unordered_map<const G4VPhysicalVolume *, int> gScoringMap;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::DetectorConstruction() : G4VUserDetectorConstruction()
{
  fDetectorMessenger = new DetectorMessenger(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::~DetectorConstruction()
{
  delete fShowerModel;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume *DetectorConstruction::Construct()
{
  // Read the geometry, regions and cuts from the GDML file
  G4cout << "Reading " << fGDML_file << " transiently on CPU for Geant4 ...\n";
  G4GDMLParser parser;
  parser.Read(fGDML_file, false); // turn off schema checker
  G4VPhysicalVolume *world = parser.GetWorldVolume();

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

  // For now the number of sensitive volumes matches the number of placed volumes
  int numSensitive = vecgeom::GeoManager::Instance().GetPlacedVolumesCount();

  SensitiveDetector *caloSD = new SensitiveDetector("AdePTDetector", numSensitive);
  caloSD->fScoringMap       = &gScoringMap;
  G4SDManager::GetSDMpointer()->AddNewDetector(caloSD);
  auto detectorRegion = G4RegionStore::GetInstance()->GetRegion(fRegion_name);

  // attaching sensitive detector to the volumes on sentitive_volumes list
  auto const &store = *G4LogicalVolumeStore::GetInstance();
  if (fAllInRegionSensitive) {
    fSensitive_volumes.clear();
    fSensitive_group.clear();
    for (auto lvol : store) {
      if (lvol->GetRegion() == detectorRegion) {
        fSensitive_volumes.push_back(lvol->GetName());
        fSensitive_group.push_back(lvol->GetName());
      }
    }
  }
  int index = 1;
  for (auto name : fSensitive_volumes) {
    G4cout << "Making " << name << " sensitive with index " << index << G4endl;
    caloSD->fSensitive_volume_index[name] = index;
    index++;
    // iterate G4LogicalVolumeStore and set sensitive volumes
    for (auto lvol : store) {
      if (lvol->GetName() == name || lvol->GetName().rfind(name + "0x") == 0) SetSensitiveDetector(lvol, caloSD);
    }
  }

  fShowerModel = new EMShowerModel("AdePT", detectorRegion);

  fShowerModel->SetSensitiveVolumes(&(caloSD->fSensitive_volume_index));
  fShowerModel->SetScoringMap(&gScoringMap);
  fShowerModel->SetVerbosity(fVerbosity);
  fShowerModel->SetBufferThreshold(fBufferThreshold);
  fShowerModel->SetTrackSlots(fTrackSlotsGPU);

  try {
    fShowerModel->Initialize(fActivate_AdePT);
  } catch (const std::runtime_error &ex) {
    std::cerr << ex.what() << "\n";
    exit(EXIT_FAILURE);
    return;
  }
  if (fActivate_AdePT)
    G4cout << "Assigning AdePT transport to region: " << fRegion_name << G4endl;
  else
    G4cout << "Deactivating AdePT, running with Geant4 only!" << G4endl;
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
