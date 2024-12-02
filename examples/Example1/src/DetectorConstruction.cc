// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
//
#include <unordered_map>

#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"
#include "SensitiveDetector.hh"

#include "G4LogicalVolume.hh"
#include "G4UniformMagField.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"

#include "G4SDManager.hh"

#include "G4VPhysicalVolume.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4ProductionCuts.hh"
#include <G4ProductionCutsTable.hh>

#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::DetectorConstruction(bool allSensitive) : fAllSensitive(allSensitive), G4VUserDetectorConstruction()
{
  fDetectorMessenger = new DetectorMessenger(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::~DetectorConstruction() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume *DetectorConstruction::Construct()
{
  // Read the geometry, regions and cuts from the GDML file
  G4cout << "Reading " << fGDML_file << " transiently on CPU for Geant4 ...\n";
  fParser.Read(fGDML_file, false); // turn off schema checker
  G4VPhysicalVolume *world = fParser.GetWorldVolume();

  if (world == nullptr) {
    std::cerr << "Example23: World volume not set properly check your setup selection criteria or GDML input!\n";
    return world;
  }

  fWorld = world;

  return world;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::ConstructSDandField()
{

  
  std::unique_ptr<IField> field;

#ifdef ADEPT_USE_EXT_BFIELD
  // Set a 3D magnetic field from file. If no file is provided, no magnetic field is used in the G4 transport
  if (fFieldFile != "") {
    G4cout << G4endl << " *** SETTING MAGNETIC FIELD TO READ FROM FILE " <<  fFieldFile << " *** " << G4endl << G4endl;
    
    field = std::make_unique<CovfieField>(fFieldFile);
#else
  // Set a 3D magnetic field vector for a uniform field in Bz. If no file is provided, no magnetic field is used in the G4 transport
  if (std::abs(fMagFieldVector[2]) > 0.0) {
    G4cout << G4endl << " *** SETTING CONSTANT MAGNETIC FIELD IN Z: fieldValue in z = " << fMagFieldVector[2] / kilogauss
      << " [kilogauss] *** " << G4endl << G4endl;
      
    field = std::make_unique<UniformField>(fMagFieldVector);
#endif
    MagneticField* magneticField = new MagneticField(std::move(field));

    G4FieldManager *fieldMgr = G4TransportationManager::GetTransportationManager()->GetFieldManager();
    fieldMgr->SetDetectorField(magneticField);
    fieldMgr->CreateChordFinder(magneticField);

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
  int numTouchables          = 0;

  SensitiveDetector *caloSD = new SensitiveDetector("AdePTDetector");

  G4SDManager::GetSDMpointer()->AddNewDetector(caloSD);
  std::vector<G4LogicalVolume *> aSensitiveLogicalVolumes;

  std::function<void(G4VPhysicalVolume const *)> visitAndSetupScoring = [&](G4VPhysicalVolume const *pvol) {
    const auto lvol = pvol->GetLogicalVolume();
    int nd          = lvol->GetNoDaughters();
    numTouchables++;

    if(fAllSensitive) // For easier validation of geometries with no SD info, we may set all volumes as sensitive
    {
      // Record the PV
      if (caloSD->fSensitivePhysicalVolumes.find(pvol) == caloSD->fSensitivePhysicalVolumes.end()) {
        caloSD->fSensitivePhysicalVolumes.insert(pvol);
      }
      // If this is the first time we see this LV
      if (std::find(caloSD->fSensitiveLogicalVolumes.begin(), caloSD->fSensitiveLogicalVolumes.end(), lvol) ==
          caloSD->fSensitiveLogicalVolumes.end()) {
        // Make LogicalVolume sensitive by registering a SensitiveDetector for it
        SetSensitiveDetector(lvol, caloSD);
        // We keep a list of Logical sensitive volumes, used for initializing AdePTTransport
        caloSD->fSensitiveLogicalVolumes.push_back(lvol);
      }
      numSensitiveTouchables++;
    }
    else
    {
      // Check if the LogicalVolume is marked as sensitive in the geometry
      auto aAuxInfoList = fParser.GetVolumeAuxiliaryInformation(lvol);
      for (auto iaux = aAuxInfoList.begin(); iaux != aAuxInfoList.end(); iaux++) {
        G4String str  = iaux->type;
        G4String val  = iaux->value;
        G4String unit = iaux->unit;

        if (str == "SensDet") {
          // If it is, record the PV
          if (caloSD->fSensitivePhysicalVolumes.find(pvol) == caloSD->fSensitivePhysicalVolumes.end()) {
            caloSD->fSensitivePhysicalVolumes.insert(pvol);
          }
          // If this is the first time we see this LV
          if (std::find(caloSD->fSensitiveLogicalVolumes.begin(), caloSD->fSensitiveLogicalVolumes.end(), lvol) ==
              caloSD->fSensitiveLogicalVolumes.end()) {
            // Make LogicalVolume sensitive by registering a SensitiveDetector for it
            SetSensitiveDetector(lvol, caloSD);
            // We keep a list of Logical sensitive volumes, used for initializing AdePTTransport
            caloSD->fSensitiveLogicalVolumes.push_back(lvol);
          }
          numSensitiveTouchables++;
          break;
        }
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
  G4cout << "Num sensitive PVs: " << caloSD->fSensitivePhysicalVolumes.size() << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::Print() const {}
