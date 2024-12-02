// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DETECTORCONSTRUCTION_H
#define DETECTORCONSTRUCTION_H

#include "MagneticFields.hh"

#include "G4VUserDetectorConstruction.hh"
#include "G4Material.hh"
#include "G4ThreeVector.hh"

#include <G4GDMLParser.hh>

class G4Box;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Material;

class G4UniformMagField;
class G4FieldManager;

class DetectorMessenger;

/**
 * @brief Detector construction.
 *
 * Sensitive detector SensitiveDetector is attached to the
 * Absorber volumes.
 *
 */

class DetectorConstruction : public G4VUserDetectorConstruction {
public:
  DetectorConstruction(bool allSensitive = false);
  virtual ~DetectorConstruction();

  virtual G4VPhysicalVolume *Construct() final;
  void CreateVecGeomWorld();

  virtual void ConstructSDandField() final;

  void SetGDMLFile(G4String &file) { fGDML_file = file; }

  // Set uniform magnetic field
  inline void SetMagField(const G4ThreeVector &fv) { fMagFieldVector = fv; }
  void SetFieldFile(G4String &file) { fFieldFile = file; }

  // Print detector information
  void Print() const;

private:
  G4String fGDML_file;
  G4VPhysicalVolume *fWorld;

  /// Messenger that allows to modify geometry
  DetectorMessenger *fDetectorMessenger;

  // field related members
  G4ThreeVector fMagFieldVector;
  G4String fFieldFile;
  std::unique_ptr<MagneticField> fMagneticField;

  G4GDMLParser fParser;
  bool fAllSensitive;
};

#endif /* DETECTORCONSTRUCTION_H */
