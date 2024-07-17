// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DETECTORCONSTRUCTION_H
#define DETECTORCONSTRUCTION_H

#include "G4VUserDetectorConstruction.hh"
#include "G4Material.hh"
#include "G4ThreeVector.hh"

#include <memory>

class G4VPhysicalVolume;
class DetectorMessenger;
class G4GDMLParser;

/**
 * @brief Detector construction.
 *
 * Sensitive detector SensitiveDetector is attached to the
 * Absorber volumes.
 *
 */

class DetectorConstruction : public G4VUserDetectorConstruction {
public:
  DetectorConstruction();
  virtual ~DetectorConstruction();

  virtual G4VPhysicalVolume *Construct() final;
  void CreateVecGeomWorld();

  virtual void ConstructSDandField() final;

  void SetGDMLFile(G4String &file) { fGDML_file = file; }

  // Set uniform magnetic field
  inline void SetMagField(const G4ThreeVector &fv) { fMagFieldVector = fv; }

  // Print detector information
  void Print() const;

private:
  G4String fGDML_file;
  G4VPhysicalVolume *fWorld;

  /// Messenger that allows to modify geometry
  std::unique_ptr<DetectorMessenger> fDetectorMessenger;
  std::unique_ptr<G4GDMLParser> fParser;

  // field related members
  G4ThreeVector fMagFieldVector;
};

#endif /* DETECTORCONSTRUCTION_H */
