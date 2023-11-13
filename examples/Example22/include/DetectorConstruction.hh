// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
//
#ifndef DETECTORCONSTRUCTION_H
#define DETECTORCONSTRUCTION_H

#include "G4VUserDetectorConstruction.hh"
#include "G4SystemOfUnits.hh"
#include "G4Material.hh"
#include "G4ThreeVector.hh"
#include "AdeptIntegration.h"
#include "AdePTTrackingManager.hh"

#include <G4GDMLParser.hh>

class G4Box;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Material;

class G4UniformMagField;
class G4FieldManager;
class PrimaryGeneratorAction;
class EMShowerModel;

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
  DetectorConstruction();
  virtual ~DetectorConstruction();

  virtual G4VPhysicalVolume *Construct() final;
  void CreateVecGeomWorld();

  virtual void ConstructSDandField() final;

  void SetGDMLFile(G4String &file) { fGDML_file = file; }
  void SetRegionName(G4String &reg) { fRegion_name = reg; }
  void AddSensitiveVolume(G4String volume)
  {
    if (volume == "*")
      fAllInRegionSensitive = true;
    else
      fSensitive_volumes.push_back(volume);
  }
  void AddSensitiveGroup(G4String group) { fSensitive_group.push_back(group); }

  // Set uniform magnetic field
  inline void SetMagField(const G4ThreeVector &fv) { fMagFieldVector = fv; }
  void SetPrimaryGenerator(PrimaryGeneratorAction *pg) { fPrimaryGenerator = pg; }

  // Print detector information
  void Print() const;

  // Activate AdePT
  void SetActivateAdePT(bool act) { fActivate_AdePT = act; }

  // Set verbosity
  void SetVerbosity(int verbosity) { fVerbosity = verbosity; }

  // Set AdePT buffer threshold
  void SetBufferThreshold(int value) { fBufferThreshold = value; }

  int GetBufferThreshold() { return fBufferThreshold; }

  // Set total number of track slots on GPU
  void SetTrackSlots(double value) { fTrackSlotsGPU = value; }

  // Set total number of hit slots
  void SetHitSlots(double value) { fHitSlots = value; }

  double GetTrackSlots() { return fTrackSlotsGPU; }

  double GetHitSlots() { return fHitSlots; }

  std::vector<G4String> &GetSensitiveGroups() { return fSensitive_group; }

  G4String &GetRegionName() { return fRegion_name; }

private:
  int fVerbosity{0};        ///< Actually verbosity for AdePT integration
  int fBufferThreshold{20}; ///< Buffer threshold for AdePT transport
  double fTrackSlotsGPU{1}; ///< Total number of track slots allocated on GPU (millions)
  double fHitSlots{1};      ///< Total number of hit slots allocated on GPU and Host (millions)
  G4String fGDML_file;
  G4String fRegion_name;
  std::vector<G4String> fSensitive_volumes;
  std::vector<G4String> fSensitive_group;
  G4VPhysicalVolume *fWorld;

  bool fAllInRegionSensitive{false};
  bool fActivate_AdePT{true};

  /// Messenger that allows to modify geometry
  DetectorMessenger *fDetectorMessenger;

  // field related members
  G4ThreeVector fMagFieldVector;
  PrimaryGeneratorAction *fPrimaryGenerator;

  // Physical Volumes where we will score
  std::set<const G4VPhysicalVolume *> fSensitivePhysicalVolumes;

  G4GDMLParser fParser;
};

#endif /* DETECTORCONSTRUCTION_H */
