
// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#ifndef PhysListAdePT_h
#define PhysListAdePT_h 1

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include "AdePTTrackingManager.hh"
#include "DetectorConstruction.hh"

class PhysListAdePT : public G4VPhysicsConstructor {
public:
  PhysListAdePT(DetectorConstruction *aDetector, const G4String &name = "AdePT-physics-list");
  ~PhysListAdePT();
  AdePTTrackingManager* GetTrackingManager(){return fTrackingManager;}

public:
  // This method is dummy for physics: particles are constructed in PhysicsList
  void ConstructParticle() override{};

  // This method will be invoked in the Construct() method.
  // each physics process will be instantiated and
  // registered to the process manager of each particle type
  void ConstructProcess() override;

private:
  AdePTTrackingManager* fTrackingManager; 
  DetectorConstruction* fDetector;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
