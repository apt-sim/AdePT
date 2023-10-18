
// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#ifndef PhysListAdePT_h
#define PhysListAdePT_h 1

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include "AdePTTrackingManager.hh"

class PhysListAdePT : public G4VPhysicsConstructor {
public:
  PhysListAdePT(AdePTTrackingManager* trmgr, const G4String &name = "AdePT-physics-list");
  ~PhysListAdePT();

public:
  // This method is dummy for physics: particles are constructed in PhysicsList
  void ConstructParticle() override{};

  // This method will be invoked in the Construct() method.
  // each physics process will be instantiated and
  // registered to the process manager of each particle type
  void ConstructProcess() override;

private:
  AdePTTrackingManager* trackingManager; 
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
