
// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#ifndef AdePTPhysics_h
#define AdePTPhysics_h 1

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class AdePTTrackingManager;
class AdePTConfiguration;

class AdePTPhysics : public G4VPhysicsConstructor {
public:
  AdePTPhysics(int ver = 1, const G4String &name = "AdePT-physics-list");
  ~AdePTPhysics();
  AdePTTrackingManager *GetTrackingManager() { return fTrackingManager; }

public:
  // This method is dummy for physics: particles are constructed in PhysicsList
  void ConstructParticle() override {};

  // This method will be invoked in the Construct() method.
  // each physics process will be instantiated and
  // registered to the process manager of each particle type
  void ConstructProcess() override;

private:
  AdePTTrackingManager *fTrackingManager;
  AdePTConfiguration *fAdePTConfiguration;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
