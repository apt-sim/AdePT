
// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
#ifndef PhysListHepEm_h
#define PhysListHepEm_h 1

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class PhysListHepEm : public G4VPhysicsConstructor {
public:
  PhysListHepEm(const G4String &name = "G4HepEm-physics-list");
  ~PhysListHepEm();

public:
  // This method is dummy for physics: particles are constructed in PhysicsList
  void ConstructParticle() override{};

  // This method will be invoked in the Construct() method.
  // each physics process will be instantiated and
  // registered to the process manager of each particle type
  void ConstructProcess() override;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
