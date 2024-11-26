
// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
#ifndef HepEMPhysics_h
#define HepEMPhysics_h 1

// #include "G4VPhysicsConstructor.hh"
#include "G4EmStandardPhysics.hh"
#include "globals.hh"

// class HepEMPhysics : public G4VPhysicsConstructor {
class HepEMPhysics : public G4EmStandardPhysics {
public:
  HepEMPhysics(int ver, const G4String &name = "G4HepEm-physics-list");
  ~HepEMPhysics();

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
