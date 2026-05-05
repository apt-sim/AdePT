// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/G4EmStandardPhysics_HepEm.hh>

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4HepEmTrackingManager.hh"
#include "G4Positron.hh"

G4EmStandardPhysics_HepEm::G4EmStandardPhysics_HepEm(G4int ver, const G4String &name) : G4EmStandardPhysics(ver, name)
{
}

G4EmStandardPhysics_HepEm::~G4EmStandardPhysics_HepEm()
{
  // Keep Geant4-owned cleanup behavior for compatibility.
}

void G4EmStandardPhysics_HepEm::ConstructProcess()
{
  // First register the standard Geant4 EM processes for this constructor.
  G4EmStandardPhysics::ConstructProcess();

  // Register custom G4HepEm tracking manager for e-/e+ and gammas.
  fTrackingManager = new G4HepEmTrackingManager();

  G4Electron::Definition()->SetTrackingManager(fTrackingManager);
  G4Positron::Definition()->SetTrackingManager(fTrackingManager);
  G4Gamma::Definition()->SetTrackingManager(fTrackingManager);
}
