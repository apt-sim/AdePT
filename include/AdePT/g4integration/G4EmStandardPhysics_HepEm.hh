// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef G4EmStandardPhysics_HepEm_h
#define G4EmStandardPhysics_HepEm_h 1

#include "G4EmStandardPhysics.hh"

class G4HepEmTrackingManager;

// EM constructor that preserves the standard Geant4 EM setup and then
// attaches the G4HepEm tracking manager for e-/e+ and gammas.
class G4EmStandardPhysics_HepEm : public G4EmStandardPhysics {
public:
  explicit G4EmStandardPhysics_HepEm(G4int ver = 1, const G4String &name = "");
  ~G4EmStandardPhysics_HepEm() override;

  void ConstructProcess() override;

private:
  G4HepEmTrackingManager *fTrackingManager{nullptr};
};

#endif
