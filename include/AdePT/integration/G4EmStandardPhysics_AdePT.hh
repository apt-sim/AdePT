// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef G4EmStandardPhysics_AdePT_h
#define G4EmStandardPhysics_AdePT_h 1

#include "G4EmStandardPhysics.hh"

class AdePTTrackingManager;
class AdePTConfiguration;

// EM constructor that preserves the standard Geant4 EM setup and then
// attaches the AdePT tracking manager for e-/e+ and gammas.
class G4EmStandardPhysics_AdePT : public G4EmStandardPhysics {
public:
  explicit G4EmStandardPhysics_AdePT(G4int ver = 1, const G4String &name = "");
  ~G4EmStandardPhysics_AdePT() override;

  void ConstructProcess() override;

private:
  AdePTTrackingManager *fTrackingManager{nullptr};
  AdePTConfiguration *fAdePTConfiguration{nullptr};
};

#endif
