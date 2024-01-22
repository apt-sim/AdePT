// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0
//
/// \brief Definition of the AdePTConfigurationMessenger class
//
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef ADEPTCONFIGURATIONMESSENGER_HH
#define ADEPTCONFIGURATIONMESSENGER_HH

#include "G4UImessenger.hh"
#include "globals.hh"

class AdePTConfiguration;
class G4UIdirectory;
class G4UIcmdWithAnInteger;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithADouble;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class AdePTConfigurationMessenger : public G4UImessenger {
public:
  AdePTConfigurationMessenger(AdePTConfiguration *);
  ~AdePTConfigurationMessenger();

  virtual void SetNewValue(G4UIcommand *, G4String);

private:
  AdePTConfiguration *fAdePTConfiguration;

  G4UIdirectory *fDir;
  G4UIcmdWithAnInteger *fSetSeedCmd;
  G4UIcmdWithAString *fSetRegionCmd;
  G4UIcmdWithABool *fActivateAdePTCmd;
  G4UIcmdWithAnInteger *fSetVerbosityCmd;
  G4UIcmdWithAnInteger *fSetTransportBufferThresholdCmd;
  G4UIcmdWithADouble *fSetMillionsOfTrackSlotsCmd;
  G4UIcmdWithADouble *fSetMillionsOfHitSlotsCmd;
  G4UIcmdWithADouble *fSetHitBufferFlushThresholdCmd;

};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif // ADEPTCONFIGURATIONMESSENGER_HH
