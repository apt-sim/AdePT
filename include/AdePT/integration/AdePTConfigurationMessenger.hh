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
  G4UIcmdWithAnInteger *fSetCUDAStackLimitCmd;
  G4UIcmdWithAnInteger *fSetCUDAHeapLimitCmd;
  G4UIcmdWithAnInteger *fSetFinishOnCpuCmd;
  G4UIcmdWithABool *fSetTrackInAllRegionsCmd;
  G4UIcmdWithABool *fSetCallUserSteppingActionCmd;
  G4UIcmdWithABool *fSetCallUserTrackingActionCmd;
  G4UIcmdWithABool *fSetSpeedOfLightCmd;
  G4UIcmdWithABool *fSetMultipleStepsInMSCWithTransportationCmd;
  G4UIcmdWithABool *fSetEnergyLossFluctuationCmd;
  G4UIcmdWithAString *fAddRegionCmd;
  G4UIcmdWithABool *fActivateAdePTCmd;
  G4UIcmdWithAnInteger *fSetVerbosityCmd;
  G4UIcmdWithAnInteger *fSetTransportBufferThresholdCmd;
  G4UIcmdWithADouble *fSetMillionsOfTrackSlotsCmd;
  G4UIcmdWithADouble *fSetMillionsOfHitSlotsCmd;
  G4UIcmdWithADouble *fSetHitBufferFlushThresholdCmd;
  G4UIcmdWithADouble *fSetCPUCapacityFactorCmd;

  // Temporary method for setting the VecGeom geometry.
  // In the future the geometry will be converted from Geant4 rather than loaded from GDML.
  G4UIcmdWithAString *fSetGDMLCmd;

  // Set the covfie file for reading in an external B field
  G4UIcmdWithAString *fSetCovfieFileCmd;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif // ADEPTCONFIGURATIONMESSENGER_HH
