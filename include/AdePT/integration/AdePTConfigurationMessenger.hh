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

#include <memory>

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
  ~AdePTConfigurationMessenger() override;

  void SetNewValue(G4UIcommand *, G4String) override;

private:
  AdePTConfiguration *fAdePTConfiguration;

  std::unique_ptr<G4UIdirectory> fDir;
  std::unique_ptr<G4UIcmdWithAnInteger> fSetCUDAStackLimitCmd;
  std::unique_ptr<G4UIcmdWithAnInteger> fSetCUDAHeapLimitCmd;
  std::unique_ptr<G4UIcmdWithAnInteger> fSetAdePTSeedCmd;
  std::unique_ptr<G4UIcmdWithAnInteger> fSetFinishOnCpuCmd;
  std::unique_ptr<G4UIcmdWithAnInteger> fSetMaxWDTIterCmd;
  std::unique_ptr<G4UIcmdWithADouble> fSetWDTKineticEnergyLimitCmd;
  std::unique_ptr<G4UIcmdWithABool> fSetTrackInAllRegionsCmd;
  std::unique_ptr<G4UIcmdWithABool> fSetCallUserSteppingActionCmd;
  std::unique_ptr<G4UIcmdWithABool> fSetCallUserTrackingActionCmd;
  std::unique_ptr<G4UIcmdWithABool> fSetSpeedOfLightCmd;
  std::unique_ptr<G4UIcmdWithABool> fSetMultipleStepsInMSCWithTransportationCmd;
  std::unique_ptr<G4UIcmdWithABool> fSetEnergyLossFluctuationCmd;
  std::unique_ptr<G4UIcmdWithAString> fAddRegionCmd;
  std::unique_ptr<G4UIcmdWithAString> fRemoveRegionCmd;
  std::unique_ptr<G4UIcmdWithAString> fAddWDTRegionCmd;
  std::unique_ptr<G4UIcmdWithAnInteger> fSetVerbosityCmd;
  std::unique_ptr<G4UIcmdWithADouble> fSetMillionsOfTrackSlotsCmd;
  std::unique_ptr<G4UIcmdWithADouble> fSetMillionsOfLeakSlotsCmd;
  std::unique_ptr<G4UIcmdWithADouble> fSetMillionsOfHitSlotsCmd;
  std::unique_ptr<G4UIcmdWithADouble> fSetHitBufferFlushThresholdCmd;
  std::unique_ptr<G4UIcmdWithADouble> fSetCPUCapacityFactorCmd;
  std::unique_ptr<G4UIcmdWithADouble> fSetHitBufferSafetyFactorCmd;

  // Temporary method for setting the VecGeom geometry.
  // In the future the geometry will be converted from Geant4 rather than loaded from GDML.
  std::unique_ptr<G4UIcmdWithAString> fSetGDMLCmd;

  // Set the covfie file for reading in an external B field
  std::unique_ptr<G4UIcmdWithAString> fSetCovfieFileCmd;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif // ADEPTCONFIGURATIONMESSENGER_HH
