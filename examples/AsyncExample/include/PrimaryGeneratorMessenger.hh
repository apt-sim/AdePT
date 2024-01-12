// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
//
/// \brief Definition of the PrimaryGeneratorMessenger class
//
// $Id: PrimaryGeneratorMessenger.hh 66241 2012-12-13 18:34:42Z gunter $
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef PRIMARYGENERATORMESSENGER_HH
#define PRIMARYGENERATORMESSENGER_HH

#include "G4UImessenger.hh"
#include "globals.hh"

class PrimaryGeneratorAction;
class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithABool;
class G4UIcmdWithAString;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class PrimaryGeneratorMessenger : public G4UImessenger {
public:
  PrimaryGeneratorMessenger(PrimaryGeneratorAction *);
  ~PrimaryGeneratorMessenger();

  virtual void SetNewValue(G4UIcommand *, G4String);

private:
  PrimaryGeneratorAction *fAction;

  G4UIdirectory *fGunDir;
  G4UIcmdWithoutParameter *fHepmcCmd;
  G4UIcmdWithoutParameter *fDefaultCmd;
  G4UIcmdWithABool *fPrintCmd;
  G4UIcmdWithADouble *fRndmCmd;
  G4UIcmdWithADouble *fRndmDirCmd;
  G4UIcmdWithABool *fRandomizeGunCmd;
  G4UIcmdWithAString *fAddParticleCmd;
  G4UIcmdWithADoubleAndUnit *fMinPhiCmd;
  G4UIcmdWithADoubleAndUnit *fMaxPhiCmd;
  G4UIcmdWithADoubleAndUnit *fMinThetaCmd;
  G4UIcmdWithADoubleAndUnit *fMaxThetaCmd;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif // PRIMARYGENERATORMESSENGER_HH
