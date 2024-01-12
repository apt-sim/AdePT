// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
//
/// \brief Definition of the ParticleGunMessenger class
//
// $Id: ParticleGunMessenger.hh 66241 2012-12-13 18:34:42Z gunter $
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef PARTICLEGUNMESSENGER_HH
#define PARTICLEGUNMESSENGER_HH

#include "G4UImessenger.hh"
#include "globals.hh"

class ParticleGun;
class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithABool;
class G4UIcmdWithAString;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class ParticleGunMessenger : public G4UImessenger {
public:
  ParticleGunMessenger(ParticleGun *);
  ~ParticleGunMessenger();

  virtual void SetNewValue(G4UIcommand *, G4String);

private:
  ParticleGun *fGun;

  G4UIdirectory *fDir;
  G4UIcmdWithoutParameter *fHepmcCmd;
  G4UIcmdWithoutParameter *fDefaultCmd;
  G4UIcmdWithABool *fPrintCmd;
  G4UIcmdWithABool *fRandomizeGunCmd;
  G4UIcmdWithAString *fAddParticleCmd;
  G4UIcmdWithADoubleAndUnit *fMinPhiCmd;
  G4UIcmdWithADoubleAndUnit *fMaxPhiCmd;
  G4UIcmdWithADoubleAndUnit *fMinThetaCmd;
  G4UIcmdWithADoubleAndUnit *fMaxThetaCmd;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif // PARTICLEGUNMESSENGER_HH
