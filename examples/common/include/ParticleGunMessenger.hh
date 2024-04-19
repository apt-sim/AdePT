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

#include <memory>

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
  ParticleGun *fGun = nullptr;

  std::unique_ptr<G4UIdirectory> fDir;
  std::unique_ptr<G4UIcmdWithoutParameter> fHepmcCmd;
  std::unique_ptr<G4UIcmdWithoutParameter> fDefaultCmd;
  std::unique_ptr<G4UIcmdWithABool> fPrintCmd;
  std::unique_ptr<G4UIcmdWithABool> fRandomizeGunCmd;
  std::unique_ptr<G4UIcmdWithAString> fAddParticleCmd;
  std::unique_ptr<G4UIcmdWithADoubleAndUnit> fMinPhiCmd;
  std::unique_ptr<G4UIcmdWithADoubleAndUnit> fMaxPhiCmd;
  std::unique_ptr<G4UIcmdWithADoubleAndUnit> fMinThetaCmd;
  std::unique_ptr<G4UIcmdWithADoubleAndUnit> fMaxThetaCmd;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif // PARTICLEGUNMESSENGER_HH
