// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
/// \brief Implementation of the PAr04ParticleGunMessenger class
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "ParticleGunMessenger.hh"
#include "ParticleGun.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "G4ParticleTable.hh"
#include "G4Tokenizer.hh"
#include "G4UnitsTable.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ParticleGunMessenger::ParticleGunMessenger(ParticleGun *Gun) : G4UImessenger(), fGun(Gun), fDir(0), fDefaultCmd(0)
{
  fDir = new G4UIdirectory("/gun/");
  fDir->SetGuidance("gun control");

  fHepmcCmd = new G4UIcmdWithoutParameter("/gun/hepmc", this);
  fHepmcCmd->SetGuidance("select hepmc input");
  fHepmcCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fDefaultCmd = new G4UIcmdWithoutParameter("/gun/setDefault", this);
  fDefaultCmd->SetGuidance("set/reset kinematic defined in PrimaryGenerator");
  fDefaultCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fPrintCmd = new G4UIcmdWithABool("/gun/print", this);
  fPrintCmd->SetGuidance("print gun kinematics in PrimaryGenerator");
  fPrintCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fRandomizeGunCmd = new G4UIcmdWithABool("/gun/randomizeGun", this);
  fRandomizeGunCmd->SetGuidance("Shoot particles in random directions within defined Phi and Theta ranges, the "
                                "particle type is also selected at random from the selected options");
  fRandomizeGunCmd->AvailableForStates(G4State_Idle);

  fAddParticleCmd = new G4UIcmdWithAString("/gun/addParticle", this);
  fAddParticleCmd->SetGuidance("When using randomization, add a particle to the list of possibilities\n\
                                Usage: /gun/addParticle type [\"weight\" weight] [\"energy\" energy unit]\n\
                                type: particle name\n\
                                weight: probability that the particle will appear, between 0 and 1\n\
                                energy: energy and unit for this type of particle\n");
  fAddParticleCmd->SetParameterName("rParticleName", false);
  fAddParticleCmd->SetDefaultValue("geantino");

  fMinPhiCmd = new G4UIcmdWithADoubleAndUnit("/gun/minPhi", this);
  fMinPhiCmd->SetGuidance("Minimum phi angle when using randomization, units deg or rad");
  fMinPhiCmd->SetParameterName("rMinPhi", false);
  fMinPhiCmd->SetRange("rMinPhi>=0.&&rMinPhi<=360.");
  fMinPhiCmd->SetDefaultUnit("deg");
  fMinPhiCmd->AvailableForStates(G4State_Idle);

  fMaxPhiCmd = new G4UIcmdWithADoubleAndUnit("/gun/maxPhi", this);
  fMaxPhiCmd->SetGuidance("Maximum phi angle when using randomization, units deg or rad");
  fMaxPhiCmd->SetParameterName("rMaxPhi", false);
  fMaxPhiCmd->SetRange("rMaxPhi>=0.&&rMaxPhi<=360.");
  fMaxPhiCmd->SetDefaultUnit("deg");
  fMaxPhiCmd->AvailableForStates(G4State_Idle);

  fMinThetaCmd = new G4UIcmdWithADoubleAndUnit("/gun/minTheta", this);
  fMinThetaCmd->SetGuidance("Minimum Theta angle when using randomization, units deg or rad");
  fMinThetaCmd->SetParameterName("rMinTheta", false);
  fMinThetaCmd->SetRange("rMinTheta>=0.&&rMinTheta<=180.");
  fMinThetaCmd->SetDefaultUnit("deg");
  fMinThetaCmd->AvailableForStates(G4State_Idle);

  fMaxThetaCmd = new G4UIcmdWithADoubleAndUnit("/gun/maxTheta", this);
  fMaxThetaCmd->SetGuidance("Maximum Theta angle when using randomization, units deg or rad");
  fMaxThetaCmd->SetParameterName("rMaxTheta", false);
  fMaxThetaCmd->SetRange("rMaxTheta>=0.&&rMaxTheta<=180.");
  fMaxThetaCmd->SetDefaultUnit("deg");
  fMaxThetaCmd->AvailableForStates(G4State_Idle);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ParticleGunMessenger::~ParticleGunMessenger()
{
  delete fHepmcCmd;
  delete fDefaultCmd;
  delete fPrintCmd;
  delete fRandomizeGunCmd;
  delete fAddParticleCmd;
  delete fMinPhiCmd;
  delete fMaxPhiCmd;
  delete fMinThetaCmd;
  delete fMaxThetaCmd;
  delete fDir;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ParticleGunMessenger::SetNewValue(G4UIcommand *command, G4String newValue)
{

  if (command == fHepmcCmd) {
    fGun->SetHepMC();
  }

  if (command == fDefaultCmd) {
    fGun->SetDefaultKinematic();
  }

  if (command == fPrintCmd) {
    fGun->SetPrintGun(fPrintCmd->GetNewBoolValue(newValue));
  }

  if (command == fRandomizeGunCmd) {
    fGun->SetRandomizeGun(fRandomizeGunCmd->GetNewBoolValue(newValue));
  }

  if (command == fAddParticleCmd) {
    G4Tokenizer tkn(newValue);
    G4String str;
    std::vector<G4String> *token_vector = new std::vector<G4String>();
    while ((str = tkn()) != "") {
      token_vector->push_back(str);
    }
    // The particle type is mandatory and must be the first argument
    G4ParticleDefinition *pd;
    float weight  = -1;
    double energy = -1;
    if (token_vector->size() >= 1) {
      pd = G4ParticleTable::GetParticleTable()->FindParticle((*token_vector)[0]);
    } else {
      G4Exception("ParticleGunMessenger::SetNewValue()", "Notification", JustWarning,
                  "No arguments provided. Usage: addParticle type [\"weight\" weight] [\"energy\" energy unit]");
    }

    for (int i = 1; i < token_vector->size(); i++) {
      if ((*token_vector)[i] == "weight") {
        weight = stof((*token_vector)[++i]);
      }
      if ((*token_vector)[i] == "energy") {
        energy = stof((*token_vector)[++i]) * G4UnitDefinition::GetValueOf((*token_vector)[++i]);
      }
    }

    fGun->AddParticle(pd, weight, energy);
  }

  if (command == fMinPhiCmd) {
    fGun->SetMinPhi(fMinPhiCmd->GetNewDoubleValue(newValue));
  }

  if (command == fMaxPhiCmd) {
    fGun->SetMaxPhi(fMaxPhiCmd->GetNewDoubleValue(newValue));
  }

  if (command == fMinThetaCmd) {
    fGun->SetMinTheta(fMinThetaCmd->GetNewDoubleValue(newValue));
  }

  if (command == fMaxThetaCmd) {
    fGun->SetMaxTheta(fMaxThetaCmd->GetNewDoubleValue(newValue));
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......