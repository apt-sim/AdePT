// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
/// \brief Implementation of the PAr04PrimaryGeneratorMessenger class
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "PrimaryGeneratorMessenger.hh"

#include "PrimaryGeneratorAction.hh"
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

PrimaryGeneratorMessenger::PrimaryGeneratorMessenger(PrimaryGeneratorAction *Gun)
    : G4UImessenger(), fAction(Gun), fGunDir(0), fDefaultCmd(0), fRndmCmd(0)
{
  fGunDir = new G4UIdirectory("/example17/gun/");
  fGunDir->SetGuidance("gun control");

  fHepmcCmd = new G4UIcmdWithoutParameter("/example17/gun/hepmc", this);
  fHepmcCmd->SetGuidance("select hepmc input");
  fHepmcCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fDefaultCmd = new G4UIcmdWithoutParameter("/example17/gun/setDefault", this);
  fDefaultCmd->SetGuidance("set/reset kinematic defined in PrimaryGenerator");
  fDefaultCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fPrintCmd = new G4UIcmdWithABool("/example17/gun/print", this);
  fPrintCmd->SetGuidance("print gun kinematics in PrimaryGenerator");
  fPrintCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fRndmCmd = new G4UIcmdWithADouble("/example17/gun/rndm", this);
  fRndmCmd->SetGuidance("random lateral extension on the beam");
  fRndmCmd->SetParameterName("rBeam", false);
  fRndmCmd->SetRange("rBeam>=0.&&rBeam<=1.");
  fRndmCmd->AvailableForStates(G4State_Idle);

  fRndmDirCmd = new G4UIcmdWithADouble("/example17/gun/rndmDir", this);
  fRndmDirCmd->SetGuidance("random angular extension on the beam");
  fRndmDirCmd->SetParameterName("rBeamDir", false);
  fRndmDirCmd->SetRange("rBeamDir>=0.&&rBeamDir<=1.");
  fRndmDirCmd->AvailableForStates(G4State_Idle);

  fRandomizeGunCmd = new G4UIcmdWithABool("/example17/gun/randomizeGun", this);
  fRandomizeGunCmd->SetGuidance("Shoot particles in random directions within defined Phi and Theta ranges, the "
                                "particle type is also selected at random from the selected options");
  fRandomizeGunCmd->AvailableForStates(G4State_Idle);

  fAddParticleCmd = new G4UIcmdWithAString("/example17/gun/addParticle", this);
  fAddParticleCmd->SetGuidance("When using randomization, add a particle to the list of possibilities\n\
                                Usage: /example17/gun/addParticle type [\"weight\" weight] [\"energy\" energy unit]\n\
                                type: particle name\n\
                                weight: probability that the particle will appear, between 0 and 1\n\
                                energy: energy and unit for this type of particle\n");
  fAddParticleCmd->SetParameterName("rParticleName", false);
  fAddParticleCmd->SetDefaultValue("geantino");

  fMinPhiCmd = new G4UIcmdWithADoubleAndUnit("/example17/gun/minPhi", this);
  fMinPhiCmd->SetGuidance("Minimum phi angle when using randomization, units deg or rad");
  fMinPhiCmd->SetParameterName("rMinPhi", false);
  fMinPhiCmd->SetRange("rMinPhi>=0.&&rMinPhi<=360.");
  fMinPhiCmd->SetDefaultUnit("deg");
  fMinPhiCmd->AvailableForStates(G4State_Idle);

  fMaxPhiCmd = new G4UIcmdWithADoubleAndUnit("/example17/gun/maxPhi", this);
  fMaxPhiCmd->SetGuidance("Maximum phi angle when using randomization, units deg or rad");
  fMaxPhiCmd->SetParameterName("rMaxPhi", false);
  fMaxPhiCmd->SetRange("rMaxPhi>=0.&&rMaxPhi<=360.");
  fMaxPhiCmd->SetDefaultUnit("deg");
  fMaxPhiCmd->AvailableForStates(G4State_Idle);

  fMinThetaCmd = new G4UIcmdWithADoubleAndUnit("/example17/gun/minTheta", this);
  fMinThetaCmd->SetGuidance("Minimum Theta angle when using randomization, units deg or rad");
  fMinThetaCmd->SetParameterName("rMinTheta", false);
  fMinThetaCmd->SetRange("rMinTheta>=0.&&rMinTheta<=180.");
  fMinThetaCmd->SetDefaultUnit("deg");
  fMinThetaCmd->AvailableForStates(G4State_Idle);

  fMaxThetaCmd = new G4UIcmdWithADoubleAndUnit("/example17/gun/maxTheta", this);
  fMaxThetaCmd->SetGuidance("Maximum Theta angle when using randomization, units deg or rad");
  fMaxThetaCmd->SetParameterName("rMaxTheta", false);
  fMaxThetaCmd->SetRange("rMaxTheta>=0.&&rMaxTheta<=180.");
  fMaxThetaCmd->SetDefaultUnit("deg");
  fMaxThetaCmd->AvailableForStates(G4State_Idle);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorMessenger::~PrimaryGeneratorMessenger()
{
  delete fHepmcCmd;
  delete fDefaultCmd;
  delete fPrintCmd;
  delete fRndmCmd;
  delete fRndmDirCmd;
  delete fRandomizeGunCmd;
  delete fAddParticleCmd;
  delete fMinPhiCmd;
  delete fMaxPhiCmd;
  delete fMinThetaCmd;
  delete fMaxThetaCmd;
  delete fGunDir;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PrimaryGeneratorMessenger::SetNewValue(G4UIcommand *command, G4String newValue)
{

  if (command == fHepmcCmd) {
    fAction->SetHepMC();
  }

  if (command == fDefaultCmd) {
    fAction->SetDefaultKinematic();
  }

  if (command == fPrintCmd) {
    fAction->SetPrintGun(fPrintCmd->GetNewBoolValue(newValue));
  }

  if (command == fRndmCmd) {
    fAction->SetRndmBeam(fRndmCmd->GetNewDoubleValue(newValue));
  }

  if (command == fRndmDirCmd) {
    fAction->SetRndmDirection(fRndmDirCmd->GetNewDoubleValue(newValue));
  }

  if (command == fRandomizeGunCmd) {
    fAction->SetRandomizeGun(fRandomizeGunCmd->GetNewBoolValue(newValue));
  }

  if (command == fAddParticleCmd) {
    G4Tokenizer tkn(newValue);
    G4String str;
    std::vector<G4String> *token_vector = new std::vector<G4String>();
    while ((str = tkn()) != "") {
      token_vector->push_back(str);
    }
    //The particle type is mandatory and must be the first argument
    G4ParticleDefinition *pd;
    float weight = -1;
    double energy = -1;
    if (token_vector->size() >= 1) {
      pd = G4ParticleTable::GetParticleTable()->FindParticle((*token_vector)[0]);
    }
    else
    {
      G4Exception("PrimaryGeneratorMessenger::SetNewValue()", "Notification", JustWarning,
                  "No arguments provided. Usage: addParticle type [\"weight\" weight] [\"energy\" energy unit]");
    }

    for(int i=1; i<token_vector->size(); i++)
    {
      if((*token_vector)[i] == "weight")
      {
        weight = stof((*token_vector)[++i]);
      }
      if((*token_vector)[i] == "energy")
      {
        energy = stof((*token_vector)[++i]) * G4UnitDefinition::GetValueOf((*token_vector)[++i]);
      }
    }

    fAction->AddParticle(pd, weight, energy);
  }

  if (command == fMinPhiCmd) {
    fAction->SetMinPhi(fMinPhiCmd->GetNewDoubleValue(newValue));
  }

  if (command == fMaxPhiCmd) {
    fAction->SetMaxPhi(fMaxPhiCmd->GetNewDoubleValue(newValue));
  }

  if (command == fMinThetaCmd) {
    fAction->SetMinTheta(fMinThetaCmd->GetNewDoubleValue(newValue));
  }

  if (command == fMaxThetaCmd) {
    fAction->SetMaxTheta(fMaxThetaCmd->GetNewDoubleValue(newValue));
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......