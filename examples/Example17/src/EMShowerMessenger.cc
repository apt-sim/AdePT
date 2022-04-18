// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
#include "EMShowerModel.hh"
#include "EMShowerMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithAnInteger.hh"

EMShowerMessenger::EMShowerMessenger(EMShowerModel *aModel) : fModel(aModel)
{
  fDirectory = new G4UIdirectory("/example17/fastSim/");
  fDirectory->SetGuidance("Set mesh parameters for the example fast sim model.");

  fPrintCmd = new G4UIcmdWithoutParameter("/example17/fastSim/print", this);
  fPrintCmd->SetGuidance("Print current settings.");

  fNbOfHitsCmd = new G4UIcmdWithAnInteger("/example17/fastSim/numberOfHits", this);
  fNbOfHitsCmd->SetGuidance("Set number of (same energy) energy deposits created in fast simulation. "
                            "Those deposits will be scored in the detector according to the readout of "
                            "the sensitive detector.");
  fNbOfHitsCmd->SetParameterName("Number", false);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EMShowerMessenger::~EMShowerMessenger()
{
  delete fPrintCmd;
  delete fNbOfHitsCmd;
  delete fDirectory;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EMShowerMessenger::SetNewValue(G4UIcommand *aCommand, G4String aNewValues)
{
  if (aCommand == fPrintCmd) {
    fModel->Print();
  }
}
