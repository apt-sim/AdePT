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
#include "ActionInitialisation.hh"
#include "PrimaryGeneratorAction.hh"
#include "EventAction.hh"
#include "RunAction.hh"
#include "TrackingAction.hh"
#include "SteppingAction.hh"
#include "TestManager.h"
#include "DetectorConstruction.hh"

ActionInitialisation::ActionInitialisation(DetectorConstruction *aDetector, G4String aOutputDirectory,
                                           G4String aOutputFilename,
                                            bool aDoBenchmark, bool aDoValidation)
    : G4VUserActionInitialization(), fDetector(aDetector), fOutputDirectory(aOutputDirectory),
      fOutputFilename(aOutputFilename), fDoBenchmark(aDoBenchmark), fDoValidation(aDoValidation)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ActionInitialisation::~ActionInitialisation() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ActionInitialisation::BuildForMaster() const
{
  new PrimaryGeneratorAction(fDetector);
  SetUserAction(new RunAction(fDetector, fOutputDirectory, fOutputFilename, fDoBenchmark, fDoValidation));
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ActionInitialisation::Build() const
{
  SetUserAction(new PrimaryGeneratorAction(fDetector));
  SetUserAction(new EventAction(fDetector));
  RunAction *aRunAction = new RunAction(fDetector, fOutputDirectory, fOutputFilename, fDoBenchmark, fDoValidation);
  SetUserAction(aRunAction);
  TrackingAction *aTrackingAction = new TrackingAction(fDetector);
  SetUserAction(aTrackingAction);

// Do not register this if the TestManager is not active or if benchmark is not selected
#if defined TEST
  if(fDoBenchmark)
  {
    SetUserAction(new SteppingAction(fDetector, aRunAction, aTrackingAction));
  }
#endif
}
