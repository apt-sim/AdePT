// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
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
#include "RunAction.hh"
#include "Run.hh"
#include "G4Run.hh"
#include "G4Threading.hh"
#include <AdePT/benchmarking/TestManager.h>

#include <thread>
#include <mutex>

RunAction::RunAction() : G4UserRunAction(), fOutputDirectory(""), fOutputFilename("") {}

RunAction::RunAction(G4String aOutputDirectory, G4String aOutputFilename, bool aDoBenchmark, bool aDoValidation,
                     bool aDoAccumulatedEvents)
    : G4UserRunAction(), fOutputDirectory(aOutputDirectory), fOutputFilename(aOutputFilename),
      fDoBenchmark(aDoBenchmark), fDoValidation(aDoValidation), fDoAccumulatedEvents(aDoAccumulatedEvents)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::~RunAction() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::BeginOfRunAction(const G4Run *)
{
  fTimer.Start();
  auto tid = G4Threading::G4GetThreadId();
  if (tid < 0) {
    if (fDoBenchmark) {
      fRun->GetTestManager()->timerStart(Run::timers::TOTAL);
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::EndOfRunAction(const G4Run *)
{
  static std::mutex print_mutex;
  auto time = fTimer.Stop();
  auto tid  = G4Threading::G4GetThreadId();
  // Just protect the printout to avoid interlacing text
  const std::lock_guard<std::mutex> lock(print_mutex);

  if (GetDoValidation() && GetDoAccumulatedEvents()) {
    // overwrite to have all validation data written into a single line for all events
    fRun->GetTestManager()->exportCSV(false);
  }

  // Print timer just for the master thread since this is called when all workers are done
  if (tid < 0) {
    std::cout << "Run time: " << time << "\n";
    if (fDoBenchmark) {
      fRun->GetTestManager()->timerStop(Run::timers::TOTAL);
    }
    if (fDoBenchmark || fDoValidation) {
      fRun->EndOfRunSummary(fOutputDirectory, fOutputFilename);
    }
  }
}

G4Run *RunAction::GenerateRun()
{
  fRun = new Run(this);
  return fRun;
}
