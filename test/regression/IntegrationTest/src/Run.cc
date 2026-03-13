// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include "Run.hh"
#include "TestManager.h"

#include "G4Run.hh"
#include "G4RunManager.hh"

Run::Run(RunAction *aRunAction) : fRunAction(aRunAction)
{
  fTestManager = new TestManager<TAG_TYPE>();
  // Set output directory and filename, needed for validation where this test manager is used for output
  fTestManager->setOutputDirectory(fRunAction->GetOutputDirectory());
  fTestManager->setOutputFilename(fRunAction->GetOutputFilename());
}

Run::~Run() {}

void Run::EndOfRunSummary(G4String aOutputDirectory, G4String aOutputFilename, double aRunWallTime)
{
  TestManager<std::string> aOutputTestManager;
  aOutputTestManager.setAccumulator("Totaltime", aRunWallTime);
  aOutputTestManager.setAccumulator("NumParticles", fTestManager->getAccumulator(accumulators::NUM_PARTICLES));

  aOutputTestManager.setOutputDirectory(aOutputDirectory);
  aOutputTestManager.setOutputFilename(aOutputFilename + "_global");
  aOutputTestManager.exportCSV();
}
