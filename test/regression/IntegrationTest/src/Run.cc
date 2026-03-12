// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include "G4RunManager.hh"
#include "G4Run.hh"
#include "Run.hh"

#include <AdePT/benchmarking/TestManager.h>

Run::Run(RunAction *aRunAction) : fRunAction(aRunAction)
{
  fTestManager = new TestManager<TAG_TYPE>();
  // Set output directory and filename, needed for validation where this test manager is used for output
  fTestManager->setOutputDirectory(fRunAction->GetOutputDirectory());
  fTestManager->setOutputFilename(fRunAction->GetOutputFilename());
}

Run::~Run() {}

void Run::Merge(const G4Run *run)
{
  const Run *localRun                = static_cast<const Run *>(run);
  TestManager<TAG_TYPE> *testManager = localRun->GetTestManager();

  // Merge all worker accumulators so the master can emit a single, consistent
  // accumulated-events CSV row.
  for (const auto &entry : *testManager->getAccumulators()) {
    fTestManager->addToAccumulator(entry.first, entry.second);
  }

  G4Run::Merge(run);
}

void Run::EndOfRunSummary(G4String aOutputDirectory, G4String aOutputFilename, double aRunWallTime)
{
  TestManager<std::string> aOutputTestManager;
  aOutputTestManager.setAccumulator("Totaltime", aRunWallTime);
  aOutputTestManager.setAccumulator("NumParticles", fTestManager->getAccumulator(accumulators::NUM_PARTICLES));

  aOutputTestManager.setOutputDirectory(aOutputDirectory);
  aOutputTestManager.setOutputFilename(aOutputFilename + "_global");
  aOutputTestManager.exportCSV();
}
