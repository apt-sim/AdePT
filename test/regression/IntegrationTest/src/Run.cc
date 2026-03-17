// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include "Run.hh"
#include "TestManager.h"
#include "TruthHistogrammer.hh"

#include "G4Run.hh"
#include "G4RunManager.hh"

Run::Run(RunAction *aRunAction) : fRunAction(aRunAction)
{
  fTestManager = std::make_unique<TestManager<TAG_TYPE>>();
  // Keep the existing CSV accumulator path untouched for the legacy drift and
  // validation consumers.
  fTestManager->setOutputDirectory(fRunAction->GetOutputDirectory());
  fTestManager->setOutputFilename(fRunAction->GetOutputFilename());

  if (fRunAction->GetWriteTruthROOT()) {
    fTruthHistogrammer = std::make_unique<TruthHistogrammer>();
  }
}

Run::~Run() {}

void Run::Merge(const G4Run *aRun)
{
  const auto *otherRun = static_cast<const Run *>(aRun);

  for (const auto &[tag, value] : *otherRun->GetTestManager()->getAccumulators()) {
    fTestManager->addToAccumulator(tag, value);
  }

  for (const auto &[tag, otherInfo] : *otherRun->GetTestManager()->getTimers()) {
    auto &info = (*fTestManager->getTimers())[tag];
    info.accumulatedDuration += otherInfo.accumulatedDuration;
    info.counting = false;
  }

  if (fTruthHistogrammer && otherRun->GetTruthHistogrammer()) {
    // ROOT objects are only created on the master thread; workers merge plain
    // C++ maps into the master's collector here.
    fTruthHistogrammer->MergeFrom(*otherRun->GetTruthHistogrammer());
  }

  G4Run::Merge(aRun);
}

void Run::EndOfRunSummary(G4String aOutputDirectory, G4String aOutputFilename, double aRunWallTime)
{
  TestManager<std::string> aOutputTestManager;
  aOutputTestManager.setAccumulator("Totaltime", aRunWallTime);
  aOutputTestManager.setAccumulator("NumParticles", fTestManager->getAccumulator(accumulators::NUM_PARTICLES));

  aOutputTestManager.setOutputDirectory(aOutputDirectory);
  aOutputTestManager.setOutputFilename(aOutputFilename + "_global");
  aOutputTestManager.exportCSV();

  if (fTruthHistogrammer) {
    // The ROOT truth file is written once on the master after worker merging.
    fTruthHistogrammer->WriteROOTFile(aOutputDirectory + "/" + aOutputFilename + ".root");
  }
}
