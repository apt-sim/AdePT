// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#include "G4RunManager.hh"
#include "G4Run.hh"
#include "Run.hh"

#include "BenchmarkManager.h"
#include <cmath>

#define STDEV(N, MEAN, SUM_SQUARES) N > 1 ? sqrt((SUM_SQUARES - N * MEAN * MEAN) / N) : 0

Run::Run()
{
  fBenchmarkManager = new BenchmarkManager<TAG_TYPE>();
  fAuxBenchmarkManager = new BenchmarkManager<std::string>();
}

Run::~Run() {}

void Run::Merge(const G4Run *run)
{
  const Run *localRun = static_cast<const Run *>(run);

  BenchmarkManager<TAG_TYPE> *aBenchmarkManager = localRun->getBenchmarkManager();

  fBenchmarkManager->addToAccumulator(accumulators::EVENT_SUM, aBenchmarkManager->getAccumulator(accumulators::EVENT_SUM));
  fBenchmarkManager->addToAccumulator(accumulators::EVENT_SQ, aBenchmarkManager->getAccumulator(accumulators::EVENT_SQ));
  fBenchmarkManager->addToAccumulator(accumulators::NONEM_SUM, aBenchmarkManager->getAccumulator(accumulators::NONEM_SUM));
  fBenchmarkManager->addToAccumulator(accumulators::NONEM_SQ, aBenchmarkManager->getAccumulator(accumulators::NONEM_SQ));
  fBenchmarkManager->addToAccumulator(accumulators::ECAL_SUM, aBenchmarkManager->getAccumulator(accumulators::ECAL_SUM));
  fBenchmarkManager->addToAccumulator(accumulators::ECAL_SQ, aBenchmarkManager->getAccumulator(accumulators::ECAL_SQ));

  G4Run::Merge(run);
}

void Run::EndOfRunSummary()
{
  double runTime    = fBenchmarkManager->getDurationSeconds(timers::TOTAL);
  double eventMean  = fBenchmarkManager->getAccumulator(accumulators::EVENT_SUM) / GetNumberOfEvent();
  double eventStdev = STDEV(GetNumberOfEvent(), eventMean, fBenchmarkManager->getAccumulator(accumulators::EVENT_SQ));

  double nonEMMean  = fBenchmarkManager->getAccumulator(accumulators::NONEM_SUM) / GetNumberOfEvent();
  double nonEMStdev = STDEV(GetNumberOfEvent(), nonEMMean, fBenchmarkManager->getAccumulator(accumulators::NONEM_SQ));

  double ecalMean  = fBenchmarkManager->getAccumulator(accumulators::ECAL_SUM) / GetNumberOfEvent();
  double ecalStdev = STDEV(GetNumberOfEvent(), ecalMean, fBenchmarkManager->getAccumulator(accumulators::ECAL_SQ));

  G4cout << "------------------------------------------------------------"
         << "\n";
  G4cout << "BENCHMARK: Run: " << GetRunID() << "\n";
  G4cout << "BENCHMARK: Number of Events: " << GetNumberOfEvent() << "\n";
  G4cout << "BENCHMARK: Run time: " << runTime << "\n";
  G4cout << "BENCHMARK: Mean Event time: " << eventMean << "\n";
  G4cout << "BENCHMARK: Event Standard Deviation: " << eventStdev << "\n";
  G4cout << "BENCHMARK: Mean Non EM time: " << nonEMMean << "\n";
  G4cout << "BENCHMARK: Non EM Standard Deviation: " << nonEMStdev << "\n";
  G4cout << "BENCHMARK: Mean ECAL e-, e+ and gammas time: " << ecalMean << "\n";
  G4cout << "BENCHMARK: ECAL e-, e+ and gammas Standard Deviation: " << ecalStdev << "\n";
  G4cout << "BENCHMARK: Mean proportion of time spent simulating e-, e+ and gammas in ECAL: "
         << 100 * ecalMean / eventMean << "%\n";

  // Export the results
  BenchmarkManager<std::string> *aBenchmarkManager = new BenchmarkManager<std::string>();
  aBenchmarkManager->setAccumulator("Number of Events", GetNumberOfEvent());
  aBenchmarkManager->setAccumulator("Number of Threads", G4RunManager::GetRunManager()->GetNumberOfThreads());
  aBenchmarkManager->setAccumulator("Run time", runTime);
  aBenchmarkManager->setAccumulator("Event mean", eventMean);
  aBenchmarkManager->setAccumulator("Event stdev", eventStdev);
  aBenchmarkManager->setAccumulator("Non EM mean", nonEMMean);
  aBenchmarkManager->setAccumulator("Non EM stdev", nonEMStdev);
  aBenchmarkManager->setAccumulator("ECAL mean", ecalMean);
  aBenchmarkManager->setAccumulator("ECAL stdev", ecalStdev);

  aBenchmarkManager->setOutputFilename("example17");
  aBenchmarkManager->exportCSV();
}