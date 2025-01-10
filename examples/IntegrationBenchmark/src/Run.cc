// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include "G4RunManager.hh"
#include "G4Run.hh"
#include "Run.hh"

#include <AdePT/benchmarking/TestManager.h>
#include <AdePT/benchmarking/TestManagerStore.h>
#include <cmath>

#define STDEV(N, MEAN, SUM_SQUARES) N > 1 ? sqrt((SUM_SQUARES - N * MEAN * MEAN) / N) : 0

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
  if (fRunAction->GetDoBenchmark()) {
    const Run *localRun = static_cast<const Run *>(run);

    TestManager<TAG_TYPE> *aTestManager = localRun->GetTestManager();

    fTestManager->addToAccumulator(accumulators::EVENT_SUM, aTestManager->getAccumulator(accumulators::EVENT_SUM));
    fTestManager->addToAccumulator(accumulators::EVENT_SQ, aTestManager->getAccumulator(accumulators::EVENT_SQ));
    fTestManager->addToAccumulator(accumulators::NONEM_SUM, aTestManager->getAccumulator(accumulators::NONEM_SUM));
    fTestManager->addToAccumulator(accumulators::NONEM_SQ, aTestManager->getAccumulator(accumulators::NONEM_SQ));
    fTestManager->addToAccumulator(accumulators::ECAL_SUM, aTestManager->getAccumulator(accumulators::ECAL_SUM));
    fTestManager->addToAccumulator(accumulators::ECAL_SQ, aTestManager->getAccumulator(accumulators::ECAL_SQ));
    fTestManager->addToAccumulator(accumulators::NUM_PARTICLES,
                                   aTestManager->getAccumulator(accumulators::NUM_PARTICLES));

    // TEMP: DELETE THIS
    fTestManager->addToAccumulator(accumulators::EVENT_HIT_COPY_SIZE,
                                   aTestManager->getAccumulator(accumulators::EVENT_HIT_COPY_SIZE));
    fTestManager->addToAccumulator(accumulators::EVENT_HIT_COPY_SIZE_SQ,
                                   aTestManager->getAccumulator(accumulators::EVENT_HIT_COPY_SIZE_SQ));
  }

  G4Run::Merge(run);
}

void Run::EndOfRunSummary(G4String aOutputDirectory, G4String aOutputFilename)
{
  if (fRunAction->GetDoBenchmark() && !fRunAction->GetDoValidation()) {
    // Printout of global statistics
    double runTime    = fTestManager->getDurationSeconds(timers::TOTAL);
    double eventMean  = fTestManager->getAccumulator(accumulators::EVENT_SUM) / GetNumberOfEvent();
    double eventStdev = STDEV(GetNumberOfEvent(), eventMean, fTestManager->getAccumulator(accumulators::EVENT_SQ));

    // currently unused:
    // double nonEMMean  = fTestManager->getAccumulator(accumulators::NONEM_SUM) / GetNumberOfEvent();
    // double nonEMStdev = STDEV(GetNumberOfEvent(), nonEMMean, fTestManager->getAccumulator(accumulators::NONEM_SQ));
    // double ecalMean  = fTestManager->getAccumulator(accumulators::ECAL_SUM) / GetNumberOfEvent();
    // double ecalStdev = STDEV(GetNumberOfEvent(), ecalMean, fTestManager->getAccumulator(accumulators::ECAL_SQ));

    G4cout << "------------------------------------------------------------"
           << "\n";
    G4cout << "BENCHMARK: Run: " << GetRunID() << "\n";
    G4cout << "BENCHMARK: Number of Events: " << GetNumberOfEvent() << "\n";
    G4cout << "BENCHMARK: Run time: " << runTime << "\n";
    G4cout << "BENCHMARK: Mean Event time: " << eventMean << "\n";
    G4cout << "BENCHMARK: Event Standard Deviation: " << eventStdev << "\n";
  }

  // Export the results per event
  std::vector<std::map<int, double>> *aBenchmarkStates = TestManagerStore<int>::GetInstance()->GetStates();
  TestManager<std::string> aOutputTestManager;

  // aBenchmarkStates->size() should correspond to the number of events
  for (size_t i = 0; i < aBenchmarkStates->size(); i++) {
    if (fRunAction->GetDoValidation()) {
      // If we are taking validation data, export it to the specified file
      // Each benchmark state contains one counter per LogicalVolume
      // Export one CSV containing a list of volume IDs and Edep per event
      // for (auto iter = (*aBenchmarkStates)[i].begin(); iter != (*aBenchmarkStates)[i].end(); ++iter) {
      //   if(iter->first >= Run::accumulators::NUM_ACCUMULATORS)
      //     aOutputTestManager.setAccumulator(std::to_string(iter->first - Run::accumulators::NUM_ACCUMULATORS),
      //     iter->second);
      // }

      // aOutputTestManager.setOutputDirectory(aOutputDirectory);
      // aOutputTestManager.setOutputFilename(aOutputFilename);
      // aOutputTestManager.exportCSV(false);

      // aOutputTestManager.reset();
    } else if (fRunAction->GetDoBenchmark()) {
      // Recover the results from each event and output them to the specified file
      double eventTime = (*aBenchmarkStates)[i][Run::timers::EVENT];
      double nonEMTime = (*aBenchmarkStates)[i][Run::accumulators::NONEM_EVT];
      double ecalTime  = eventTime - nonEMTime;
      aOutputTestManager.setAccumulator("Event", eventTime);
      aOutputTestManager.setAccumulator("Non EM", nonEMTime);
      aOutputTestManager.setAccumulator("ECAL", ecalTime);

      aOutputTestManager.setOutputDirectory(fRunAction->GetOutputDirectory());
      aOutputTestManager.setOutputFilename(fRunAction->GetOutputFilename());
      aOutputTestManager.exportCSV();

      aOutputTestManager.reset();
    }
  }
  TestManagerStore<int>::GetInstance()->Reset();

  // Export global results

  aOutputTestManager.setAccumulator("Totaltime", fTestManager->getDurationSeconds(timers::TOTAL));
  aOutputTestManager.setAccumulator("NumParticles", fTestManager->getAccumulator(accumulators::NUM_PARTICLES));

  aOutputTestManager.setOutputDirectory(aOutputDirectory);
  aOutputTestManager.setOutputFilename(aOutputFilename + "_global");
  aOutputTestManager.exportCSV();
}
