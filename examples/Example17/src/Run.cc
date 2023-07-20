// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include "G4RunManager.hh"
#include "G4Run.hh"
#include "Run.hh"

#include "TestManager.h"
#include "TestManagerStore.h"
#include <cmath>

#define STDEV(N, MEAN, SUM_SQUARES) N > 1 ? sqrt((SUM_SQUARES - N * MEAN * MEAN) / N) : 0

Run::Run()
{
  fTestManager = new TestManager<TAG_TYPE>();
}

Run::~Run() {}

void Run::Merge(const G4Run *run)
{
  if(fDoBenchmark)
  {
    const Run *localRun = static_cast<const Run *>(run);

    TestManager<TAG_TYPE> *aTestManager = localRun->GetTestManager();

    fTestManager->addToAccumulator(accumulators::EVENT_SUM,
                                        aTestManager->getAccumulator(accumulators::EVENT_SUM));
    fTestManager->addToAccumulator(accumulators::EVENT_SQ,
                                        aTestManager->getAccumulator(accumulators::EVENT_SQ));
    fTestManager->addToAccumulator(accumulators::NONEM_SUM,
                                        aTestManager->getAccumulator(accumulators::NONEM_SUM));
    fTestManager->addToAccumulator(accumulators::NONEM_SQ,
                                        aTestManager->getAccumulator(accumulators::NONEM_SQ));
    fTestManager->addToAccumulator(accumulators::ECAL_SUM,
                                        aTestManager->getAccumulator(accumulators::ECAL_SUM));
    fTestManager->addToAccumulator(accumulators::ECAL_SQ, aTestManager->getAccumulator(accumulators::ECAL_SQ));
  }

  G4Run::Merge(run);
}

void Run::EndOfRunSummary(G4String aOutputDirectory, G4String aOutputFilename,
                          DetectorConstruction *aDetector)
{
  if(fDoBenchmark && !fDoValidation)
  {
    // Printout of global statistics
    double runTime    = fTestManager->getDurationSeconds(timers::TOTAL);
    double eventMean  = fTestManager->getAccumulator(accumulators::EVENT_SUM) / GetNumberOfEvent();
    double eventStdev = STDEV(GetNumberOfEvent(), eventMean, fTestManager->getAccumulator(accumulators::EVENT_SQ));

    double nonEMMean  = fTestManager->getAccumulator(accumulators::NONEM_SUM) / GetNumberOfEvent();
    double nonEMStdev = STDEV(GetNumberOfEvent(), nonEMMean, fTestManager->getAccumulator(accumulators::NONEM_SQ));

    double ecalMean  = fTestManager->getAccumulator(accumulators::ECAL_SUM) / GetNumberOfEvent();
    double ecalStdev = STDEV(GetNumberOfEvent(), ecalMean, fTestManager->getAccumulator(accumulators::ECAL_SQ));

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
  }

  // Export the results per event
  std::vector<std::map<int, double>> *aBenchmarkStates = TestManagerStore<int>::GetInstance()->GetStates();
  TestManager<std::string> aOutputTestManager;

  for (int i = 0; i < aBenchmarkStates->size(); i++) {
    if(fDoValidation)
    {
      // If we are taking validation data, export it to the specified file
      auto &groups = aDetector->GetSensitiveGroups();
      int ngroups  = groups.size();
      for (int igroup = 0; igroup < ngroups; ++igroup) {
        aOutputTestManager.setAccumulator(groups[igroup],
                                              (*aBenchmarkStates)[i][igroup + Run::accumulators::NUM_ACCUMULATORS]);
      }
      aOutputTestManager.setOutputDirectory(aOutputDirectory);
      aOutputTestManager.setOutputFilename(aOutputFilename);
      aOutputTestManager.exportCSV();

      aOutputTestManager.reset();
    }
    else if(fDoBenchmark)
    {
      // Recover the results from each event and output them to the specified file
      double eventTime = (*aBenchmarkStates)[i][Run::timers::EVENT];
      double nonEMTime = (*aBenchmarkStates)[i][Run::accumulators::NONEM_EVT];
      double ecalTime  = eventTime - nonEMTime;
      aOutputTestManager.setAccumulator("Event", eventTime);
      aOutputTestManager.setAccumulator("Non EM", nonEMTime);
      aOutputTestManager.setAccumulator("ECAL", ecalTime);

      aOutputTestManager.setOutputDirectory(aOutputDirectory);
      aOutputTestManager.setOutputFilename(aOutputFilename);
      aOutputTestManager.exportCSV();

      aOutputTestManager.reset();
    }
  }
  TestManagerStore<int>::GetInstance()->Reset();
}