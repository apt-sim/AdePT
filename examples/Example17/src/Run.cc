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
#include "G4RunManager.hh"
#include "G4Run.hh"
#include "Run.hh"

#include "BenchmarkManager.h"
#include <cmath>

#include "DetectorConstruction.hh"

#define STDEV(N, MEAN, SUM_SQUARES) N>1 ? sqrt((SUM_SQUARES-N*pow(MEAN,2))/N) : 0

Run::Run(DetectorConstruction* aDetector) : fDetector(aDetector)
{
  fBenchmarkManager = new BenchmarkManager<TAG_TYPE>();
  fAuxBenchmarkManager = new BenchmarkManager<std::string>();
}

Run::~Run()
{

}

void Run::Merge(const G4Run* run)
{
  const Run* localRun = static_cast<const Run*>(run);

  BenchmarkManager<TAG_TYPE>* aBenchmarkManager = localRun->getBenchmarkManager();

  fBenchmarkManager->addDurationSeconds(timers::EVENT_SUM, aBenchmarkManager->getDurationSeconds(timers::EVENT_SUM));
  fBenchmarkManager->addDurationSeconds(timers::EVENT_SQ, aBenchmarkManager->getDurationSeconds(timers::EVENT_SQ));
  fBenchmarkManager->addDurationSeconds(timers::NONEM_SUM, aBenchmarkManager->getDurationSeconds(timers::NONEM_SUM));
  fBenchmarkManager->addDurationSeconds(timers::NONEM_SQ, aBenchmarkManager->getDurationSeconds(timers::NONEM_SQ));
  fBenchmarkManager->addDurationSeconds(timers::ECAL_SUM, aBenchmarkManager->getDurationSeconds(timers::ECAL_SUM));
  fBenchmarkManager->addDurationSeconds(timers::ECAL_SQ, aBenchmarkManager->getDurationSeconds(timers::ECAL_SQ));

  /*
  BenchmarkManager<std::string>* aAuxBenchmarkManager = localRun->getAuxBenchmarkManager();
  auto &groups        = fDetector->GetSensitiveGroups();
  int ngroups         = groups.size();
  for(int evt=0; evt<localRun->GetNumberOfEvent(); evt++)
  {
    for (int igroup = 0; igroup < ngroups; ++igroup) 
    {
      std::cout << aAuxBenchmarkManager->getDurationSeconds(groups[igroup]+std::to_string(evt)) << std::endl;
      fAuxBenchmarkManager->addDurationSeconds(groups[igroup]+"_SUM", aAuxBenchmarkManager->getDurationSeconds(groups[igroup]+std::to_string(evt)));
      fAuxBenchmarkManager->addDurationSeconds(groups[igroup]+"_SQ", pow(aAuxBenchmarkManager->getDurationSeconds(groups[igroup]+std::to_string(evt)), 2));
    } 
  }
  */

  G4Run::Merge(run);
}

void Run::EndOfRunSummary()
{
  double runTime = fBenchmarkManager->getDurationSeconds(timers::TOTAL);
  double eventMean = fBenchmarkManager->getDurationSeconds(timers::EVENT_SUM)/GetNumberOfEvent();
  double eventStdev = STDEV(GetNumberOfEvent(), eventMean, fBenchmarkManager->getDurationSeconds(timers::EVENT_SQ));

  double nonEMMean = fBenchmarkManager->getDurationSeconds(timers::NONEM_SUM)/GetNumberOfEvent();
  double nonEMStdev = STDEV(GetNumberOfEvent(), nonEMMean, fBenchmarkManager->getDurationSeconds(timers::NONEM_SQ));

  double ecalMean = fBenchmarkManager->getDurationSeconds(timers::ECAL_SUM)/GetNumberOfEvent();
  double ecalStdev = STDEV(GetNumberOfEvent(), ecalMean, fBenchmarkManager->getDurationSeconds(timers::ECAL_SQ));

  G4cout << "------------------------------------------------------------" << "\n";
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
        << 100*ecalMean/eventMean << "%\n"; 
  G4cout << "------------------------------------------------------------" << "\n";

  //Export the results
  BenchmarkManager<std::string>* aBenchmarkManager = new BenchmarkManager<std::string>();
  aBenchmarkManager->addDurationSeconds("Number of Events", GetNumberOfEvent());
  aBenchmarkManager->addDurationSeconds("Number of Threads", G4RunManager::GetRunManager()->GetNumberOfThreads());
  aBenchmarkManager->addDurationSeconds("Run time", runTime);
  aBenchmarkManager->addDurationSeconds("Event mean", eventMean);
  aBenchmarkManager->addDurationSeconds("Event stdev", eventStdev);
  aBenchmarkManager->addDurationSeconds("Non EM mean", nonEMMean);
  aBenchmarkManager->addDurationSeconds("Non EM stdev", nonEMStdev);
  aBenchmarkManager->addDurationSeconds("ECAL mean", ecalMean);
  aBenchmarkManager->addDurationSeconds("ECAL stdev", ecalStdev);

  aBenchmarkManager->exportCSV("example17");

  /*
  auto &groups        = fDetector->GetSensitiveGroups();
  int ngroups         = groups.size();
  for (int igroup = 0; igroup < ngroups; ++igroup) 
  {
    fAuxBenchmarkManager->addDurationSeconds(groups[igroup]+"_MEAN", fAuxBenchmarkManager->getDurationSeconds(groups[igroup]+"_SUM")/GetNumberOfEvent());
    fAuxBenchmarkManager->addDurationSeconds(groups[igroup]+"_STDEV", STDEV(GetNumberOfEvent(), fAuxBenchmarkManager->getDurationSeconds(groups[igroup]+"_MEAN"), fAuxBenchmarkManager->getDurationSeconds(groups[igroup]+"_SQ")));
    fAuxBenchmarkManager->removeTimer(groups[igroup]+"_SUM");
    fAuxBenchmarkManager->removeTimer(groups[igroup]+"_SQ");
  }
  fAuxBenchmarkManager->exportCSV("example17_energy");
  */
}