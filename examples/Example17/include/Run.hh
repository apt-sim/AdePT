// SPDX-FileCopyrightText: 2023 CERN
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
#ifndef RUN_HH
#define RUN_HH

#include "G4Run.hh"

#define TAG_TYPE int

template <class TTag> 
class BenchmarkManager;
class DetectorConstruction;

class Run : public G4Run {

public:
  Run(DetectorConstruction* aDetector);
  ~Run();

  void Merge(const G4Run* run) override;

  BenchmarkManager<TAG_TYPE>* getBenchmarkManager() const {return fBenchmarkManager;}
  BenchmarkManager<std::string>* getAuxBenchmarkManager() const {return fAuxBenchmarkManager;}

  void EndOfRunSummary();

  enum timers
  {
    //Non electromagnetic timer (Track time outside of the GPU region)
    NONEM,
    //Accumulator within an event (Sum of track times)
    NONEM_EVT,
    //Acummulator for the sum and squared sum of the timings across all events
    NONEM_SUM,
    NONEM_SQ,

    //Acummulator for the sum and squared sum of the timings across all events
    ECAL_SUM,
    ECAL_SQ,

    //Event timer (Timer from start to end)
    EVENT,
    //Acummulator for the sum and squared sum of the timings across all events
    EVENT_SUM,
    EVENT_SQ,

    //Global execution timer
    TOTAL
  };

private:
  BenchmarkManager<TAG_TYPE>* fBenchmarkManager;
  BenchmarkManager<std::string>* fAuxBenchmarkManager;
  DetectorConstruction* fDetector;
};

#endif