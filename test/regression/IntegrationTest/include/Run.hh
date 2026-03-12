// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef RUN_HH
#define RUN_HH

#include "G4Run.hh"

#include "RunAction.hh"

#define TAG_TYPE int

template <class TTag>
class TestManager;

/**
 * @brief Run class for merging and displaying info collected by different worker threads
 */
class Run : public G4Run {

public:
  Run(RunAction *aRunAction);
  ~Run();

  TestManager<TAG_TYPE> *GetTestManager() const { return fTestManager; }

  /** @brief Compute and display collected metrics */
  void EndOfRunSummary(G4String aOutputDirectory, G4String aOutputFilenam, double aRunWallTime);

  /**
   * @brief Enum defining the run accumulators
   */
  enum accumulators { NUM_PARTICLES, NUM_ACCUMULATORS };

private:
  TestManager<TAG_TYPE> *fTestManager;
  RunAction *fRunAction;
};

#endif
