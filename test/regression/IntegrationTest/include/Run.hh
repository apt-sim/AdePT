// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef RUN_HH
#define RUN_HH

#include "G4Run.hh"

#include "RunAction.hh"

#include <memory>

#define TAG_TYPE int

template <class TTag>
class TestManager;
class TruthHistogrammer;

/**
 * @brief Run-scoped state for the integration benchmark.
 *
 * Besides the existing CSV accumulators, the run optionally owns a
 * TruthHistogrammer when ROOT truth output is enabled. Worker runs merge both
 * pieces of state into the master run at end-of-run.
 */
class Run : public G4Run {

public:
  Run(RunAction *aRunAction);
  ~Run();

  TestManager<TAG_TYPE> *GetTestManager() const { return fTestManager.get(); }
  /// Returns the optional ROOT truth collector for this run.
  TruthHistogrammer *GetTruthHistogrammer() const { return fTruthHistogrammer.get(); }
  void Merge(const G4Run *aRun) override;

  /** @brief Compute and display collected metrics */
  void EndOfRunSummary(G4String aOutputDirectory, G4String aOutputFilenam, double aRunWallTime);

  /**
   * @brief Enum defining the run accumulators
   */
  enum accumulators { NUM_PARTICLES, NUM_ACCUMULATORS };

private:
  std::unique_ptr<TestManager<TAG_TYPE>> fTestManager;
  std::unique_ptr<TruthHistogrammer> fTruthHistogrammer;
  RunAction *fRunAction;
};

#endif
