// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef RUN_HH
#define RUN_HH

#include "G4Run.hh"
#include "DetectorConstruction.hh"

#define TAG_TYPE int

template <class TTag>
class TestManager;

/**
 * @brief Run class for merging and displaying info collected by different worker threads
 */
class Run : public G4Run {

public:
  Run();
  ~Run();

  /** @brief Merge the results of the worker threads */
  void Merge(const G4Run *run) override;

  TestManager<TAG_TYPE> *GetTestManager() const { return fTestManager; }
  void SetDoBenchmark( bool aDoBenchmark ){ fDoBenchmark = aDoBenchmark; }
  bool GetDoBenchmark(){ return fDoBenchmark; }
  void SetDoValidation( bool aDoValidation ){ fDoValidation = aDoValidation; }
  bool GetDoValidation(){ return fDoValidation; }

  /** @brief Compute and display collected metrics */
  void EndOfRunSummary(G4String aOutputDirectory, G4String aOutputFilename, DetectorConstruction *aDetector);

  /**
   * @brief Enum defining the timers that we can use for benchmarking
   */
  enum timers {
    // Non electromagnetic timer (Track time outside of the GPU region)
    NONEM,
    // Event timer (Timer from start to end)
    EVENT,
    // Global execution timer
    TOTAL,
    NUM_TIMERS
  };

  /**
   * @brief Enum defining the accumulators that we can use for benchmarking
   */
  enum accumulators {
    // Accumulator within an event (Sum of track times)
    NONEM_EVT = timers::NUM_TIMERS,
    // Acummulators for the sum and squared sum of the timings across all events
    NONEM_SUM,
    NONEM_SQ,
    ECAL_SUM,
    ECAL_SQ,
    EVENT_SUM,
    EVENT_SQ,
    NUM_ACCUMULATORS
  };

private:
  TestManager<TAG_TYPE> *fTestManager;
  bool fDoBenchmark;
  bool fDoValidation;
};

#endif