// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRANSPORT_STATE_TRANSPORT_STATS_HH
#define ADEPT_TRANSPORT_STATE_TRANSPORT_STATS_HH

#include <AdePT/transport/queues/ParticleQueues.cuh>
#include <AdePT/transport/state/EventState.hh>

namespace AsyncAdePT {

// A data structure to transfer statistics after each iteration.
struct Stats {
  int inFlight[GPUQueueIndex::NumSpecies];
  float queueFillLevel[GPUQueueIndex::NumParticleQueues];
  float slotFillLevel[GPUQueueIndex::NumSpecies];
  unsigned int perEventInFlight[kMaxThreads];         // Updated asynchronously
  unsigned int perEventInFlightPrevious[kMaxThreads]; // Used in transport kernels
  unsigned int stepBufferOccupancy;
};

/// Host-only counters accumulating transport-loop stop/stall/flush action reasons across the full run.
/// These are incremented on the host transport thread and printed at shutdown when verbosity >= 1.
struct TransportLoopCounters {
  unsigned long long totalIterations{0};               ///< Total transport iterations executed
  unsigned long long leakExtractionByQueuePressure{0}; ///< Iterations where leak queue exceeded 50% threshold
  unsigned long long leakExtractionByEventFlush{0};    ///< Iterations where an event flush requested leak extraction
  unsigned long long leakExtractionBlocked{0};         ///< Times transport stalled waiting for in-progress extraction
  unsigned long long eventDrainedToStepFlush{0};      ///< Events that transitioned to RequestStepFlush (queues drained)
  unsigned long long stepBufferSwaps{0};              ///< Total step-buffer swaps performed
  unsigned long long stepBufferSwapByOccupancy{0};    ///< Swaps triggered by occupancy >= half capacity
  unsigned long long stepBufferSwapByOccupancy10k{0}; ///< Swaps triggered by occupancy >= 10000
  unsigned long long stepBufferSwapByPressure{0};     ///< Swaps triggered by nextStepMightFail (overflow risk)
  unsigned long long stepBufferSwapByEventFlush{0};   ///< Swaps triggered by event RequestStepFlush
};

/// @brief Array of flags whether the event can be finished off
struct AllowFinishOffEventArray {
  unsigned short flags[kMaxThreads];

  __host__ __device__ unsigned short operator[](int idx) const { return flags[idx]; }
};

} // namespace AsyncAdePT

#endif
