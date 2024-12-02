// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef PER_EVENT_SCORING_STRUCT_CUH
#define PER_EVENT_SCORING_STRUCT_CUH

#include <AdePT/core/ScoringCommons.hh>

namespace AsyncAdePT {

struct PerEventScoring {
  GlobalCounters fGlobalCounters;
  PerEventScoring *const fScoring_dev;

  PerEventScoring(PerEventScoring *gpuScoring) : fScoring_dev{gpuScoring} { 
    ClearGPU(); 
  }
  PerEventScoring(PerEventScoring &&other) = default;
  ~PerEventScoring()                       = default;

  /// @brief Copy hits to host for a single event
  void CopyToHost(cudaStream_t cudaStream = 0);

  /// @brief Clear hits on device to reuse for next event
  void ClearGPU(cudaStream_t cudaStream = 0);

  /// @brief Print scoring info
  void Print() { fGlobalCounters.Print(); };
};

}

using AdePTScoring = AsyncAdePT::PerEventScoring;

#endif