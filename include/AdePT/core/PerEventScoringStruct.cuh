// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef PER_EVENT_SCORING_STRUCT_CUH
#define PER_EVENT_SCORING_STRUCT_CUH

#include <AdePT/core/ScoringCommons.hh>

namespace AsyncAdePT {

struct PerEventScoring {
  GlobalCounters fGlobalCounters;
  PerEventScoring *const fScoring_dev;

  PerEventScoring(PerEventScoring *gpuScoring) : fScoring_dev{gpuScoring} { ClearGPU(); }
  PerEventScoring(PerEventScoring &&other) = default;
  ~PerEventScoring()                       = default;

  /// @brief Copy hits to host for a single event
  void CopyToHost(cudaStream_t cudaStream = 0);
  // {
  //   const auto oldPointer = fScoring_dev;
  //   COPCORE_CUDA_CHECK(
  //       cudaMemcpyAsync(&fGlobalCounters, fScoring_dev, sizeof(GlobalCounters), cudaMemcpyDeviceToHost, cudaStream));
  //   COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
  //   assert(oldPointer == fScoring_dev);
  //   (void)oldPointer;
  // }

  /// @brief Clear hits on device to reuse for next event
  void ClearGPU(cudaStream_t cudaStream = 0);
  // {
  //   COPCORE_CUDA_CHECK(cudaMemsetAsync(fScoring_dev, 0, sizeof(GlobalCounters), cudaStream));
  //   COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
  // }

  /// @brief Print scoring info
  void Print() { fGlobalCounters.Print(); };
};

}

using AdePTScoring = AsyncAdePT::PerEventScoring;

#endif