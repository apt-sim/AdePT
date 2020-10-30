// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_track_executor.cu
 * @brief Unit test for the CUDA executor.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include "executor_functions.h"

///______________________________________________________________________________________
int executePipelineGPU()
{
  int result;
  result = simplePipeline<copcore::BackendType::CUDA>();
  return result;
}
