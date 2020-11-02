// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_track_executor.cu
 * @brief Unit test for the CUDA executor.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include "run_simulation.hpp"

///______________________________________________________________________________________
int executePipelineGPU()
{
  int result;
  result = runSimulation<copcore::BackendType::CUDA>();
  return result;
}
