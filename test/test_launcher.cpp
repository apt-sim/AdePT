// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
/**
 * @file test_track_executor.cu
 * @brief Unit test for the CUDA executor.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <iostream>
#include <cassert>
#include <AdePT/BlockData.h>
#include "run_simulation.hpp"

// Forward declare the GPU version of the class.
int executePipelineGPU();

int executePipelineCPU()
{
  int result = runSimulation<copcore::BackendType::CPU>();
  return result;
}

///______________________________________________________________________________________
int main(void)
{
  executePipelineCPU();
  executePipelineGPU();
}
