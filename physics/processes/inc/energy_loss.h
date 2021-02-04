// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ENERGYLOSSH
#define ENERGYLOSSH

#include "process.h"

class energy_loss : public process {
public:
  __device__ energy_loss() {}
  __device__ virtual float GetPhysicsInteractionLength(int particle_index, adept::BlockData<track> *block) const;
  __device__ virtual void GenerateInteraction(int particle_index, adept::BlockData<track> *block);
};

__device__ float energy_loss::GetPhysicsInteractionLength(int particle_index, adept::BlockData<track> *block) const
{
  track *mytrack = &((*block)[particle_index]);
  // here I will need to calculate the IL based on the particle energy, material, etc
  float current_length = mytrack->uniform() * 0.2f;
  return current_length;
}

__device__ void energy_loss::GenerateInteraction(int particle_index, adept::BlockData<track> *block)
{

  track *mytrack = &((*block)[particle_index]);

  float eloss = 0.2f * mytrack->energy;
  // energy loss
  mytrack->energy_loss           = (eloss < 0.001f ? mytrack->energy : eloss); // primitive version of scoring
  mytrack->energy                = (eloss < 0.001f ? 0.0f : mytrack->energy - eloss);

  // if particle has E=0 kill it
  if (mytrack->energy < 0.001f) mytrack->status = dead;
}

#endif
