// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ENERGYLOSSH
#define ENERGYLOSSH

#include "process.h"

class energy_loss : public process {
public:
  __device__ energy_loss() {}
  __device__ virtual double GetPhysicsInteractionLength(int particle_index, adept::BlockData<track> *block) const;
  __device__ virtual void GenerateInteraction(int particle_index, adept::BlockData<track> *block);
};

__device__ double energy_loss::GetPhysicsInteractionLength(int particle_index, adept::BlockData<track> *block) const
{
  using copcore::units::cm;
  track *mytrack = &((*block)[particle_index]);
  // here I will need to calculate the IL based on the particle energy, material, etc
  double current_length = mytrack->uniform() * 0.2 * cm;
  return current_length;
}

__device__ void energy_loss::GenerateInteraction(int particle_index, adept::BlockData<track> *block)
{
  using copcore::units::GeV;
  track *mytrack = &((*block)[particle_index]);

  double eloss = 0.2 * mytrack->energy;
  // energy loss
  mytrack->energy_loss = (eloss < 0.001 * GeV ? mytrack->energy : eloss); // primitive version of scoring
  mytrack->energy      = (eloss < 0.001 * GeV ? 0.0 : mytrack->energy - eloss);

  // if particle has E=0 kill it
  if (mytrack->energy < 0.001 * GeV) mytrack->status = dead;
}

#endif
