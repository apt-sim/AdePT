// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef PAIRPRODUCTIONH
#define PAIRPRODUCTIONH

#include "process.h"

class pair_production : public process {
public:
  __device__ pair_production() {}
  __device__ virtual float GetPhysicsInteractionLength(int particle_index, adept::BlockData<track> *block) const;
  __device__ virtual void GenerateInteraction(int particle_index, adept::BlockData<track> *block);
};

__device__ float pair_production::GetPhysicsInteractionLength(int particle_index, adept::BlockData<track> *block) const
{
  track *mytrack = &((*block)[particle_index]);
  // here I will need to calculate the IL based on the particle energy, material, etc
  float current_length = mytrack->uniform() * 0.25f;
  return current_length;
}

__device__ void pair_production::GenerateInteraction(int particle_index, adept::BlockData<track> *block)
{
  track *mytrack = &((*block)[particle_index]);

  float eloss = 0.5f * mytrack->energy;

  // pair production
  mytrack->energy -= eloss;
  mytrack->energy_loss           = 0;
  mytrack->number_of_secondaries = 1;

  auto secondary_track = block->NextElement();
  assert(secondary_track != nullptr && "No slot available for secondary track");
  secondary_track->energy                = eloss;
  secondary_track->status                = alive;
  secondary_track->energy_loss           = 0;
  secondary_track->number_of_secondaries = 0;
  // Inherit current position and direction.
  secondary_track->pos           = mytrack->pos;
  secondary_track->dir           = mytrack->dir;
  secondary_track->current_state = mytrack->current_state;
  // Initialize a new PRNG state.
  secondary_track->rng_state = mytrack->rng_state;
  secondary_track->rng_state.Skip(1 << 15);
}

#endif
