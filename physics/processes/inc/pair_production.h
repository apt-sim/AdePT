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

  float esecond = 0.5f * mytrack->energy;

  // pair production
  mytrack->energy -= esecond;
  mytrack->energy_loss           = 0;
  mytrack->number_of_secondaries++;

  auto secondary_track = block->NextElement();
  if (secondary_track == nullptr) {
    COPCORE_EXCEPTION("No slot available for secondary track");

  secondary_track->energy                = esecond;  
  secondary_track->status                = alive;
  secondary_track->energy_loss           = 0;
  
  // Book-keeping parts of state
  secondary_track->index                 = 100 * mytrack->index + mytrack->number_of_secondaries;  // For tracing / debugging
  secondary_track->mother_index          = mytrack->index;
  secondary_track->number_of_secondaries = 0;     
  secondary_track->eventId    = mytrack->eventId;
  secondary_track->num_step   = 0;

  // Inherit current position and direction.
  secondary_track->pos           = mytrack->pos;
  secondary_track->dir           = mytrack->dir;
  secondary_track->current_state = mytrack->current_state;
  secondary_track->next_state =    mytrack->current_state;

  // Inherit current position and direction.
  secondary_track->pos           = mytrack->pos;
  secondary_track->dir           = mytrack->dir;
  secondary_track->current_state = mytrack->current_state;
  // Initialize a new PRNG state.
  secondary_track->rng_state = mytrack->rng_state;
  secondary_track->rng_state.Skip(1 << 15);

  // secondary_track->index              = ++slowIndex;  // ???  Relevant for debugging etc only
}

#endif
