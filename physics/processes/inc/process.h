// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef PROCESS
#define PROCESS

#include "track.h"

#include <AdePT/BlockData.h>

class process {
public:
  __device__ virtual double GetPhysicsInteractionLength(int particle_index, adept::BlockData<track> *block) const = 0;
  __device__ virtual void GenerateInteraction(int particle_index, adept::BlockData<track> *block)                 = 0;
};

#endif
