// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef PROCESS
#define PROCESS

#include "track.h"

class process  {
    public:
        __device__ virtual float GetPhysicsInteractionLength(int particle_index, adept::BlockData<track> *block, curandState_t *states) const = 0;
        __device__ virtual void GenerateInteraction(int particle_index, adept::BlockData<track> *block, curandState_t *states) = 0;
};

#endif
