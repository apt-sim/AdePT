// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef PHYSICS_H
#define PHYSICS_H

#include "common.h"

class particle;

__host__ __device__ float absorption(float E);
__host__ __device__ float scattering(float E);
__host__ __device__ float production(float E);

__host__ __device__ float sample_interaction_length(particle &);

__host__ __device__ void energy_loss(particle &);
__host__ __device__ void interact(particle &);

#endif
