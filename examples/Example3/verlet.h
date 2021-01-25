// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef VERLET_H
#define VERLET_H

#include "common.h"

class particle;

__host__ __device__ void integrate(particle &p, float dt);
__host__ __device__ void integrate(particle &p, float dt, float s);

#endif
