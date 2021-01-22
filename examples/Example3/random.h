// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef RANDOM_H
#define RANDOM_H

#include "common.h"

class vector;

#ifdef __CUDACC__
__host__ void rng_alloc(size_t size);
__host__ void rng_resize(size_t size);
__host__ void rng_free();

__global__ void rng_init(size_t size, unsigned long long seed);
#endif

__host__ __device__ float uniform(float a = 0.0f, float b = 1.0f);
__host__ __device__ vector random_unit_vector();

#endif
