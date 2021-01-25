// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "random.h"

#include "common.h"
#include "vector.h"

#include <cmath>
#include <random>

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

using std::cos;
using std::sin;
using std::sqrt;

/* host rng state */

static std::random_device rd;
static std::default_random_engine rng(rd());
static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

/* device rng state */

#ifdef __CUDACC__
static size_t nstates;
static __constant__ curandState *rngs;

void rng_alloc(size_t size)
{
	nstates = size;
	curandState *ptr = nullptr;
	cudaSafe(cudaMalloc(&ptr, size * sizeof(curandState)));
	cudaSafe(cudaMemcpyToSymbol(rngs, &ptr, sizeof(ptr)));
}

void rng_free()
{
	curandState *ptr = nullptr;
	cudaSafe(cudaMemcpyFromSymbol(&ptr, rngs, sizeof(ptr)));
	cudaSafe(cudaFree(ptr));
}

void rng_resize(size_t size)
{
	/* no shrinking */
	if (size <= nstates)
		return;

        curandState *old_ptr = nullptr, *new_ptr = nullptr;
        cudaSafe(cudaMalloc((void**)&new_ptr, size * sizeof(curandState)));
        cudaSafe(cudaMemcpyFromSymbol(&old_ptr, rngs, sizeof(old_ptr)));
        cudaSafe(cudaMemcpyToSymbol(rngs, &new_ptr, sizeof(new_ptr)));
        cudaSafe(cudaFree(old_ptr));
	cudaSafe(cudaDeviceSynchronize());
	nstates = size;
}

__global__ void rng_init(size_t size, unsigned long long seed)
{
	for(unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
                id < size; id += blockDim.x * gridDim.x) {
		curand_init(seed, id, 0, &rngs[id]);
	}
}
#endif

__host__ __device__ float uniform(float a, float b)
{
#ifdef __CUDA_ARCH__
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState local_state = rngs[id];
	float ret = curand_uniform(&local_state);
	rngs[id] = local_state;
	return a + (b - a) * ret;
#else
	return a + (b - a) * dist(rng);
#endif
}

__host__ __device__ vector random_unit_vector()
{
	float z = uniform(-1.0f, 1.0f);
	float r = sqrt(1.0f - z * z);
	float t = uniform(0.0f, 6.2831853f);
	return {r * cos(t), r * sin(t), z};
}
