// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "particle.h"

#include "user.h"

#include <algorithm>
#include <cstdio>
#include <vector>

static int ntotal;
static int nalive;
static std::vector<particle> particles;

#ifdef __CUDACC__
static __device__ int d_nalive;
static __device__ int d_ntotal;
static __device__ particle *d_particles;

// static int newcur;
static int newtot;
static __device__ int d_newcur;
static __device__ int d_newtot;
static __device__ particle *d_new_particles;
#endif

void particle_alloc(int nparticles)
{
	nalive = 0;
	ntotal = nparticles;
#ifndef __CUDACC__
	particles.reserve(nparticles);
#else
	/* create space for particles */
	particle *ptr = nullptr;
	cudaSafe(cudaMalloc((void **)&ptr, nparticles * sizeof(particle)));
	cudaSafe(cudaMemcpyToSymbol(d_particles, &ptr, sizeof(ptr)));
	cudaSafe(cudaMemcpyToSymbol(d_ntotal, &ntotal, sizeof(int)));

	/* create space for newly created particles */
	newtot = nparticles;
	cudaSafe(cudaMalloc((void **)&ptr, nparticles * sizeof(particle)));
	cudaSafe(cudaMemcpyToSymbol(d_new_particles, &ptr, sizeof(ptr)));
	cudaSafe(cudaMemcpyToSymbol(d_newcur, &nalive, sizeof(int)));
	cudaSafe(cudaMemcpyToSymbol(d_newtot, &newtot, sizeof(int)));
#endif
}

void particle_free()
{
#ifndef __CUDACC__
	particles.clear();
#else
	ntotal = 0;
	nalive = 0;
	particle *ptr = nullptr;

	/* free space for particles */
	cudaSafe(cudaMemcpyFromSymbol(&ptr, d_particles, sizeof(ptr)));
	cudaSafe(cudaFree(ptr));

	/* free space for new particles */
	cudaSafe(cudaMemcpyFromSymbol(&ptr, d_new_particles, sizeof(ptr)));
	cudaSafe(cudaFree(ptr));
#endif
}

void particle_resize(int nparticles)
{
#ifndef __CUDACC__
	particles.reserve(nparticles);
#else
	/* no shrinking */
	if (nparticles <= ntotal)
		return;

	ntotal = nparticles;
	particle *old_ptr = nullptr, *new_ptr = nullptr;
	cudaSafe(cudaMalloc((void **)&new_ptr, nparticles * sizeof(particle)));
	cudaSafe(cudaMemcpyFromSymbol(&old_ptr, d_particles, sizeof(old_ptr)));
	cudaSafe(cudaMemcpy((void **)&new_ptr, old_ptr, ntotal * sizeof(particle),
			    cudaMemcpyDeviceToDevice));
	cudaSafe(cudaMemcpyToSymbol(d_particles, &new_ptr, sizeof(new_ptr)));
	cudaSafe(cudaMemcpyToSymbol(d_ntotal, &ntotal, sizeof(int)));
	cudaSafe(cudaFree(old_ptr));
	cudaSafe(cudaDeviceSynchronize());
#endif
}

#if 0
__device__ void new_particle_resize(int nparticles)
{
	particle *old_ptr = d_new_particles;
	particle *new_ptr = malloc(nparticles * sizeof(particle));
	memcpy(new_ptr, old_ptr, d_newtot * sizeof(particle));
	d_newtot = nparticles;
	d_new_particles = new_ptr;
	free(old_ptr);
}
#endif

#ifdef __CUDACC__
static __global__ void reset_new()
{
	if (threadIdx.x == 0 && blockIdx.x == 0 && d_newcur < 0)
		d_newcur = 0;
}

void reset_new_particle_count()
{
	reset_new<<<1, 1>>>();
	cudaCheckKernelCall();
}

__global__ void fetch_new_particles_from_buffer()
{
	for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < d_ntotal;
	     id += blockDim.x * gridDim.x) {
		/* replace dead particle with new particle */
		if (!d_particles[id].alive) {
			int newid = atomicSub(&d_newcur, 1) - 1;

			/* no more new particles to fetch */
			if (newid < 0)
				break;

			debug_printf("moving particle (%d)\n", newid);
			d_particles[id] = d_new_particles[newid];
			track_init(d_particles[id]);
		}
	}
}
#endif

#ifdef __CUDACC__
static __global__ void update_number_of_particles_alive()
{
	__shared__ int nalive_s;

	if (threadIdx.x == 0 && blockIdx.x == 0)
		d_nalive = 0;

	if (threadIdx.x == 0)
		nalive_s = 0;

	for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < d_ntotal;
	     id += blockDim.x * gridDim.x) {
		if (d_particles[id].alive)
			atomicAdd(&nalive_s, 1);
	}

	__syncthreads();

	if (threadIdx.x == 0)
		atomicAdd(&d_nalive, nalive_s);
}
#endif

int particles_alive()
{
#ifdef __CUDACC__
	update_number_of_particles_alive<<<ntotal / 36 + 1, 32>>>();
	cudaCheckKernelCall();
	cudaSafe(cudaMemcpyFromSymbol(&nalive, d_nalive, sizeof(int)));
	cudaSafe(cudaDeviceSynchronize());
#else
	nalive = particles.size();
#endif
	fprintf(stderr, "\r%5d particles%s", nalive, nalive ? "" : "\n");
	return nalive;
}

int get_number_of_particles()
{
#ifndef __CUDACC__
	return particles.size();
#else
	return ntotal;
#endif
}

void cleanup_dead_particles()
{
#ifndef __CUDACC__
	auto first_dead_particle =
		std::partition(particles.begin(), particles.end(),
			[](const particle &p) { return p.alive; });
	particles.erase(first_dead_particle, particles.end());
#else
	/* we don't really clean up, we just update the numbers */
	fetch_new_particles_from_buffer<<<ntotal / 36 + 1, 32>>>();
	cudaCheckKernelCall();
	reset_new_particle_count();
#endif
}

__host__ __device__ void create_particle(particle &p)
{
#ifdef __CUDA_ARCH__
	/*
	 * kernels that may create particles must ensure enough
	 * space is available before calling this function
	 */
	int id = atomicAdd(&d_newcur, 1);

	if (id < d_newtot) {
		d_new_particles[id] = p;
		debug_printf("created new particle (%d)\n", id);
	} else {
		atomicSub(&d_newcur, 1);
		debug_printf("cannot create particle, out of memory (%d vs %d)\n", id, d_newtot);
	}
#else
	particles.push_back(p);
#endif
}

bool get_next_particle(particle &p)
{
	if (particles.empty())
		return false;

	p = particles.back();
	particles.pop_back();

	return true;
}

void foreach_particle(void (*f)(particle &))
{
	auto n = particles.size();
	for (size_t i = 0; i < n; ++i) {
		particle p = particles[i];
		(*f)(p);
		particles[i] = p;
	}
}

#ifdef __CUDACC__
__device__ particle get_particle(int id)
{
	return d_particles[id];
}

__device__ void set_particle(int id, particle &p)
{
	d_particles[id] = p;
}
#endif
