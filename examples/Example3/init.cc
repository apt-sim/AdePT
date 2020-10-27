#include "init.h"

#include "particle.h"
#include "random.h"
#include "user.h"

static __host__ __device__ void init_particle(particle &p)
{
	p.time = 0.0f;
	p.mass = uniform(1.0f, 10.0f);
	p.charge = uniform() > 0.3f ? 0.0f : 1.0f;
	p.position = {0.0f, 0.0f, 0.0f};
	p.velocity = {100.0f, 0.0f, 0.0f};
	p.acceleration = {0.0f, 0.0f, 0.0f};
	p.alive = true;
}

#ifdef __CUDACC__
static __global__ void init_gpu(int nparticles)
{
	for(unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
		id < nparticles; id += blockDim.x * gridDim.x) {
		particle p;
		init_particle(p);
		track_init(p);
		set_particle(id, p);
	}
}
#endif

void init(int nparticles)
{
#ifndef __CUDACC__
	particle p;
	while (nparticles-- > 0) {
		init_particle(p);
		create_particle(p);
	}
#else
	/* allocate and initialize rng states */
	rng_alloc(nparticles);

	rng_init<<<nparticles / 36 + 1, 32>>>(nparticles, 1);
	cudaCheckKernelCall();

	/* allocate and initialize particles */
	particle_alloc(nparticles);

	init_gpu<<<nparticles / 36 + 1, 32>>>(nparticles);
	cudaCheckKernelCall();
#endif
}
