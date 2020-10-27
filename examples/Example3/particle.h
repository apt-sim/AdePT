#ifndef PARTICLE_H
#define PARTICLE_H

#include "point.h"
#include "vector.h"

struct particle {
	point position;
	vector velocity;
	vector acceleration;
	float charge;
	float mass;
	float time;
	bool alive;

	__host__ __device__ float energy() const
	{
		return 0.5 * mass * norm2(velocity);
	}
};

void particle_alloc(int nparticles);
void particle_resize(int nparticles);
void particle_free();

int particles_alive();
int get_number_of_particles();
void cleanup_dead_particles();

__host__ __device__ void create_particle(particle &);

bool get_next_particle(particle &);
void foreach_particle(void (*f)(particle&));

#ifdef __CUDACC__
__device__ particle get_particle(int id);
__device__ void set_particle(int id, particle &p);
__global__ void fetch_new_particles_from_buffer();
void reset_new_particle_count();
#endif

#endif
