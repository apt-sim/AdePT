#include "transport.h"

#include "common.h"
#include "geometry.h"
#include "particle.h"
#include "physics.h"
#include "point.h"
#include "user.h"
#include "verlet.h"

#include <cmath>
#include <cfloat>
#include <iostream>

using std::abs;
using std::min;
using std::max;

#define min_step 1.00e-04f
#define dt 1.00e-04f
#define tmax 1.00e+01f

__host__ __device__ void step(particle &p)
{
	step_init(p);
	/* mean free path is infinite where there's no material */
	float s = density(p.position) > 0.0 ? sample_interaction_length(p) : FLT_MAX;
	float d = distance_to_boundary(p.position);

	float sd = d; /* keep sign of distance to check for boundary crossings */

	/* integrate trajectory respecting geometric boundaries */
	if (abs(d) < s) {
		while (abs(d) < s && d * sd > 0.0f &&
		       (inside_world(p.position) && p.time < tmax)) {
			float ds = max(abs(d), min_step);
			integrate(p, dt, ds);
			energy_loss(p);
			d = distance_to_boundary(p.position);
			s -= ds;
		}
	}

	if (s < abs(d)) {
		/* physics limited step */
		integrate(p, dt, s);
		debug_printf("# physics limited step\n");
		energy_loss(p);
		interact(p);
	} else {
		/* geometry limited step, do a small push to the other side of the boundary */
		integrate(p, 0.00001, 0.00001);
		debug_printf("# geometry limited step\n");
	}

	/* particle moved outside world */
	if (!inside_world(p.position)) {
		debug_printf("# outside world\n");
		p.alive = false;
	}

	if (p.time >= tmax) {
		debug_printf("# max time reached\n");
		p.alive = false;
	}

	step_exit(p);

	if (!p.alive)
		track_exit(p);
}

#ifdef __CUDACC__
static __global__ void stepping_kernel(unsigned int nparticles)
{
	for(unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
			id < nparticles; id += blockDim.x * gridDim.x) {
		particle p = get_particle(id);

		if (!p.alive)
			continue;

		step(p);

		set_particle(id, p);
	}
}
#endif

void step_all_particles()
{
#ifndef __CUDACC__
	foreach_particle(step);
#else
	int ntotal = get_number_of_particles();
	stepping_kernel<<<ntotal / 36 + 1, 32>>>(ntotal);
	cudaCheckKernelCall();
#endif
	cleanup_dead_particles();
}

void transport(particle &p)
{
	track_init(p);

	while (p.alive)
		step(p);

	track_exit(p);
}
