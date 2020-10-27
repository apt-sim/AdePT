#include "physics.h"

#include "particle.h"
#include "point.h"
#include "random.h"
#include "vector.h"

#include <cmath>
#include <iostream>

using std::exp;
using std::log;
using std::sqrt;

#define Ecut 10.0f

__host__ __device__ float absorption(float E)
{
	return (5000.0f / E) * exp(-0.001 * E / 2.0);
}

__host__ __device__ float scattering(float E)
{
	return E <= 100.0f ? 0.0f : 10.0f * (0.001 * E - 0.1f) * exp(-sqrt(0.001 * E));
}

__host__ __device__ float production(float E)
{
	return E < 100.0f ? 0.0f : 0.01 * std::log(E - 9.0f);
}

__host__ __device__ float cross_section(float E)
{
	return absorption(E) + scattering(E) + production(E);
}

__host__ __device__ float sample_interaction_length(particle &p)
{
	return -log(uniform()) / cross_section(p.energy());
}

__host__ __device__ void energy_loss(particle &p)
{
	if (std::abs(p.charge) > 0.0f)
		p.velocity *= 0.99;
}

__host__ __device__ void absorption_interaction(particle &p)
{
	p.alive = false;
	debug_printf("# absorbed\n");
}

__host__ __device__ void scattering_interaction(particle &p)
{
	float x = uniform(-1.0f, 1.0f);
	float y = uniform(-1.0f, 1.0f);
	float z = uniform(-1.0f, 1.0f);
	vector dv{x, y, z};
	dv = cross(p.velocity, dv / norm(p.velocity));
	vector v = (p.velocity + dv) / norm(p.velocity + dv);
	p.velocity = norm(p.velocity) * v;
	debug_printf("# scattered\n");
}

__host__ __device__ void production_interaction(particle &p)
{
	debug_printf("# pair production\n");
	particle p1(p);
	particle p2(p);
	p1.time = 0.0f;
	p2.time = 0.0f;

	float n = norm(p.velocity);
	float s = uniform(0.1f, 0.9f);
	vector v = p.velocity / norm(p.velocity);
	vector u = random_unit_vector();
	p1.velocity = 0.5 * n * (v + s * u);
	p2.velocity = 0.5 * n * (v - (1.0f - s) * u);

	if (p.charge == 0.0f) {
		p1.charge = +1.0f;
		p2.charge = -1.0f;
	} else {
		p1.charge = 0.0f;
		p2.charge = 0.0f;
	}

	create_particle(p1);
	create_particle(p2);
	p.alive = false;
}

__host__ __device__ void interact(particle &p)
{
	float E = p.energy();

	if (E < Ecut) {
		debug_printf("# energy is below minimum\n");
		p.alive = false;
	}

	float inv_crosssec = 1.0f / cross_section(E);
	float p_absorption = absorption(E) * inv_crosssec;
	float p_scattering = scattering(E) * inv_crosssec;

	float r = uniform();

	if (r < p_absorption)
		absorption_interaction(p);
	else if (r < p_absorption + p_scattering)
		scattering_interaction(p);
	else /* p < (p_absorption + p_scattering + p_production) = 1.0 */
		production_interaction(p);
}
