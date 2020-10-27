#include "field.h"
#include "particle.h"

#define smin 0.0001f

static inline __host__ __device__ vector acceleration(const particle &p)
{
	vector Ep = E(p.position, p.time);
	vector Bp = B(p.position, p.time);
	return p.charge / p.mass * (Ep + cross(p.velocity, Bp));
}

static __host__ __device__ void euler(particle &p, float dt)
{
	p.time += dt;
	p.position += dt * p.velocity;
	p.velocity += dt * p.acceleration;
	p.acceleration = acceleration(p);
}

static __host__ __device__ void verlet(particle &p, float dt)
{
	vector v = p.velocity;
	vector a = p.acceleration;

	p.acceleration = acceleration(p);
	p.velocity += 0.5 * dt * (a + p.acceleration);
	p.position += dt * (v + 0.5 * dt * a);
	p.time += dt;
}

__host__ __device__ void integrate(particle &p, float dt)
{
	verlet(p, dt);
}

__host__ __device__ void integrate(particle &p, float dt, float s)
{
	do {
		float ds = norm(dt * (p.velocity + 0.5 * dt * p.acceleration));

		if (ds > s) {
			dt *= 0.5;
			continue;
		}

		s -= ds;
		verlet(p, dt);

	} while (s > smin);

	if (s > 0.0)
		euler(p, s / norm(p.velocity));
}
