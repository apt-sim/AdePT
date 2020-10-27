#include "particle.h"
#include "user.h"

#include <iostream>

void user_init()
{ /* empty */ }

void user_exit()
{ /* empty */ }

void event_init()
{ /* empty */ }

void event_exit()
{ /* empty */ }

__host__ __device__ void track_init(particle &p)
{
	(void)p;
	debug_printf("%f %f %f\n", p.position.x, p.position.y, p.position.z);
}

__host__ __device__ void track_exit(particle &p)
{
	(void)p;
	debug_printf("%f %f %f\n\n", p.position.x, p.position.y, p.position.z);
}

__host__ __device__ void step_init(particle &p)
{
	(void)p;
	debug_printf("%f %f %f\n", p.position.x, p.position.y, p.position.z);
}

__host__ __device__ void step_exit(particle &p)
{
	(void)p;
	debug_printf("%f %f %f\n", p.position.x, p.position.y, p.position.z);
}
