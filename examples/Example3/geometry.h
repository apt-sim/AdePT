#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "common.h"

class point;
class vector;

__host__ __device__ bool inside_world(point p);
__host__ __device__ float distance_to_boundary(point p);
__host__ __device__ float distance_to_boundary(point p, vector dir, float tmin, float tmax);

__host__ __device__ float density(point p);
__host__ __device__ vector normal(point p);

#endif
