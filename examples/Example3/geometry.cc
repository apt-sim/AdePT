// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "geometry.h"

#include "point.h"
#include "primitives.h"
#include "vector.h"

#include <cmath>

using std::abs;

/* capsule along x-axis */

__host__ __device__ bool inside_world(point p)
{
	return distance_to_boundary(p) < 20.0f;
}

__host__ __device__ float distance_to_boundary(point p)
{
	const point p1{10.0f, 0.0f, 0.0f};
	const point p2{20.0f, 0.0f, 0.0f};
	return sd_capsule(p, p1, p2, 5.0f);
}

__host__ __device__ float distance_to_boundary(point p, vector dir, float tmin, float tmax)
{
	float t = tmin;
	float d = distance_to_boundary(p + t * dir);

	while (t < tmax && abs(d) >= 0.0001 * t)
		t += d, d = distance_to_boundary(p + t * dir);

	return t > tmax ? -1.0 : t;
}

__host__ __device__ float density(point p)
{
	return distance_to_boundary(p) < 0.0f ? 1.0f : 0.0f;
}

__host__ __device__ vector normal(point p)
{
	float h = 0.0001;
	float d = distance_to_boundary(p);
	float x = distance_to_boundary(point(p.x + h, p.y, p.z)) - d;
	float y = distance_to_boundary(point(p.x, p.y + h, p.z)) - d;
	float z = distance_to_boundary(point(p.x, p.y, p.z + h)) - d;
	vector n(x, y, z);
	return n /= norm(n);
}
