// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "primitives.h"

#include "point.h"
#include "vector.h"

#include <algorithm>
#include <cmath>

using std::abs;
using std::hypot;
using std::max;
using std::min;

__host__ __device__ float clamp(float x, float lo, float hi)
{
	return max(lo, min(hi, x));
}

/* 2D primitives */

__host__ __device__ float sd_plane(point p)
{
	return p.y;
}

__host__ __device__ float sd_segment(point p, point a, point b)
{
	vector pa = p - a;
	vector ba = b - a;
	float t = clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
	return norm(pa - t * ba);
}

/* 3D primitives */

__host__ __device__ float sd_box(point p, vector b)
{
	return (abs(p) - 0.5f * b).max();
}

__host__ __device__ float sd_capsule(point p, point a, point b, float r)
{
	return sd_segment(p, a, b) - r;
}

__host__ __device__ float sd_cylinder(point p, float h, float r)
{
	return max(hypot(p.x, p.z) - r, abs(p.y) - h / 2.0f);
}

__host__ __device__ float sd_sphere(point p, float r)
{
	return norm(p) - r;
}

__host__ __device__ float sd_torus(point p, float R, float r)
{
	return hypot(hypot(p.x, p.y) - R, p.z) - r;
}
