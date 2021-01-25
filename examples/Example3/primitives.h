// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "common.h"

class point;
class vector;

__host__ __device__ float clamp(float x, float lo, float hi);

/* 2D primitives */

__host__ __device__ float sd_plane(point p);
__host__ __device__ float sd_segment(point p, point a, point b);

/* 3D primitives */

__host__ __device__ float sd_box(point p, vector b);
__host__ __device__ float sd_capsule(point p, point a, point b, float r);
__host__ __device__ float sd_cylinder(point p, float h, float r);
__host__ __device__ float sd_sphere(point p, float r);
__host__ __device__ float sd_torus(point p, float R, float r);

#endif
