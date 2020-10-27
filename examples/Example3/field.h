#ifndef FIELD_H
#define FIELD_H

#include "common.h"

class point;
class vector;

__host__ __device__ vector E(const point &p, float t);
__host__ __device__ vector B(const point &p, float t);

#endif
