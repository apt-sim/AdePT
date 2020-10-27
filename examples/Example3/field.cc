#include "field.h"

#include "point.h"
#include "vector.h"

/* electromagnetic field symbols */

__host__ __device__ vector E(const point &, float)
{
	return {0.0f, 0.0f, 0.0f};
}

__host__ __device__ vector B(const point &, float)
{
	return {10.0f, 0.0f, 0.0f};
}
