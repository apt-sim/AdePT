#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "matrix.h"
#include "point.h"
#include "quaternion.h"
#include "vector.h"

__host__ __device__ matrix rotation_matrix(const quaternion &q);

class transform {
public:
	__host__ __device__ transform()
	    : basis(matrix(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f)),
	      origin(0.0f, 0.0f, 0.0f), type(IDENTITY)
	{}

	__host__ __device__ transform(const transform &t)
	    : basis(t.basis), origin(t.origin), type(t.type)
	{}

	__host__ __device__ transform(const matrix &M, const point &x)
	    : basis(M), origin(x), type(AFFINE)
	{}

	__host__ __device__ transform(const quaternion &q, const point &x)
	    : basis(rotation_matrix(q)), origin(x), type(ROTATION | TRANSLATION)
	{}

	__host__ __device__ ~transform()
	{}

	__host__ __device__ point operator()(const point &x) const
	{
		return origin + basis * x;
	}

	__host__ __device__ vector operator()(const vector &v) const
	{
		return basis * v;
	}

	__host__ __device__ matrix operator()(const matrix &M) const
	{
		return basis * M * (type & SCALING ? transpose(basis) : inverse(basis));
	}

	__host__ __device__ transform operator*(const transform &t)
	{
		return transform(basis * t.basis, (*this)(t.origin));
	}

	__host__ __device__ transform &operator*=(const transform &t)
	{
		origin += basis * t.origin;
		basis *= t.basis;
		type |= t.type;
		return *this;
	}

	__host__ __device__ transform &translate(const vector &v)
	{
		origin -= basis * v;
		type |= TRANSLATION;
		return *this;
	}

	__host__ __device__ transform &rotate(const quaternion &q)
	{
		basis *= rotation_matrix(q);
		type |= ROTATION;
		return *this;
	}

	__host__ __device__ void scale(const float s)
	{
		basis *= matrix(s);
		type |= SCALING;
	}

	__host__ __device__ void scale(float x, float y, float z)
	{
		basis *= matrix(x, y, z);
		type |= SCALING;
	}

private:
	enum
	{
		IDENTITY = 0x00,
		TRANSLATION = 0x01,
		ROTATION = 0x02,
		SCALING = 0x04,
		LINEAR = ROTATION | SCALING,
		AFFINE = LINEAR | TRANSLATION
	};

	matrix basis;
	point origin;
	unsigned char type;

	__host__ __device__ friend transform inverse(const transform &t);
};

__host__ __device__ transform inverse(const transform &t)
{
	matrix basis = t.type & transform::SCALING ? inverse(t.basis) : transpose(t.basis);
	point origin = point(-(basis * t.origin));
	return transform(basis, origin);
}

__host__ __device__ matrix rotation_matrix(const quaternion &q)
{
	float s = 2.0f / norm2(q);
	float x = q[0], y = q[1], z = q[2], w = q[3];
	float xs = x * s, ys = y * s, zs = z * s;
	float x2 = x * xs, y2 = y * ys, z2 = z * zs;
	float xy = x * ys, xz = x * zs, yz = y * zs;
	float wx = w * xs, wy = w * ys, wz = w * zs;

	return matrix(1.0f - (y2 + z2), xy - wz, xz + wy, xy + wz, 1.0f - (x2 + z2), yz - wx,
		      xz - wy, yz + wx, 1.0f - (x2 + y2));
}

#endif
