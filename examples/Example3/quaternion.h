#ifndef QUATERNION_H
#define QUATERNION_H

#include "vector.h"

struct quaternion {
	__host__ __device__ quaternion(const vector &v) : x(v.x), y(v.y), z(v.z), w(0.0f)
	{}

	__host__ __device__ quaternion(float x, float y, float z, float w) : x(x), y(y), z(z), w(w)
	{}

	__host__ __device__ quaternion(const vector &axis, float angle)
	{
		using std::cos;
		using std::sin;
		float s;
		sincosf(angle / 2.0f, &s, &w);
		s /= norm(axis);
		x = s * axis.x;
		y = s * axis.y;
		z = s * axis.z;
	}

	__host__ __device__ float &operator[](int k)
	{
		return (&x)[k];
	}

	__host__ __device__ float operator[](int k) const
	{
		return (&x)[k];
	}

	__host__ __device__ quaternion &operator+=(float s)
	{
		x += s;
		y += s;
		z += s;
		w += s;
		return *this;
	}

	__host__ __device__ quaternion &operator-=(float s)
	{
		x -= s;
		y -= s;
		z -= s;
		w -= s;
		return *this;
	}

	__host__ __device__ quaternion &operator*=(float s)
	{
		x *= s;
		y *= s;
		z *= s;
		w *= s;
		return *this;
	}

	__host__ __device__ quaternion &operator/=(float s)
	{
		x /= s;
		y /= s;
		z /= s;
		w /= s;
		return *this;
	}

	__host__ __device__ quaternion &operator+=(const quaternion &q)
	{
		x += q.x;
		y += q.y;
		z += q.z;
		w += q.w;
		return *this;
	}

	__host__ __device__ quaternion &operator-=(const quaternion &q)
	{
		x -= q.x;
		y -= q.y;
		z -= q.z;
		w -= q.w;
		return *this;
	}

	float x, y, z, w;
};

__host__ __device__ inline quaternion operator-(const quaternion &q)
{
	return quaternion(-q.x, -q.y, -q.z, -q.w);
}

__host__ __device__ inline quaternion operator*(float s, const quaternion &q)
{
	return quaternion(s * q.x, s * q.y, s * q.z, s * q.w);
}

__host__ __device__ inline quaternion operator/(const quaternion &q, float s)
{
	return quaternion(q.x / s, q.y / s, q.z / s, q.w / s);
}

__host__ __device__ inline quaternion operator+(const quaternion &q1, const quaternion &q2)
{
	return quaternion(q1.x + q2.x, q1.y + q2.y, q1.z + q2.z, q1.w + q2.w);
}

__host__ __device__ inline quaternion operator-(const quaternion &q1, const quaternion &q2)
{
	return quaternion(q1.x - q2.x, q1.y - q2.y, q1.z - q2.z, q1.w - q2.w);
}

__host__ __device__ inline float dot(const quaternion &q1, const quaternion &q2)
{
	return q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
}

__host__ __device__ inline quaternion cross(const quaternion &q1, const quaternion &q2)
{
	float x = q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
	float y = -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
	float z = q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z;
	float w = -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w;
	return quaternion(x, y, z, w);
}

__host__ __device__ inline float norm(const quaternion &q)
{
	using std::sqrt;
	return sqrt(dot(q, q));
}

__host__ __device__ inline float norm2(const quaternion &q)
{
	return dot(q, q);
}

__host__ __device__ inline quaternion conjugate(const quaternion &q)
{
	return quaternion(-q.x, -q.y, -q.z, q.w);
}

__host__ __device__ inline quaternion inverse(const quaternion &q)
{
	return conjugate(q) / norm(q);
}

#endif /* _QUATERNION_H_ */
