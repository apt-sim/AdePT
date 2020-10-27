#ifndef VECTOR_H
#define VECTOR_H

#include "common.h"

#include <cmath>

struct vector {
	vector() = default;

	__host__ __device__ vector(float x, float y, float z) : x(x), y(y), z(z)
	{}

	__host__ __device__ float &operator[](int k)
	{
		return (&x)[k];
	}

	__host__ __device__ float operator[](int k) const
	{
		return (&x)[k];
	}

	__host__ __device__ vector &operator+=(float s)
	{
		x += s;
		y += s;
		z += s;
		return *this;
	}

	__host__ __device__ vector &operator-=(float s)
	{
		x -= s;
		y -= s;
		z -= s;
		return *this;
	}

	__host__ __device__ vector &operator*=(float s)
	{
		x *= s;
		y *= s;
		z *= s;
		return *this;
	}

	__host__ __device__ vector &operator/=(float s)
	{
		float k = 1.0f / s;
		x *= k;
		y *= k;
		z *= k;
		return *this;
	}

	__host__ __device__ vector &operator+=(const vector &v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	__host__ __device__ vector &operator-=(const vector &v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	__host__ __device__ float min() const
	{
		return x < z ? (x < y ? x : y) : (y < z ? y : z);
	}

	__host__ __device__ float max() const
	{
		return x > z ? (x > y ? x : y) : (y > z ? y : z);
	}

	__host__ __device__ int min_element() const
	{
		return x < z ? (x < y ? 0 : 1) : (y < z ? 1 : 2);
	}

	__host__ __device__ int max_element() const
	{
		return x > z ? (x > y ? 0 : 1) : (y > z ? 1 : 2);
	}

	__host__ __device__ int closest_axis() const
	{
		using std::abs;
		return vector(abs(x), abs(y), abs(z)).max_element();
	}

	float x, y, z;
};

__host__ __device__ inline vector operator-(const vector &v)
{
	return vector(-v.x, -v.y, -v.z);
}

__host__ __device__ inline vector operator*(float s, const vector &v)
{
	return vector(s * v.x, s * v.y, s * v.z);
}

__host__ __device__ inline vector operator/(const vector &v, float s)
{
	return vector(v.x / s, v.y / s, v.z / s);
}

__host__ __device__ inline vector operator+(const vector &v1, const vector &v2)
{
	return vector(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ inline vector operator-(const vector &v1, const vector &v2)
{
	return vector(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ inline vector abs(const vector &v)
{
	using std::abs;
	return vector(abs(v.x), abs(v.y), abs(v.z));
}

__host__ __device__ inline float dot(const vector &v1, const vector &v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ inline vector cross(const vector &v1, const vector &v2)
{
	float x = v1.y * v2.z - v1.z * v2.y;
	float y = v1.z * v2.x - v1.x * v2.z;
	float z = v1.x * v2.y - v1.y * v2.x;
	return vector(x, y, z);
}

__host__ __device__ inline float norm2(const vector &v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ inline float norm(const vector &v)
{
	using std::sqrt;
	return sqrt(norm2(v));
}

__host__ __device__ inline float angle(const vector &v1, const vector &v2)
{
	using std::acos;
	using std::sqrt;
	return acos(dot(v1, v2) / sqrt(norm2(v1) * norm2(v2)));
}

#endif
