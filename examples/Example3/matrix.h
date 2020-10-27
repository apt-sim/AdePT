#ifndef MATRIX_H
#define MATRIX_H

#include "vector.h"

struct matrix {
	__host__ __device__ matrix(float s)
	    : data{{s, 0.0f, 0.0f}, {0.0f, s, 0.0f}, {0.0f, 0.0f, s}}
	{}

	__host__ __device__ matrix(float x, float y, float z)
	    : data{{x, 0.0f, 0.0f}, {0.0f, y, 0.0f}, {0.0f, 0.0f, z}}
	{}

	__host__ __device__ matrix(float xx, float xy, float xz,
	                           float yx, float yy, float yz,
	                           float zx, float zy, float zz)
	    : data{{xx, xy, xz}, {yx, yy, yz}, {zx, zy, zz}}
	{}

	__host__ __device__ matrix(const vector &v)
	    : data{{v.x, 0.0f, 0.0f}, {0.0f, v.y, 0.0f}, {0.0f, 0.0f, v.z}}
	{}

	__host__ __device__ ~matrix()
	{}

	__host__ __device__ vector &operator[](int i)
	{
		return (vector &)(data[i]);
	}

	__host__ __device__ const vector &operator[](int i) const
	{
		return (vector &)(data[i]);
	}

	__host__ __device__ matrix &operator*=(const matrix &A)
	{
		matrix M(*this);
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				data[i][j] = dot(M[i], vector(A[0][j], A[1][j], A[2][j]));
		return *this;
	}

private:
	float data[3][3]; /* rows */
};

__host__ __device__ float det(const matrix &M)
{
	return dot(M[0], cross(M[1], M[2]));
}

__host__ __device__ matrix transpose(const matrix &M)
{
	return matrix(M[0][0], M[1][0], M[2][0],
	              M[0][1], M[1][1], M[2][1],
	              M[0][2], M[1][2], M[2][2]);
}

__host__ __device__ matrix inverse(const matrix &M)
{
	float s = 1.0f / det(M);

	float m00 = (M[1][1] * M[2][2] - M[2][1] * M[1][2]) * s;
	float m01 = (M[0][2] * M[2][1] - M[2][2] * M[0][1]) * s;
	float m02 = (M[0][1] * M[1][2] - M[1][1] * M[0][2]) * s;

	float m10 = (M[1][2] * M[2][0] - M[2][2] * M[1][0]) * s;
	float m11 = (M[0][0] * M[2][2] - M[2][0] * M[0][2]) * s;
	float m12 = (M[0][2] * M[1][0] - M[1][2] * M[0][0]) * s;

	float m21 = (M[1][0] * M[2][1] - M[2][0] * M[1][1]) * s;
	float m22 = (M[0][1] * M[2][0] - M[2][1] * M[0][0]) * s;
	float m20 = (M[0][0] * M[1][1] - M[1][0] * M[0][1]) * s;

	return matrix(m00, m01, m02, m10, m11, m12, m20, m21, m22);
}

__host__ __device__ matrix operator-(const matrix &M)
{
	return matrix(-M[0][0], -M[0][1], -M[0][2],
	              -M[1][0], -M[1][1], -M[1][2],
	              -M[2][0], -M[2][1], -M[2][2]);
}

__host__ __device__ matrix operator*(float s, const matrix &M)
{
	return matrix(s * M[0][0], s * M[0][1], s * M[0][2],
	              s * M[1][0], s * M[1][1], s * M[1][2],
	              s * M[2][0], s * M[2][1], s * M[2][2]);
}

__host__ __device__ matrix operator/(const matrix &M, float s)
{
	return 1.0f / s * M;
}

__host__ __device__ matrix operator+(const matrix &A, const matrix &B)
{
	return matrix(A[0][0] + B[0][0], A[0][1] + B[0][1], A[0][2] + B[0][2],
	              A[1][0] + B[1][0], A[1][1] + B[1][1], A[1][2] + B[1][2],
	              A[2][0] + B[2][0], A[2][1] + B[2][1], A[2][2] + B[2][2]);
}

__host__ __device__ matrix operator-(const matrix &A, const matrix &B)
{
	return matrix(A[0][0] - B[0][0], A[0][1] - B[0][1], A[0][2] - B[0][2],
	              A[1][0] - B[1][0], A[1][1] - B[1][1], A[1][2] - B[1][2],
	              A[2][0] - B[2][0], A[2][1] - B[2][1], A[2][2] - B[2][2]);
}

__host__ __device__ vector operator*(const matrix &M, const vector &v)
{
	return vector(dot(M[0], v), dot(M[1], v), dot(M[2], v));
}

__host__ __device__ inline vector operator*(const vector &v, const matrix &M)
{
	return transpose(M) * v;
}

__host__ __device__ inline matrix operator*(const matrix &A, const matrix &B)
{
	matrix Bt = transpose(B);
	return matrix(dot(A[0], Bt[0]), dot(A[0], Bt[1]), dot(A[0], Bt[2]),
	              dot(A[1], Bt[0]), dot(A[1], Bt[1]), dot(A[1], Bt[2]),
	              dot(A[2], Bt[0]), dot(A[2], Bt[1]), dot(A[2], Bt[2]));
}

#endif
