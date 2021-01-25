// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "render.h"

#include "point.h"
#include "ppm.h"
#include "primitives.h"
#include "transform.h"
#include "vector.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unistd.h>

using std::abs;
using std::max;
using std::min;
using std::tan;

/* simple scene with ground plane and a few primitives */

__host__ __device__ float sd_scene(point p)
{
	const transform tr_box(quaternion{vector{0, 1, 0}, 3.1415 / 4}, point{0.0f, -2.0f, -2.0f});
	const transform tr_cyl(quaternion{vector{1, 1, 1}, -3.1415 / 3}, point{0.0f, 0.0f, 0.0f});

	float d = sd_plane(p - vector{0.0f, -3.0f, 0.0f});
	d = min(d, sd_cylinder(tr_cyl(p - vector{-9.0f, 6.0f, 5.0f}), 5.0f, 1.5f));
	d = min(d, sd_sphere(p - vector{-3.0f, 8.0f, 0.0f}, 2.0f));
	d = min(d, sd_capsule(p, point{8.0f, 8.0f, 3.0f}, point{5.0f, 3.0f, 3.0f}, 2.0f));
	d = min(d, sd_box(tr_box(p), vector{4, 4, 4}));
	d = min(d, sd_torus(tr_cyl(p - vector{2.0f, 8.0f, 9.0f}), 2.0, 0.5));
	return d;
}

__host__ __device__ vector normal(point p)
{
	float h = 0.0001;
	float d = sd_scene(p);
	float x = sd_scene(point(p.x + h, p.y, p.z)) - d;
	float y = sd_scene(point(p.x, p.y + h, p.z)) - d;
	float z = sd_scene(point(p.x, p.y, p.z + h)) - d;
	vector n(x, y, z);
	return n /= norm(n);
}

__host__ __device__ float raymarch(point ro, vector rd)
{
	float tmin = 1.000f;
	float tmax = 100.0f;

	float t = tmin;
	float d = sd_scene(ro + t * rd);

	while (t < tmax && abs(d) >= 0.001 * t)
		t += d, d = sd_scene(ro + t * rd);

	return t > tmax ? -1.0 : t;
}

__host__ __device__ int normal_to_rgb(vector n)
{
	n /= 2.0 * norm(n);
	n += {0.5f, 0.5f, 0.5f};
	int r = 255 * n.x;
	int g = 255 * n.y;
	int b = 255 * n.z;
	return (r << 16) + (g << 8) + b;
}

__host__ __device__ int color(point ro, vector rd)
{
	float t = raymarch(ro, rd);

	if (ro.y + t * rd.y < 0.001)
		return 0x555555;

	if (t < 0.0)
		return normal_to_rgb({0.0f, 0.0f, clamp(ro.y - rd.y, 0.0f, 1.0f)});

	return normal_to_rgb(normal(ro + t * rd));
}

#if defined(__CUDACC__)
__global__ void render_kernel(int *image, point origin, vector bl, vector u, vector v, int w, int h,
			      float pw, float ph)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= w || j >= h)
		return;

	vector rd = bl + (i + 0.5f) * pw * u + (j + 0.5f) * ph * v;
	image[j * w + i] = color(origin, rd);
}
#else
void render_kernel(int *image, point origin, vector bl, vector u, vector v, int w, int h, float pw,
		   float ph)
{
	(void)h;
	for (int j = 0; j < h; ++j) {
		for (int i = 0; i < w; ++i) {
			vector rd = bl + (i + 0.5f) * pw * u + (j + 0.5f) * ph * v;
			image[j * w + i] = color(origin, rd);
		}
	}
}
#endif

void render(char *name, int width, int height)
{
	point origin{0.0f, 5.0f, -15.0f};
	point lookat{0.0f, 5.0f, 0.0f};
	vector up{0.0f, 1.0f, 0.0f};

	vector w = lookat - origin;
	vector u = cross(w, up);
	vector v = cross(u, w);

	w /= norm(w);
	u /= norm(u);
	v /= norm(v);

	float fovy = 3.1415926535f / 2.0f;
	float aspect = (float)width / (float)height;

	float H = tan(fovy / 2.0f);
	float W = aspect * H;

	point ro = origin;
	point bl = point(w - 0.5f * (W * u + H * v));

	float pixel_w = W / (float)width;
	float pixel_h = H / (float)height;

#if !defined(__CUDACC__)
	int *image = (int *)malloc(height * width * sizeof(int));
	render_kernel(image, ro, bl, u, v, width, height, pixel_w, pixel_h);
	write_ppm(name, image, width, height);
	free(image);
#else
	int *image_h, *image_d;
	dim3 threads(8, 8);
	dim3 blocks(width / 8 + 1, height / 8 + 1);
	cudaSafe(cudaMallocHost((void **)&image_h, height * width * sizeof(int)));
	cudaSafe(cudaMalloc((void **)&image_d, height * width * sizeof(int)));
	cudaSafe(cudaDeviceSynchronize());
	render_kernel<<<blocks, threads>>>(image_d, ro, bl, u, v, width, height, pixel_w, pixel_h);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaSafe(cudaDeviceSynchronize());
	cudaSafe(
		cudaMemcpy(image_h, image_d, height * width * sizeof(int), cudaMemcpyDeviceToHost));
	cudaSafe(cudaDeviceSynchronize());
	sleep(1);
	write_ppm(name, image_h, width, height);
	cudaSafe(cudaFreeHost(image_h));
	cudaSafe(cudaFree(image_d));
#endif
}
