// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "random.h"

#include <cstdio>
#include <cstdlib>

__global__ void rng_test(int n, float *rng_d)
{
	for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < n; id += blockDim.x * gridDim.x)
		rng_d[id] = uniform(0.0f, 1.0f);
}

int main(int argc, char **argv)
{
	if (argc != 3) {
		printf("Usage: %s size seed\n", argv[0]);
		return 0;
	}

	int size = std::atoi(argv[1]);
	int seed = std::atoi(argv[2]);

	float *rng_h, *rng_d;

	cudaSafe(cudaMallocHost((void **)&rng_h, size * sizeof(float)));
	cudaSafe(cudaMalloc((void **)&rng_d, size * sizeof(float)));
	cudaSafe(cudaDeviceSynchronize());

	rng_alloc(size);
	rng_init<<<8, 8>>>(size, seed);

	cudaError_t err = cudaPeekAtLastError();

	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	/* test in serial mode */
	rng_test<<<1, 1>>>(size, rng_d);

	err = cudaPeekAtLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaSafe(cudaMemcpy(rng_h, rng_d, size * sizeof(float), cudaMemcpyDeviceToHost));
	cudaSafe(cudaDeviceSynchronize());

	for (int i = 0; i < size; ++i)
		printf("%f\n", rng_h[i]);

	printf("\n");

	/* test in parallel mode */
	rng_test<<<8, 8>>>(size, rng_d);

	err = cudaPeekAtLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaSafe(cudaMemcpy(rng_h, rng_d, size * sizeof(float), cudaMemcpyDeviceToHost));
	cudaSafe(cudaDeviceSynchronize());

	for (int i = 0; i < size; ++i)
		printf("%f\n", rng_h[i]);

	rng_free();
	cudaSafe(cudaDeviceSynchronize());
	cudaSafe(cudaFreeHost(rng_h));
	cudaSafe(cudaFree(rng_d));
	return 0;
}
