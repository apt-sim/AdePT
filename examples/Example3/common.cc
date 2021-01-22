// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifdef __CUDACC__

#include "common.h"

#include <cstdio>
#include <cstdlib>

void cudaCheckKernelCall()
{
	cudaError_t err = cudaPeekAtLastError();
	if (err != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(err));
		exit(1);
	}
}

#endif
