// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef COMMON_H
#define COMMON_H

#ifndef __CUDACC__

#define __host__
#define __device__
#define __global__

#define __shared__
#define __constant__

#else

#define cudaSafe(code)                                                                        \
	do {                                                                                  \
		auto result = (code);                                                         \
		if (result != cudaSuccess) {                                                  \
			auto str = cudaGetErrorString(result);                                \
			fprintf(stderr, "CUDA error:%s at %s:%d\n", str, __FILE__, __LINE__); \
			cudaDeviceReset();                                                    \
			exit(result);                                                         \
		}                                                                             \
	} while (0)

void cudaCheckKernelCall();

#endif

/* debugging */

#ifdef NDEBUG
#define debug_printf(...) /* empty */
#else
#define debug_printf(...) printf(__VA_ARGS__)
#endif

#endif
