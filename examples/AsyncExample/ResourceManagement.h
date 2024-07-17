// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#ifndef RESOURCE_MANAGEMENT_CUH
#define RESOURCE_MANAGEMENT_CUH

#include <memory>

namespace AsyncAdePT {
void cudaDeleter(void *ptr);
void cudaHostDeleter(void *ptr);
void cudaStreamDeleter(void *stream);
void cudaEventDeleter(void *event);

template <typename T = void>
using unique_ptr_cuda = std::unique_ptr<T, void (*)(void *)>;

#ifdef __CUDACC__
using unique_ptr_cudaStream = std::unique_ptr<cudaStream_t, void (*)(void *)>;
using unique_ptr_cudaEvent  = std::unique_ptr<cudaEvent_t, void (*)(void *)>;
#endif

} // namespace AsyncAdePT

#endif // RESOURCE_MANAGEMENT_CUH