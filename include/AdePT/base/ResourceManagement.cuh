// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#ifndef RESOURCE_MANAGEMENT_CUH
#define RESOURCE_MANAGEMENT_CUH

#include <memory>
#include "AdePT/copcore/Global.h"

namespace AsyncAdePT {

void freeCuda(void *ptr)
{
  if (ptr) COPCORE_CUDA_CHECK(cudaFree(ptr));
}

void freeCudaHost(void *ptr)
{
  if (ptr) COPCORE_CUDA_CHECK(cudaFreeHost(ptr));
}

void freeCudaStream(void *stream)
{
  if (stream) COPCORE_CUDA_CHECK(cudaStreamDestroy(*static_cast<cudaStream_t *>(stream)));
}

void freeCudaEvent(void *event)
{
  if (event) COPCORE_CUDA_CHECK(cudaEventDestroy(*static_cast<cudaEvent_t *>(event)));
}

template <class T>
struct CudaDeleter {
  void operator()(T *ptr) const { freeCuda(ptr); }
};
template <class T>
struct CudaHostDeleter {
  void operator()(T *ptr) const { freeCudaHost(ptr); }
};

#ifdef __CUDACC__
template <>
struct CudaDeleter<cudaStream_t> {
  void operator()(cudaStream_t *stream) const { freeCudaStream(stream); }
};
template <>
struct CudaDeleter<cudaEvent_t> {
  void operator()(cudaEvent_t *event) const { freeCudaEvent(event); }
};
#endif
template <typename T = void, typename Deleter = CudaDeleter<T>>
using unique_ptr_cuda = std::unique_ptr<T, Deleter>;

} // namespace AsyncAdePT

#endif // RESOURCE_MANAGEMENT_CUH