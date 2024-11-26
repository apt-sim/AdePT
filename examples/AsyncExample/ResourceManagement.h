// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#ifndef RESOURCE_MANAGEMENT_CUH
#define RESOURCE_MANAGEMENT_CUH

#include <memory>

namespace AsyncAdePT {
void freeCuda(void *ptr);
void freeCudaHost(void *ptr);

template <class T>
struct CudaDeleter {
  void operator()(T *ptr) const { freeCuda(ptr); }
};
template <class T>
struct CudaHostDeleter {
  void operator()(T *ptr) const { freeCudaHost(ptr); }
};

void freeCudaStream(void *stream);
void freeCudaEvent(void *event);

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