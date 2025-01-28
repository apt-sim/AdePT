// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef RESOURCE_MANAGEMENT_HH
#define RESOURCE_MANAGEMENT_HH

#include <memory>

namespace AsyncAdePT {
void freeCuda(void *ptr);
void freeCudaHost(void *ptr);
void freeCudaStream(void *stream);
void freeCudaEvent(void *event);

template <class T>
struct CudaDeleter {
  void operator()(T *ptr) const { freeCuda(ptr); }
};
template <class T>
struct CudaHostDeleter {
  void operator()(T *ptr) const { freeCudaHost(ptr); }
};

template <typename T = void, typename Deleter = CudaDeleter<T>>
using unique_ptr_cuda = std::unique_ptr<T, Deleter>;

} // namespace AsyncAdePT

#endif // RESOURCE_MANAGEMENT_HH