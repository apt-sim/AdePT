// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <memory>

namespace adept::transport {
void freeCuda(void *ptr);
void freeCudaHost(void *ptr);
void freeCudaStream(void *stream);
void freeCudaEvent(void *event);

/// Pinned (page-locked) host memory for large buffers: ordinary 2 MB aligned memory pinned with HostRegister.
/// Use it instead of MallocHost for any host buffer larger than a few MB that the GPU reads or writes:
/// on Linux, large MallocHost buffers make the kernel's memory compaction interrupt all CPU cores over and over
/// and cost several percent of throughput. A detailed explanation can be found in ResourceManagement.cuh.
void *allocPinnedHost(std::size_t bytes);
void freePinnedHost(void *ptr);

template <class T>
struct CudaDeleter {
  void operator()(T *ptr) const { freeCuda(ptr); }
};
template <class T>
struct CudaHostDeleter {
  void operator()(T *ptr) const { freeCudaHost(ptr); }
};

template <class T>
struct PinnedHostDeleter {
  void operator()(T *ptr) const { freePinnedHost(ptr); }
};

template <typename T = void, typename Deleter = CudaDeleter<T>>
using unique_ptr_cuda = std::unique_ptr<T, Deleter>;

} // namespace adept::transport
