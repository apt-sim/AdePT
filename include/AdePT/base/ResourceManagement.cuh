// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#ifndef RESOURCE_MANAGEMENT_CUH
#define RESOURCE_MANAGEMENT_CUH

#include <AdePT/core/Portability.hh>

#include <memory>
#include <AdePT/base/ResourceManagement.hh>
#include "AdePT/copcore/Global.h"

namespace AsyncAdePT {

void freeCuda(void *ptr)
{
  if (ptr) ADEPT_DEVICE_API_CALL(Free(ptr));
}

void freeCudaHost(void *ptr)
{
  if (ptr) ADEPT_DEVICE_API_CALL(FreeHost(ptr));
}

void freeCudaStream(void *stream)
{
  if (stream) ADEPT_DEVICE_API_CALL(StreamDestroy(*static_cast<cudaStream_t *>(stream)));
}

void freeCudaEvent(void *event)
{
  if (event) ADEPT_DEVICE_API_CALL(EventDestroy(*static_cast<cudaEvent_t *>(event)));
}

// Instantiate the deleters for specific types.
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

} // namespace AsyncAdePT

#endif // RESOURCE_MANAGEMENT_CUH
