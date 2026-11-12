// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <AdePT/transport/support/Portability.hh>

#include <memory>
#include <new>
#include <AdePT/transport/support/ResourceManagement.hh>
#include "AdePT/transport/support/Global.h"

namespace adept::transport {

void freeCuda(void *ptr)
{
  if (ptr) ADEPT_DEVICE_API_CALL(Free(ptr));
}

void freeCudaHost(void *ptr)
{
  if (ptr) ADEPT_DEVICE_API_CALL(FreeHost(ptr));
}

// Why large pinned host buffers are not allocated with MallocHost
// ===============================================================
// The GPU copies data directly into (and out of) host memory, so that memory must be pinned: its pages must never
// be moved by the operating system. There are two ways to get pinned memory:
//   (a) MallocHost (cudaMallocHost / hipHostMalloc), or
//   (b) allocating ordinary memory and pinning it with HostRegister (cudaHostRegister / hipHostRegister).
// Both copy equally fast. For buffers of several GB, like the returned-step buffer, (a) causes a problem on Linux:
//
// 1. Linux regularly tidies up its memory in the background ("memory compaction", on by default since Linux 5.9,
//    e.g. on every EL9 machine). It moves pages that are in use, so that free memory joins up into larger blocks.
// 2. To move a page, Linux first removes it from the process and then interrupts every CPU core the process runs
//    on, so that each core forgets the old address. Only then does it copy the page.
// 3. MallocHost creates the buffer as shared memory. For shared memory, Linux only notices that a page is pinned
//    after step 2. It gives up, puts the page back, and all those interrupts were for nothing.
// 4. A pinned buffer of several GB consists of millions of such pages spread over the whole memory. They never move,
//    so the memory stays fragmented and Linux tries again every 30 to 60 seconds, as long as the job runs.
//
// With (b), the buffer is ordinary private memory. For such pages Linux checks for the pin before doing anything
// and simply skips them, meaning no interrupts. The 2 MB alignment also lets Linux use 2 MB pages where they
// are available, and those are not moved at all.
//
// So use allocPinnedHost for every host buffer the GPU reads or writes that is larger than a few MB; MallocHost is
// fine for small ones. It uses only standard C++ and the device API, so it works wherever CUDA or HIP work.
inline constexpr std::size_t kPinnedHostAlignment = std::size_t{2} << 20;

void *allocPinnedHost(std::size_t bytes)
{
  const std::size_t size = (bytes + kPinnedHostAlignment - 1) / kPinnedHostAlignment * kPinnedHostAlignment;
  // Own the allocation until it is registered: if HostRegister fails, ADEPT_DEVICE_API_CALL throws, and the
  // (possibly several GB) buffer must be released instead of leaked.
  struct AlignedDelete {
    void operator()(void *p) const { ::operator delete(p, std::align_val_t{kPinnedHostAlignment}); }
  };
  std::unique_ptr<void, AlignedDelete> ptr{::operator new(size, std::align_val_t{kPinnedHostAlignment})};
  ADEPT_DEVICE_API_CALL(HostRegister(ptr.get(), size, ADEPT_DEVICE_API_SYMBOL(HostRegisterDefault)));
  return ptr.release();
}

void freePinnedHost(void *ptr)
{
  if (!ptr) return;
  ADEPT_DEVICE_API_CALL(HostUnregister(ptr));
  ::operator delete(ptr, std::align_val_t{kPinnedHostAlignment});
}

void freeCudaStream(void *stream)
{
  if (stream) ADEPT_DEVICE_API_CALL(StreamDestroy(*static_cast<ADEPT_DEVICE_API_SYMBOL(Stream_t) *>(stream)));
}

void freeCudaEvent(void *event)
{
  if (event) ADEPT_DEVICE_API_CALL(EventDestroy(*static_cast<ADEPT_DEVICE_API_SYMBOL(Event_t) *>(event)));
}

// Instantiate the deleters for specific types.
#ifdef __CUDACC__
template <>
struct CudaDeleter<ADEPT_DEVICE_API_SYMBOL(Stream_t)> {
  void operator()(ADEPT_DEVICE_API_SYMBOL(Stream_t) * stream) const { freeCudaStream(stream); }
};
template <>
struct CudaDeleter<ADEPT_DEVICE_API_SYMBOL(Event_t)> {
  void operator()(ADEPT_DEVICE_API_SYMBOL(Event_t) * event) const { freeCudaEvent(event); }
};
#endif

} // namespace adept::transport
