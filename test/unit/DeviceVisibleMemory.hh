// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file DeviceVisibleMemory.hh
 * @brief Device memory that host test code can also read and write, for objects updated by device atomics.
 *
 * Atomics must operate on device memory. Host-resident memory (managed memory on GPUs without page migration,
 * pinned host memory) is reached over PCIe, which supports only add, exchange and compare-and-swap: on AMD GPUs the
 * other atomic operations (subtract, and, or, ...) are silently lost there. HIP: fine-grained device memory, which
 * the host accesses directly. CUDA: managed memory, migrated to the device when kernels use it.
 */

#pragma once

#include <AdePT/transport/support/Portability.hh>

#include <cstddef>
#include <stdexcept>

namespace adept_test {

/// Allocate device memory visible to the host; returns the runtime status
template <class T>
ADEPT_DEVICE_API_SYMBOL(Error_t)
MallocDeviceVisible(T **ptr, std::size_t bytes)
{
#if defined(__HIP__)
  return hipExtMallocWithFlags(reinterpret_cast<void **>(ptr), bytes, hipDeviceMallocFinegrained);
#else
  return cudaMallocManaged(ptr, bytes);
#endif
}

/// As MallocDeviceVisible, throwing on failure
template <class T>
void AllocDeviceVisible(T **ptr, std::size_t bytes)
{
  const auto status = MallocDeviceVisible(ptr, bytes);
  if (status != ADEPT_DEVICE_API_SYMBOL(Success))
    throw std::runtime_error(ADEPT_DEVICE_API_SYMBOL(GetErrorString)(status));
}

} // namespace adept_test
