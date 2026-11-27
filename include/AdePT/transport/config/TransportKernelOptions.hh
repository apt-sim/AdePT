// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace adept::transport {

/// Threads per block of the transport kernel launches, also their launch bound: without it the AMD compiler sizes
/// per-thread arrays promoted to LDS for 1024-thread blocks, which limits occupancy.
inline constexpr int kTransportThreads = 32;

/// @brief Runtime options copied into transport kernels.
struct TransportKernelOptions {
  bool returnAllSteps{false};
  bool returnLastStep{false};
  /// Maximum charged-particle looper count before killing the track.
  /// The transport service normalizes 0 to the maximum unsigned short value to disable this kill;
  /// active limits are capped below the overflow range.
  unsigned short maxChargedLooperCount{500};
};

} // namespace adept::transport
