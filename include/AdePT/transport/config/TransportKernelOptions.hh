// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace adept::transport {

/// @brief Runtime options copied into transport kernels.
struct TransportKernelOptions {
  bool returnAllSteps{false};
  bool returnLastStep{false};
  /// Maximum charged-particle looper count before killing the track.
  /// The transport service normalizes 0 to the maximum unsigned short value to disable this kill.
  unsigned short maxChargedLooperCount{500};
};

} // namespace adept::transport
