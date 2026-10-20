// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <AdePT/AdePTBuildFlags.hh>

#include <AdePT/transport/config/CMSSteppingActionParams.hh>
#include <AdePT/transport/config/LHCbSteppingActionParams.hh>

/// @brief Empty parameter set for the stepping actions that have no configurable parameters.
struct EmptySteppingActionParams {};

namespace adept {
/// @brief Parameters of the stepping action selected at build time (ADEPT_STEPPINGACTION).
/// @details Host-visible: AdePTBuildFlags.hh defines the selection macro for every includer, inside and outside
/// of AdePT, so the host configuration can carry one object of the selected type through the transport
/// interfaces to the kernels, without the transport interfaces naming any experiment.
#if ADEPT_STEPACTION_TYPE == 1
using SteppingActionParamsT = CMSSteppingActionParams;
#elif ADEPT_STEPACTION_TYPE == 2
using SteppingActionParamsT = LHCbSteppingActionParams;
#else
using SteppingActionParamsT = EmptySteppingActionParams; // NONE, and ATLAS, which has no parameters
#endif
} // namespace adept
