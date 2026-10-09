// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/// Hooks of the optional transport validation in the transport kernels. Without ADEPT_GEOMETRY_VALIDATION they
/// expand to nothing, so the kernels carry no validation code.

#pragma once

#ifdef ADEPT_GEOMETRY_VALIDATION
#include <AdePT/transport/geometry_validation/CrossingValidation.cuh>
/// Check the landing of the relocation from `from` into `to` (see /adept/validateCrossings).
#define ADEPT_VALIDATE_CROSSING(from, to, pos, dir) adept::transport::ValidateCrossingIfEnabled(from, to, pos, dir)
#else
#define ADEPT_VALIDATE_CROSSING(from, to, pos, dir)
#endif
