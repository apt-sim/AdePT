// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/// Hooks of the optional transport validation in the transport kernels. Without ADEPT_GEOMETRY_VALIDATION they
/// expand to nothing, so the kernels carry no validation code.

#pragma once

#ifdef ADEPT_GEOMETRY_VALIDATION
#include <AdePT/transport/geometry_validation/CrossingValidation.cuh>
/// Check the landing of the relocation from `from` into `to` (see /adept/validateCrossings): `track` after the step,
/// its `charge`, pre-step point and direction, end point and final direction.
#define ADEPT_VALIDATE_CROSSING(track, charge, from, to, prePos, preDir, pos, dir) \
  adept::transport::ValidateCrossingIfEnabled(track, charge, from, to, prePos, preDir, pos, dir)
/// Check that the point lies in the volume of its navigation state (kind 5: step end, 6: step start).
#define ADEPT_VALIDATE_STATE(track, charge, state, prePos, preDir, pos, dir, kind, physStep, geomStep, safety)       \
  adept::transport::ValidateStateIfEnabled(track, charge, state, prePos, preDir, pos, dir, kind, physStep, geomStep, \
                                           safety)
#else
#define ADEPT_VALIDATE_STATE(track, charge, state, prePos, preDir, pos, dir, kind, physStep, geomStep, safety)
#define ADEPT_VALIDATE_CROSSING(track, charge, from, to, prePos, preDir, pos, dir)
#endif
