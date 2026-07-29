// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <G4Step.hh>
#include <G4Track.hh>

#include <cassert>

namespace adept::g4integration::detail {

/// Reset values conditionally changed by nuclear replay.
inline void ResetForStep(G4Track &track)
{
  track.SetTrackStatus(fAlive);
  track.SetVelocity(0.);
}

/// Reset values that this integration assigns only on some reconstructed steps.
///
/// This deliberately does not initialize the step, touch geometry, change the
/// track's step number, or overwrite kinematic and timing data.
inline void ResetForStep(G4Step &step)
{
  assert(step.GetTrack() != nullptr);
  ResetForStep(*step.GetTrack());

  step.ClearFirstStepFlag();
  step.ClearLastStepFlag();

  step.GetPostStepPoint()->SetVelocity(0.);
}

} // namespace adept::g4integration::detail
