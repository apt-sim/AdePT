// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "track.h"
#include "AdePT/LoopNavigator.h"

template <class fieldPropagator_t, bool BfieldOn = true>
class transportation {
public:
  static __host__ __device__ double transport(track &mytrack, fieldPropagator_t &fieldPropagator, double physics_step);
  // fieldPropagator_t fieldPropagator;  // => would need to be device pointer etc
};

// Check whether particle track intersects boundary within lenght 'physics_step'
//   If so, identify next
// Updates track parameters to end step, including position, direction (if in field).
// Calculates the next navigation state (if encountering boundary
//
// Description last updated: 2021.01.19

constexpr double kPushLinear = 1.0e-8 * copcore::units::millimeter;

template <class fieldPropagator_t, bool BfieldOn>
__host__ __device__ double transportation<fieldPropagator_t, BfieldOn>::transport(track &mytrack,
                                                                                  fieldPropagator_t &fieldPropagator,
                                                                                  double physics_step)
{
  double step = 0.0;

  if (!BfieldOn) {
    step = LoopNavigator::ComputeStepAndPropagatedState(mytrack.pos, mytrack.dir, physics_step, mytrack.current_state,
                                                        mytrack.next_state);
    mytrack.pos += (step + kPushLinear) * mytrack.dir;
  } else {
    step = fieldPropagator.ComputeStepAndPropagatedState(mytrack.energy, mytrack.mass(), mytrack.charge(), physics_step,
                                                         mytrack.pos, mytrack.dir, mytrack.current_state,
                                                         mytrack.next_state);
  }
  if (step < physics_step) mytrack.current_process = BfieldOn ? -2 : -1;

  return step;
}
