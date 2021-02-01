#include "track.h"
#include "AdePT/LoopNavigator.h"

template <class fieldPropagator_t, bool BfieldOn = true>
class transportation {
public:
  static __host__ __device__ float transport(track &mytrack, fieldPropagator_t &fieldPropagator, float physics_step);
  // fieldPropagator_t fieldPropagator;  // => would need to be device pointer etc
};

// Check whether particle track intersects boundary within lenght 'physics_step'
//   If so, identify next
// Updates track parameters to end step, including position, direction (if in field).
// Calculates the next navigation state (if encountering boundary
//
// Description last updated: 2021.01.19

constexpr float kPushLinear = 1.0e-8; // * copcore::units::millimeter;

template <class fieldPropagator_t, bool BfieldOn>
__host__ __device__ float transportation<fieldPropagator_t, BfieldOn>::transport(track &mytrack,
                                                                                 fieldPropagator_t &fieldPropagator,
                                                                                 float physics_step)
{
  // return value (if step limited by physics or geometry) not used for the moment
  // now, I know which process wins, so I add the particle to the appropriate queue
  float step = 0.0;

  if (!BfieldOn) {
    step = LoopNavigator::ComputeStepAndPropagatedState(mytrack.pos, mytrack.dir, physics_step, mytrack.current_state,
                                                        mytrack.next_state);
    mytrack.pos += (step + kPushLinear) * mytrack.dir;
  } else {
    step = fieldPropagator.ComputeStepAndPropagatedState(mytrack, physics_step);
    // updates state of 'mytrack'
    if (step < physics_step) {
      assert(!mytrack.next_state.IsOnBoundary() && "Field Propagator returned step<phys -- yet boundary!");
    }
  }
  if (step < physics_step) mytrack.current_process = BfieldOn ? -2 : -1;

  return step;
}
