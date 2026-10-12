// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/// Landing statistics of the boundary crossings made by the GPU transport (/adept/validateCrossings).
///
/// After every relocation into the next volume, the step end point is compared with the surfaces it crossed, using
/// the VecGeom solids of the volumes left and entered. The surface band is +-kTolerance: a landing is off-band when
/// it lies more than kTolerance inside a volume being left or outside a volume being entered, or farther than
/// kTolerance from every crossed surface. The final direction is compared with the normal of the crossed surface to
/// count tracks turning back through it (bounce-backs). Solid navigation only.

#pragma once

#include <VecGeom/navigation/NavigationState.h>

namespace adept::transport {

/// One off-band landing, with what is needed to replay the step and the relocation (geometryValidationReplay).
struct CrossingRecord {
  vecgeom::NavigationState from; ///< state before the step (volume left)
  vecgeom::NavigationState to;   ///< state after the relocation (volume reached)
  double prePoint[3];            ///< pre-step point, global frame
  double preDirection[3];        ///< pre-step direction, global frame
  double point[3];               ///< step end point, global frame
  double direction[3];           ///< final direction, global frame
  double localPoint[3];          ///< end point in the frame of the offending volume
  double localDirection[3];      ///< final direction in that frame
  double depth;                  ///< distance beyond the surface on the wrong side (kind 1, 2) or to the nearest
                                 ///< crossed surface (kind 3), from the solid's safety
  double outwardNDotD;           ///< normal . direction at the crossed surface, oriented out of the side being left
                                 ///< (negative: bounce-back); 0 if no crossed surface holds the point
  double overlapDepth;           ///< depth of the overlap evidence (min of the two strict distances)
  double overlapWitness;         ///< signed distance along the final direction from the end point to the witness point
  unsigned long trackId;         ///< GPU track id (reproducible)
  unsigned int eventId;
  int offendingLevel; ///< level of the offending volume, in `from` if offendingInFrom, else in `to`
  int kind;           ///< 1: inside a volume being left, 2: outside a volume being entered,
                      ///< 3: farther than kTolerance from every crossed surface,
                      ///< 4: in band, but a crossing of a zero-step loop (before the stuck-track push),
                      ///< 5: end of a step without relocation, beyond kStateMismatchDepth outside the state volume,
                      ///< 6: start of a step, beyond kStateMismatchDepth outside the state volume
  int overlapType;    ///< proof from the solids: 0 none, 1 overlap between branches, 2 extrusion
  int overlapVolumeA; ///< placed-volume ids of the proven pair (extrusion: daughter, ancestor)
  int overlapVolumeB;
  unsigned short stepCounter;
  unsigned short zeroStepCounter; ///< > kStepsStuckPush: the end point was pushed by kPushStuck (stuck track)
  short charge;
  bool offendingInFrom;
  bool magneticField; ///< the step was propagated in a magnetic field (curved; the pre-step chord is approximate)
  // Kinds 5 and 6 (state mismatch): step lengths, pre-step safety and MSC displacement (-1 if not applicable)
  double physicsStep;
  double geometryStep;
  double preSafety;
  double displacement;
};

/// Depth outside the state volume beyond which a state mismatch (kinds 5, 6) is recorded.
constexpr double kStateMismatchDepth = 100. * 1.e-9;

/// Zero-step count above which the transport pushes a stuck track by kPushStuck (as in the transport kernels).
constexpr unsigned short kCrossingStuckPushSteps = 5;

/// Coarse wrong-side depth bins for the text report, in kTolerance units:
/// 0, (0,0.5], (0.5,1], (1,2], (2,10], (10,100], (100,1e4], >1e4.
constexpr int kCrossingDepthBins = 8;
/// Fine wrong-side depth histogram for the output file: bin 0 holds depth 0 (and below the first edge), then
/// kCrossingLogBinsPerDecade bins per decade from 10^kCrossingLogMinDecade to 10^kCrossingLogMaxDecade kTolerance,
/// then an overflow bin.
constexpr int kCrossingLogBinsPerDecade = 5;
constexpr int kCrossingLogMinDecade     = -4;
constexpr int kCrossingLogMaxDecade     = 8;
constexpr int kCrossingLogBins = 2 + kCrossingLogBinsPerDecade * (kCrossingLogMaxDecade - kCrossingLogMinDecade);
/// Normal . final direction of the bounce-backs, in [-1, 0).
constexpr int kBounceBins = 20;

struct CrossingValidationData {
  unsigned long long crossings{0};          ///< relocations into a next volume
  unsigned long long leavingGPU{0};         ///< ... into a volume outside the GPU regions
  unsigned long long offBand{0};            ///< landings beyond the +-kTolerance band (any kind)
  unsigned long long offBandLeft{0};        ///< kind 1
  unsigned long long offBandEntered{0};     ///< kind 2
  unsigned long long offSurface{0};         ///< kind 3
  unsigned long long offBandOverlap{0};     ///< off-band landings proven to come from an overlap or an extrusion
  unsigned long long offBandStuckPush{0};   ///< off-band landings of steps ended by the stuck-track push
  unsigned long long zeroStepLoop{0};       ///< in-band crossings of zero-step loops (2nd zero step up to the push)
  unsigned long long stateMismatchEnd{0};   ///< kind 5
  unsigned long long stateMismatchStart{0}; ///< kind 6
  unsigned long long bounceBack{0};         ///< final direction pointing back through the crossed surface
  unsigned long long tangent{0};            ///< final direction tangent to it
  unsigned long long noNormal{0};           ///< no valid normal at the crossed surface
  unsigned long long depthHistogram[kCrossingDepthBins]{};  ///< wrong-side depth of every landing (coarse)
  unsigned long long depthLogHistogram[kCrossingLogBins]{}; ///< same, fine logarithmic bins
  unsigned long long bounceHistogram[kBounceBins]{};        ///< normal . direction of the bounce-backs
  unsigned int recordCapacity{0};
  unsigned int nRecords{0}; ///< records seen (records beyond the capacity are dropped)
  CrossingRecord *records{nullptr};
};

} // namespace adept::transport
