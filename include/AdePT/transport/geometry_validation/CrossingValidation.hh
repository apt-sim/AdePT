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

namespace adept::transport {

/// One off-band landing, enough to replay it on the offending solid (local frame) or in the full geometry (global).
struct CrossingRecord {
  double point[3];          ///< step end point, global frame
  double direction[3];      ///< final direction, global frame
  double localPoint[3];     ///< end point in the frame of the offending volume
  double localDirection[3]; ///< final direction in that frame
  double depth;             ///< distance beyond the surface on the wrong side (kind 1, 2) or to the nearest crossed
                            ///< surface (kind 3), from the solid's safety
  double outwardNDotD;      ///< normal . direction at the crossed surface, oriented out of the side being left
                            ///< (negative: bounce-back); 0 if no crossed surface holds the point
  int offendingVolume;      ///< placed-volume id of the volume evaluated at localPoint
  int fromVolume;           ///< placed-volume id of the volume left
  int toVolume;             ///< placed-volume id of the volume reached
  int kind;                 ///< 1: inside a volume being left, 2: outside a volume being entered,
                            ///< 3: farther than kTolerance from every crossed surface
  int overlapType;          ///< proof from the solids: 0 none, 1 the point is strictly inside two volumes of
                            ///< different branches (overlap), 2 strictly inside a volume and outside its ancestor
                            ///< (extrusion)
  int overlapVolumeA;       ///< placed-volume ids of the proven pair (extrusion: daughter, ancestor)
  int overlapVolumeB;
  double overlapDepth;   ///< depth of the evidence: min of the two strict distances, beyond kTolerance
  double overlapWitness; ///< distance along the final direction from the end point to the witness point
};

/// Depth histogram bins, in kTolerance units: 0, (0,0.5], (0.5,1], (1,2], (2,10], (10,100], (100,1e4], >1e4.
constexpr int kCrossingDepthBins = 8;

struct CrossingValidationData {
  unsigned long long crossings{0};      ///< relocations into a next volume
  unsigned long long leavingGPU{0};     ///< ... into a volume outside the GPU regions
  unsigned long long offBand{0};        ///< landings beyond the +-kTolerance band (any kind)
  unsigned long long offBandLeft{0};    ///< kind 1
  unsigned long long offBandEntered{0}; ///< kind 2
  unsigned long long offSurface{0};     ///< kind 3
  unsigned long long offBandOverlap{0}; ///< off-band landings proven to come from an overlap or an extrusion
  unsigned long long bounceBack{0};     ///< final direction pointing back through the crossed surface
  unsigned long long tangent{0};        ///< final direction tangent to it
  unsigned long long noNormal{0};       ///< no valid normal at the crossed surface
  unsigned long long depthHistogram[kCrossingDepthBins]{}; ///< wrong-side depth of every landing
  unsigned int recordCapacity{0};
  unsigned int nRecords{0}; ///< off-band landings seen (records beyond the capacity are dropped)
  CrossingRecord *records{nullptr};
};

} // namespace adept::transport
