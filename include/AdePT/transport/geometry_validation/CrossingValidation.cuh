// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/// Device side of the crossing validation (see CrossingValidation.hh).

#pragma once

#include <AdePT/transport/geometry_validation/CrossingValidation.hh>
#include <AdePT/transport/state/DeviceGlobals.cuh>

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/PlacedVolume.h>

namespace adept::transport {

/// Validation data on the device, nullptr when the validation is off.
extern __constant__ __device__ CrossingValidationData *gCrossingValidation;

namespace crossing_detail {

/// Signed distance of a point to a solid's surface from its safeties: > 0 inside, < 0 outside.
__device__ inline double SignedDistance(vecgeom::VUnplacedVolume const *solid, vecgeom::Vector3D<double> const &local)
{
  const auto inside = solid->Inside(local);
  if (inside == vecgeom::EInside::kOutside) return -vecgeom::Max(solid->SafetyToIn(local), 0.);
  return vecgeom::Max(solid->SafetyToOut(local), 0.);
}

__device__ inline int DepthBin(double depth)
{
  const double d = depth / vecgeom::kTolerance;
  return d <= 0. ? 0 : d <= 0.5 ? 1 : d <= 1. ? 2 : d <= 2. ? 3 : d <= 10. ? 4 : d <= 100. ? 5 : d <= 1.e4 ? 6 : 7;
}

/// Bin of the fine logarithmic depth histogram (see kCrossingLogBins).
__device__ inline int DepthLogBin(double depth)
{
  const double d = depth / vecgeom::kTolerance;
  if (d <= 0.) return 0;
  const double x = (log10(d) - kCrossingLogMinDecade) * kCrossingLogBinsPerDecade;
  if (x < 0.) return 0;
  return x >= kCrossingLogBins - 2 ? kCrossingLogBins - 1 : 1 + int(x);
}

/// Context of the step that ended with the relocation.
struct StepContext {
  vecgeom::Vector3D<double> prePoint;
  vecgeom::Vector3D<double> preDirection;
  unsigned long trackId;
  unsigned int eventId;
  unsigned short stepCounter;
  unsigned short zeroStepCounter;
  short charge;
};

struct OverlapProof {
  int type{0}; // 1 overlap between branches, 2 extrusion
  int volumeA{-1};
  int volumeB{-1};
  double depth{0.};
  double witness{0.};
};

/// Prove an overlap at `q` from the solids: strictly inside (beyond kTolerance) volumes of both paths below their
/// deepest common volume, which lie in different branches, or strictly inside a volume and strictly outside one of
/// its ancestors (extrusion). Levels below the common volume and the common volume itself are evaluated.
__device__ inline bool ProveOverlapAt(vecgeom::NavigationState const &from, vecgeom::NavigationState const &to,
                                      int common, vecgeom::Vector3D<double> const &q, OverlapProof &proof)
{
  auto strict = [&](vecgeom::NavigationState const &path, int level) {
    vecgeom::Transformation3D m; // TopMatrix fills an identity transformation
    path.TopMatrix(level, m);
    const double sd = SignedDistance(path.At(level)->GetLogicalVolume()->GetUnplacedVolume(), m.Transform(q));
    return vecgeom::Abs(sd) > vecgeom::kTolerance ? sd : 0.;
  };
  int inside[2]     = {-1, -1};
  double depth[2]   = {0., 0.};
  const int tops[2] = {from.GetLevel(), to.GetLevel()};
  for (int side = 0; side < 2; ++side) {
    auto const &path = side == 0 ? from : to;
    for (int level = common + 1; level <= tops[side]; ++level) {
      const double d = strict(path, level);
      if (d > 0.) {
        inside[side] = level;
        depth[side]  = d;
      }
    }
  }
  if (inside[0] >= 0 && inside[1] >= 0) {
    proof = {1, static_cast<int>(from.At(inside[0])->id()), static_cast<int>(to.At(inside[1])->id()),
             vecgeom::Min(depth[0], depth[1]), 0.};
    return true;
  }
  for (int side = 0; side < 2; ++side) {
    if (inside[side] < 0) continue;
    auto const &path = side == 0 ? from : to;
    for (int level = vecgeom::Max(common, 0); level < inside[side]; ++level) {
      const double d = strict(path, level);
      if (d < 0.) {
        proof = {2, static_cast<int>(path.At(inside[side])->id()), static_cast<int>(path.At(level)->id()),
                 vecgeom::Min(depth[side], -d), 0.};
        return true;
      }
    }
  }
  return false;
}

} // namespace crossing_detail

/// @brief Check the landing of a relocation from `from` into `to` at the final position and direction.
/// @details Levels below the deepest volume common to both paths are crossed: those of `from` are left, those of `to`
/// are entered. The crossed surface is the deepest left level (or the outermost entered one) within kTolerance of
/// the point. Not inlined (internal linkage): the transport kernels only carry a guarded call, so their code and
/// results stay as without the check.
static __device__ __noinline__ void ValidateCrossing(vecgeom::NavigationState const &from,
                                                     vecgeom::NavigationState const &to,
                                                     vecgeom::Vector3D<double> const &point,
                                                     vecgeom::Vector3D<double> const &direction,
                                                     crossing_detail::StepContext const &step, bool leavesGPU)
{
  using namespace crossing_detail;
  auto *data = gCrossingValidation;
  atomicAdd(&data->crossings, 1ull);
  if (leavesGPU) atomicAdd(&data->leavingGPU, 1ull);

  const int fromLevel = from.GetLevel(), toLevel = to.GetLevel();
  int common = -1;
  for (int level = 0; level <= vecgeom::Min(fromLevel, toLevel) && from.At(level) == to.At(level); ++level)
    common = level;

  // Worst wrong-side landing over the crossed levels, and the crossed surface holding the point.
  double worstDepth = 0., crossedNd = 0.;
  int worstKind = 0, worstLevel = -1, crossedLevel = -1;
  bool worstFromLeft = true, crossedNormal = false;
  for (int pass = 0; pass < 2; ++pass) {
    auto const &path = pass == 0 ? from : to;
    const int top    = pass == 0 ? fromLevel : toLevel;
    for (int k = 0; k < top - common; ++k) {
      // left levels from the deepest up, entered levels from the outermost down
      const int level = pass == 0 ? top - k : common + 1 + k;
      vecgeom::Transformation3D m; // TopMatrix fills an identity transformation
      path.TopMatrix(level, m);
      auto const *solid  = path.At(level)->GetLogicalVolume()->GetUnplacedVolume();
      const auto local   = m.Transform(point);
      const double sd    = SignedDistance(solid, local); // > 0 inside
      const double wrong = pass == 0 ? sd : -sd;         // inside a left volume, outside an entered one
      if (wrong > worstDepth) {
        worstDepth    = wrong;
        worstKind     = pass == 0 ? 1 : 2;
        worstLevel    = level;
        worstFromLeft = pass == 0;
      }
      if (crossedLevel < 0 && vecgeom::Abs(sd) <= vecgeom::kTolerance) {
        crossedLevel = level;
        vecgeom::Vector3D<double> normal;
        crossedNormal = solid->Normal(local, normal);
        if (crossedNormal) crossedNd = (pass == 0 ? 1. : -1.) * normal.Dot(m.TransformDirection(direction));
      }
    }
  }
  atomicAdd(&data->depthHistogram[DepthBin(worstDepth)], 1ull);
  atomicAdd(&data->depthLogHistogram[DepthLogBin(worstDepth)], 1ull);

  if (crossedLevel >= 0) {
    if (!crossedNormal)
      atomicAdd(&data->noNormal, 1ull);
    else if (crossedNd < 0.) {
      atomicAdd(&data->bounceBack, 1ull);
      atomicAdd(&data->bounceHistogram[vecgeom::Min(int(-crossedNd * kBounceBins), kBounceBins - 1)], 1ull);
    } else if (crossedNd == 0.)
      atomicAdd(&data->tangent, 1ull);
  }

  int kind = 0;
  if (worstDepth > vecgeom::kTolerance) {
    kind = worstKind;
    atomicAdd(kind == 1 ? &data->offBandLeft : &data->offBandEntered, 1ull);
  } else if (crossedLevel < 0) {
    kind = 3;
    atomicAdd(&data->offSurface, 1ull);
  }
  if (kind != 0) {
    atomicAdd(&data->offBand, 1ull);
    if (step.zeroStepCounter > kCrossingStuckPushSteps) atomicAdd(&data->offBandStuckPush, 1ull);
  } else if (step.zeroStepCounter >= 2 && step.zeroStepCounter <= kCrossingStuckPushSteps) {
    // In band, but the track keeps crossing without moving: record the states the loop alternates between.
    kind = 4;
    atomicAdd(&data->zeroStepLoop, 1ull);
  } else {
    return;
  }

  // Proof of an overlap at the end point or at witness points before (the side the track came from) or after it along
  // the final direction.
  OverlapProof proof;
  for (double witness : {0., -2., 2., -10., 10., -100., 100.}) {
    if (kind == 4) break;
    if (ProveOverlapAt(from, to, common, point + witness * vecgeom::kTolerance * direction, proof)) {
      proof.witness = witness * vecgeom::kTolerance;
      atomicAdd(&data->offBandOverlap, 1ull);
      break;
    }
  }

  const unsigned int slot = atomicAdd(&data->nRecords, 1u);
  if (slot >= data->recordCapacity) return;
  // Replay data in the frame of the offending volume: the worst wrong-side level, or the deepest crossed level.
  // No wrong-side volume: use the deepest left level, else the outermost entered one, else (a relocation back into
  // the same volume, nothing crossed) the volume itself.
  const bool crossedSide = kind >= 3;
  const bool useTo       = crossedSide ? fromLevel <= common && toLevel > common : !worstFromLeft;
  auto const &path       = useTo ? to : from;
  const int level        = crossedSide ? (useTo ? common + 1 : fromLevel) : worstLevel;
  vecgeom::Transformation3D m;
  path.TopMatrix(level, m);
  auto &record        = data->records[slot];
  const auto local    = m.Transform(point);
  const auto localDir = m.TransformDirection(direction);
  record.from         = from;
  record.to           = to;
  for (int i = 0; i < 3; ++i) {
    record.prePoint[i]       = step.prePoint[i];
    record.preDirection[i]   = step.preDirection[i];
    record.point[i]          = point[i];
    record.direction[i]      = direction[i];
    record.localPoint[i]     = local[i];
    record.localDirection[i] = localDir[i];
  }
  record.depth           = crossedSide
                               ? vecgeom::Abs(SignedDistance(path.At(level)->GetLogicalVolume()->GetUnplacedVolume(), local))
                               : worstDepth;
  record.outwardNDotD    = crossedLevel >= 0 && crossedNormal ? crossedNd : 0.;
  record.kind            = kind;
  record.offendingLevel  = level;
  record.offendingInFrom = &path == &from;
  record.trackId         = step.trackId;
  record.eventId         = step.eventId;
  record.stepCounter     = step.stepCounter;
  record.zeroStepCounter = step.zeroStepCounter;
  record.charge          = step.charge;
  record.magneticField   = step.charge != 0 && gMagneticField != nullptr;
  record.overlapType     = proof.type;
  record.overlapVolumeA  = proof.volumeA;
  record.overlapVolumeB  = proof.volumeB;
  record.overlapDepth    = proof.depth;
  record.overlapWitness  = proof.witness;
  record.physicsStep     = -1.;
  record.geometryStep    = -1.;
  record.preSafety       = -1.;
  record.displacement    = -1.;
}

/// @brief Record a state mismatch: a point lying beyond kStateMismatchDepth outside the volume of its navigation state,
/// or as deep inside one of its daughters, at the end of a step without relocation (kind 5) or at the start of a step
/// (kind 6). A daughter mismatch is recorded with the located deeper path as `to` (offending volume in `to`).
static __device__ __noinline__ void ValidateState(vecgeom::NavigationState const &state,
                                                  vecgeom::Vector3D<double> const &point,
                                                  vecgeom::Vector3D<double> const &direction,
                                                  crossing_detail::StepContext const &step, int kind,
                                                  double physicsStep, double geometryStep, double preSafety)
{
  using namespace crossing_detail;
  if (state.IsOutside() || state.Top() == nullptr) return;
  const int level = state.GetLevel();
  vecgeom::Transformation3D m;
  state.TopMatrix(level, m);
  const auto local = m.Transform(point);
  const double sd  = SignedDistance(state.Top()->GetLogicalVolume()->GetUnplacedVolume(), local);
  vecgeom::NavigationState located;
  state.CopyTo(&located);
  double depth    = -sd;
  int deeperLevel = -1;
  if (sd >= -kStateMismatchDepth) {
    // In the state volume: is the point already deep inside one of its daughters?
    vecgeom::BVHNavigator::LocatePointInNavState(local, located, false, nullptr);
    if (located.GetLevel() <= level) return;
    deeperLevel = level + 1;
    vecgeom::Transformation3D md;
    located.TopMatrix(deeperLevel, md);
    depth = SignedDistance(located.At(deeperLevel)->GetLogicalVolume()->GetUnplacedVolume(), md.Transform(point));
    if (depth <= kStateMismatchDepth) return;
  }
  auto *data = gCrossingValidation;
  atomicAdd(kind == 5 ? &data->stateMismatchEnd : &data->stateMismatchStart, 1ull);
  const unsigned int slot = atomicAdd(&data->nRecords, 1u);
  if (slot >= data->recordCapacity) return;
  auto &record = data->records[slot];
  vecgeom::Transformation3D mo; // frame of the offending volume
  located.TopMatrix(deeperLevel >= 0 ? deeperLevel : level, mo);
  const auto localDir = mo.TransformDirection(direction);
  const auto offLocal = mo.Transform(point);
  record.from         = state;
  record.to           = located;
  // Displacement beyond the straight geometry step (MSC), for a step without field
  const auto straight = step.prePoint + geometryStep * step.preDirection;
  for (int i = 0; i < 3; ++i) {
    record.prePoint[i]       = step.prePoint[i];
    record.preDirection[i]   = step.preDirection[i];
    record.point[i]          = point[i];
    record.direction[i]      = direction[i];
    record.localPoint[i]     = offLocal[i];
    record.localDirection[i] = localDir[i];
  }
  record.depth           = depth;
  record.outwardNDotD    = 0.;
  record.kind            = kind;
  record.offendingLevel  = deeperLevel >= 0 ? deeperLevel : level;
  record.offendingInFrom = deeperLevel < 0;
  record.trackId         = step.trackId;
  record.eventId         = step.eventId;
  record.stepCounter     = step.stepCounter;
  record.zeroStepCounter = step.zeroStepCounter;
  record.charge          = step.charge;
  record.magneticField   = step.charge != 0 && gMagneticField != nullptr;
  record.overlapType     = 0;
  record.overlapVolumeA  = -1;
  record.overlapVolumeB  = -1;
  record.overlapDepth    = 0.;
  record.overlapWitness  = 0.;
  record.physicsStep     = physicsStep;
  record.geometryStep    = geometryStep;
  record.preSafety       = preSafety;
  record.displacement    = kind == 5 ? (point - straight).Mag() : -1.;
}

/// @brief Kernel hook (diagnostic): check that the point lies in the volume of its navigation state.
template <typename Track, typename Vector>
__device__ inline void ValidateStateIfEnabled(Track const &track, short charge, vecgeom::NavigationState const &state,
                                              Vector const &prePos, Vector const &preDir, Vector const &pos,
                                              Vector const &dir, int kind, double physicsStep, double geometryStep,
                                              double preSafety)
{
  if (gCrossingValidation == nullptr) return;
  const crossing_detail::StepContext step{vecgeom::Vector3D<double>(prePos[0], prePos[1], prePos[2]),
                                          vecgeom::Vector3D<double>(preDir[0], preDir[1], preDir[2]),
                                          static_cast<unsigned long>(track.trackId),
                                          track.eventId,
                                          track.stepCounter,
                                          track.zeroStepCounter,
                                          charge};
  ValidateState(state, vecgeom::Vector3D<double>(pos[0], pos[1], pos[2]),
                vecgeom::Vector3D<double>(dir[0], dir[1], dir[2]), step, kind, physicsStep, geometryStep, preSafety);
}

/// @brief Kernel hook: validate the relocation just done, if the validation is on (one constant load otherwise).
/// @param track Track after the step (ids and step counters), `charge` its charge.
template <typename Track, typename Vector>
__device__ inline void ValidateCrossingIfEnabled(Track const &track, short charge, vecgeom::NavigationState const &from,
                                                 vecgeom::NavigationState const &to, Vector const &prePos,
                                                 Vector const &preDir, Vector const &pos, Vector const &dir)
{
  if (gCrossingValidation == nullptr) return;
  const crossing_detail::StepContext step{vecgeom::Vector3D<double>(prePos[0], prePos[1], prePos[2]),
                                          vecgeom::Vector3D<double>(preDir[0], preDir[1], preDir[2]),
                                          static_cast<unsigned long>(track.trackId),
                                          track.eventId,
                                          track.stepCounter,
                                          track.zeroStepCounter,
                                          charge};
  ValidateCrossing(from, to, vecgeom::Vector3D<double>(pos[0], pos[1], pos[2]),
                   vecgeom::Vector3D<double>(dir[0], dir[1], dir[2]), step,
                   gVolAuxData[to.GetLogicalId()].fGPUregionId < 0);
}

} // namespace adept::transport
