// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/// Device side of the crossing validation (see CrossingValidation.hh).

#pragma once

#include <AdePT/transport/geometry_validation/CrossingValidation.hh>
#include <AdePT/transport/state/DeviceGlobals.cuh>

#include <VecGeom/base/Transformation3D.h>
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
                                                     vecgeom::Vector3D<double> const &direction, bool leavesGPU)
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

  if (crossedLevel >= 0) {
    if (!crossedNormal)
      atomicAdd(&data->noNormal, 1ull);
    else if (crossedNd < 0.)
      atomicAdd(&data->bounceBack, 1ull);
    else if (crossedNd == 0.)
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
  if (kind == 0) return;
  atomicAdd(&data->offBand, 1ull);

  // Proof of an overlap at the end point or at witness points just beyond it along the final direction.
  OverlapProof proof;
  for (double witness : {0., 2., 10., 100.}) {
    if (ProveOverlapAt(from, to, common, point + witness * vecgeom::kTolerance * direction, proof)) {
      proof.witness = witness * vecgeom::kTolerance;
      atomicAdd(&data->offBandOverlap, 1ull);
      break;
    }
  }

  const unsigned int slot = atomicAdd(&data->nRecords, 1u);
  if (slot >= data->recordCapacity) return;
  // Replay data in the frame of the offending volume: the worst wrong-side level, or the deepest crossed level.
  auto const &path = kind == 3 ? (fromLevel > common ? from : to) : (worstFromLeft ? from : to);
  const int level  = kind == 3 ? (fromLevel > common ? fromLevel : common + 1) : worstLevel;
  vecgeom::Transformation3D m;
  path.TopMatrix(level, m);
  auto &record        = data->records[slot];
  const auto local    = m.Transform(point);
  const auto localDir = m.TransformDirection(direction);
  for (int i = 0; i < 3; ++i) {
    record.point[i]          = point[i];
    record.direction[i]      = direction[i];
    record.localPoint[i]     = local[i];
    record.localDirection[i] = localDir[i];
  }
  record.depth           = kind == 3
                               ? vecgeom::Abs(SignedDistance(path.At(level)->GetLogicalVolume()->GetUnplacedVolume(), local))
                               : worstDepth;
  record.outwardNDotD    = crossedLevel >= 0 && crossedNormal ? crossedNd : 0.;
  record.offendingVolume = path.At(level)->id();
  record.fromVolume      = from.Top()->id();
  record.toVolume        = to.Top()->id();
  record.kind            = kind;
  record.overlapType     = proof.type;
  record.overlapVolumeA  = proof.volumeA;
  record.overlapVolumeB  = proof.volumeB;
  record.overlapDepth    = proof.depth;
  record.overlapWitness  = proof.witness;
}

/// @brief Kernel hook: validate the relocation just done, if the validation is on (one constant load otherwise).
template <typename Vector>
__device__ inline void ValidateCrossingIfEnabled(vecgeom::NavigationState const &from,
                                                 vecgeom::NavigationState const &to, Vector const &pos,
                                                 Vector const &dir)
{
  if (gCrossingValidation == nullptr) return;
  ValidateCrossing(from, to, vecgeom::Vector3D<double>(pos[0], pos[1], pos[2]),
                   vecgeom::Vector3D<double>(dir[0], dir[1], dir[2]), gVolAuxData[to.GetLogicalId()].fGPUregionId < 0);
}

} // namespace adept::transport
