// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file SurfNavigator.h
 * @brief Navigation methods using the surface model.
 */

#ifndef RT_SURF_NAVIGATOR_H_
#define RT_SURF_NAVIGATOR_H_

#include <AdePT/copcore/Global.h>

#include <VecGeom/base/Global.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/surfaces/BVHSurfNavigator.h>

#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

inline namespace COPCORE_IMPL {

template <typename Real_t>
class SurfNavigator {

public:
  using Precision = vecgeom::Precision;
  using Vector3D  = vecgeom::Vector3D<vecgeom::Precision>;
  using SurfData  = vgbrep::SurfData<Real_t>;
  using Real_b    = typename SurfData::Real_b;

  static constexpr Precision kBoundaryPush = 10 * vecgeom::kTolerance;

  /// @brief Locates the point in the geometry volume tree
  /// @param pvol_id Placed volume id to be checked first
  /// @param point Point to be checked, in the local frame of pvol
  /// @param path Path to a parent of pvol that must contain the point
  /// @param top Check first if pvol contains the point
  /// @param exclude Placed volume id to exclude from the search
  /// @return Index of the placed volume that contains the point
  __host__ __device__ static int LocatePointIn(int pvol_id, Vector3D const &point, vecgeom::NavigationState &path,
                                               bool top, int *exclude = nullptr)
  {
    return vgbrep::protonav::BVHSurfNavigator<Real_t>::LocatePointIn(pvol_id, point, path, top, exclude);
  }

  /// @brief Computes the isotropic safety from the globalpoint.
  /// @param globalpoint Point in global coordinates
  /// @param state Path where to compute safety
  /// @return Isotropic safe distance
  __host__ __device__ static Precision ComputeSafety(Vector3D const &globalpoint, vecgeom::NavigationState const &state)
  {
    auto safety = vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeSafety(globalpoint, state);
    return safety;
  }

  // Computes a step from the globalpoint (which must be in the current volume)
  // into globaldir, taking step_limit into account. If a volume is hit, the
  // function calls out_state.SetBoundaryState(true) and relocates the state to
  // the next volume.
  //
  // The surface model does automatic relocation, so this function does it as well.
  __host__ __device__ static Precision ComputeStepAndNextVolume(Vector3D const &globalpoint, Vector3D const &globaldir,
                                                                Precision step_limit,
                                                                vecgeom::NavigationState const &in_state,
                                                                vecgeom::NavigationState &out_state,
                                                                long &hitsurf_index, Precision push = 0)
  {
    if (step_limit <= 0) {
      in_state.CopyTo(&out_state);
      out_state.SetBoundaryState(false);
      return step_limit;
    }
    auto step = vgbrep::protonav::BVHSurfNavigator<Real_t>::ComputeStepAndNextSurface(
        globalpoint, globaldir, in_state, out_state, hitsurf_index, step_limit);
    return step;
  }

  // Computes a step from the globalpoint (which must be in the current volume)
  // into globaldir, taking step_limit into account. If a volume is hit, the
  // function calls out_state.SetBoundaryState(true) and relocates the state to
  // the next volume.

  __host__ __device__ static Precision ComputeStepAndPropagatedState(Vector3D const &globalpoint,
                                                                     Vector3D const &globaldir, Precision step_limit,
                                                                     vecgeom::NavigationState const &in_state,
                                                                     vecgeom::NavigationState &out_state,
                                                                     long &hitsurf_index, Precision push = 0)
  {
    return ComputeStepAndNextVolume(globalpoint, globaldir, step_limit, in_state, out_state, hitsurf_index, push);
  }

  // Relocate a state that was returned from ComputeStepAndNextVolume: the surface
  // model does this computation within ComputeStepAndNextVolume, so the relocation does nothing
  __host__ __device__ static void RelocateToNextVolume(Vector3D const &globalpoint, Vector3D const &globaldir,
                                                       long hitsurf_index, vecgeom::NavigationState &out_state)
  {
    vgbrep::CrossedSurface crossed_surf;
    vgbrep::protonav::BVHSurfNavigator<Real_t>::RelocateToNextVolume(globalpoint, globaldir, Precision(0),
                                                                     hitsurf_index, out_state, crossed_surf);
  }
};

} // End namespace COPCORE_IMPL
#endif // RT_SURF_NAVIGATOR_H_
