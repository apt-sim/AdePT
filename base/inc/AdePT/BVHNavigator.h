// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file BVHNavigator.h
 * @brief Navigation methods for geometry.
 */

#ifndef RT_NAVIGATOR_H_
#define RT_NAVIGATOR_H_

#include <CopCore/Global.h>

#include <VecGeom/base/Global.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/navigation/NavStateIndex.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/management/GeoManager.h>

#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

inline namespace COPCORE_IMPL {

class BVHNavigator {

public:
  using Precision          = vecgeom::Precision;
  using Vector3D           = vecgeom::Vector3D<vecgeom::Precision>;
  using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const *;

  static constexpr Precision kBoundaryPush = 10 * vecgeom::kTolerance;

  /*
  * @param[in] aLVIndex Global index of a LogicalVolume
  * @param[in] index Index within the list of daughters of the specified LogicalVolume
  * @returns The PlacedVolume defined by @p aLVIndex and @p index
  */
  __device__ static inline vecgeom::Daughter GetPlacedVolume(int aLVIndex, int index)
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    return vecgeom::globaldevicegeomdata::gDeviceLogicalVolumes[aLVIndex].GetDaughters()[index];
#else
    return vecgeom::GeoManager::Instance().GetLogicalVolume(aLVIndex)->GetDaughters()[index];
#endif
  }

  /*
  * @param[in] global_index Global index of a PlacedVolume
  * @returns The PlacedVolume with global index @p global_index
  */
  __host__ __device__ static inline vecgeom::VPlacedVolume *GetPlacedVolume(int global_index)
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    return &vecgeom::globaldevicegeomdata::gCompactPlacedVolBuffer[global_index];
#else
    return vecgeom::GeoManager::Instance().GetPlacedVolume(global_index);
#endif
  }

  /*
  * @param[in] aLVIndex Global index of a LogicalVolume
  * @param[in] index Index within the list of daughters of the specified LogicalVolume
  * @param[in] localpoint Point in the local coordinates of the LV specified by @aLVIndex
  * @returns The safety to in to the PlacedVolume defined by @p aLVIndex and @p index for the point @p localpoint
  */
  __device__ static Precision CandidateSafetyToIn(int aLVIndex, int index, Vector3D localpoint)
  {
    return GetPlacedVolume(aLVIndex, index)->SafetyToIn(localpoint);
  };

  /*
  * @param[in] aLVIndex Global index of a LogicalVolume
  * @param[in] index Index within the list of daughters of the specified LogicalVolume
  * @param[in] localpoint Point in the local coordinates of the LV specified by @aLVIndex
  * @param[in] localdir Direction in the local coordinates of the LV specified by @aLVIndex
  * @param[in] step Maximum step length
  * @returns The distance to in to the PlacedVolume defined by @p aLVIndex and @p index for the point @p localpoint 
  * and direction @p localdir
  */
  __device__ static Precision CandidateDistanceToIn(int aLVIndex, int index, Vector3D localpoint, Vector3D localdir,
                                                    Precision step)
  {
    vecgeom::Daughter vol = GetPlacedVolume(aLVIndex, index);
    return vol->DistanceToIn(localpoint, localdir, step);
  };

  /*
  * @param[in] aLVIndex Global index of a LogicalVolume
  * @param[in] index Index within the list of daughters of the specified LogicalVolume
  * @param[in] localpoint Point in the local coordinates of the LV specified by @aLVIndex
  * @param[out] daughterlocalpoint Point in the local coordinates of the PlacedVolume defined by 
  * @p aLVIndex and @p index
  * @returns Whether @localpoint falls within the PlacedVolume defined by @p aLVIndex and @p index
  */
  __device__ static bool CandidateContains(int aLVIndex, int index, Vector3D const &localpoint,
                                           Vector3D &daughterlocalpoint)
  {
    return GetPlacedVolume(aLVIndex, index)->Contains(localpoint, daughterlocalpoint);
  };

  /*
  * @param[in] aLVIndex Global index of a LogicalVolume
  * @param[in] index Index within the list of daughters of the specified LogicalVolume
  * @param[in] localpoint Point in the local coordinates of the LV specified by @aLVIndex
  * @param[in] localdir Direction in the local coordinates of the LV specified by @aLVIndex
  * @returns The distance to in to the Bounding Box of the PlacedVolume defined by @p aLVIndex 
  * and @p index for the point @p localpoint and direction @p localdir
  */
  __device__ static Precision CandidateApproachSolid(int aLVIndex, int index, Vector3D localpoint, Vector3D localdir)
  {
    auto vol                            = GetPlacedVolume(aLVIndex, index);
    vecgeom::Transformation3D const *tr = vol->GetTransformation();
    Vector3D pv_localpoint              = tr->Transform(localpoint);
    Vector3D pv_localdir                = tr->TransformDirection(localdir);
    Vector3D pv_invlocaldir(1.0 / vecgeom::NonZero(localdir[0]), 1.0 / vecgeom::NonZero(localdir[1]),
                            1.0 / vecgeom::NonZero(localdir[2]));
    return vol->GetUnplacedVolume()->ApproachSolid(pv_localpoint, pv_invlocaldir);
  };

  /*
  * Used by the BVH to determine if it needs to skip checking a placed volume. The global index of the volume 
  * defined by @p aLVIndex and @p index can only be accessed from the navigator
  * @param[in] aLVIndex Global index of a LogicalVolume
  * @param[in] index Index within the list of daughters of the specified LogicalVolume
  * @param[in] global_id Global id of a PLacedVolume
  * @returns Whether the global id of the PlacedVolume defined by @p aLVIndex and @p index is the same as @p global_id
  */
  __device__ static bool SkipItem(int aLVIndex, int index, long const global_id)
  {
    return (global_id == GetPlacedVolume(aLVIndex, index)->id());
  }

  /*
  * @param[in] aLVIndex Global index of a LogicalVolume
  * @param[in] index Index within the list of daughters of the specified LogicalVolume
  * @returns The global id of the PlacedVolume defined by @p aLVIndex and @p index
  */
  __device__ static uint ItemId(int aLVIndex, int index) { return GetPlacedVolume(aLVIndex, index)->id(); }

  __host__ __device__ static VPlacedVolumePtr_t LocatePointIn(vecgeom::VPlacedVolume const *vol, Vector3D const &point,
                                                              vecgeom::NavStateIndex &path, bool top,
                                                              vecgeom::VPlacedVolume const *exclude = nullptr)
  {
    if (top) {
      assert(vol != nullptr);
      if (!vol->UnplacedContains(point)) return nullptr;
    }

    path.Push(vol);

    Vector3D currentpoint(point);
    Vector3D daughterlocalpoint;
    long exclude_id = -1;
    long vol_id     = -1;

    for (auto v = vol; v->GetDaughters().size() > 0;) {
      auto bvh = vecgeom::BVHManager::GetBVH(v->GetLogicalVolume()->id());

      exclude_id = -1;
      if (exclude != nullptr) {
        exclude_id = exclude->id();
      }
      vol_id = -1;

      if (!bvh->LevelLocate<BVHNavigator>(exclude_id, currentpoint, vol_id, daughterlocalpoint)) break;

      currentpoint = daughterlocalpoint;
      // Update the current volume v
      v = GetPlacedVolume(vol_id);
      path.Push(v);
      // Only exclude the placed volume once since we could enter it again via a
      // different volume history.
      exclude = nullptr;
    }

    return path.Top();
  }

  __host__ __device__ static VPlacedVolumePtr_t RelocatePoint(Vector3D const &localpoint, vecgeom::NavStateIndex &path)
  {
    vecgeom::VPlacedVolume const *currentmother = path.Top();
    Vector3D transformed                        = localpoint;
    do {
      path.Pop();
      transformed   = currentmother->GetTransformation()->InverseTransform(transformed);
      currentmother = path.Top();
    } while (currentmother && (currentmother->IsAssembly() || !currentmother->UnplacedContains(transformed)));

    if (currentmother) {
      path.Pop();
      return LocatePointIn(currentmother, transformed, path, false);
    }
    return currentmother;
  }

private:
  // Computes a step in the current volume from the localpoint into localdir,
  // taking step_limit into account. If a volume is hit, the function calls
  // out_state.SetBoundaryState(true) and hitcandidate is set to the hit
  // daughter volume, or kept unchanged if the current volume is left.
  __host__ __device__ static Precision ComputeStepAndHit(Vector3D const &localpoint, Vector3D const &localdir,
                                                         Precision step_limit, vecgeom::NavStateIndex const &in_state,
                                                         vecgeom::NavStateIndex &out_state,
                                                         VPlacedVolumePtr_t &hitcandidate)
  {
    if (step_limit <= 0) {
      // We don't need to ask any solid, this step is not limited by geometry.
      in_state.CopyTo(&out_state);
      out_state.SetBoundaryState(false);
      return 0;
    }

    Precision step                 = step_limit;
    VPlacedVolumePtr_t pvol        = in_state.Top();
    VPlacedVolumePtr_t last_exited = in_state.GetLastExited();
    long hitcandidate_index        = -1;
    long last_exited_id            = -1;

    // need to calc DistanceToOut first
    step = pvol->DistanceToOut(localpoint, localdir, step_limit);

    if (step < 0) step = 0;

    if (pvol->GetDaughters().size() > 0) {
      auto bvh = vecgeom::BVHManager::GetBVH(pvol->GetLogicalVolume()->id());

      hitcandidate_index = -1;
      // id is an uint, however we use a long in order to be able to fit the full uint range, and -1 in case there is no
      // last exited volume in the navigation state.
      last_exited_id = -1;
      //if (last_exited != nullptr) last_exited_id = last_exited->id();

      bvh->CheckDaughterIntersections<BVHNavigator>(localpoint, localdir, step, last_exited_id, hitcandidate_index);

      if (hitcandidate_index >= 0) hitcandidate = pvol->GetLogicalVolume()->GetDaughters()[hitcandidate_index];
    }

    // now we have the candidates and we prepare the out_state
    in_state.CopyTo(&out_state);
    if (step == vecgeom::kInfLength && step_limit > 0) {
      out_state.SetBoundaryState(true);
      do {
        out_state.Pop();
      } while (out_state.Top()->IsAssembly());

      return vecgeom::kTolerance;
    }

    // Is geometry further away than physics step?
    if (step > step_limit) {
      // Then this is a phyics step and we don't need to do anything.
      out_state.SetBoundaryState(false);
      return step_limit;
    }

    // Otherwise it is a geometry step and we push the point to the boundary.
    out_state.SetBoundaryState(true);

    if (step < 0) {
      step = 0;
    }

    return step;
  }

  // Computes a step in the current volume from the localpoint into localdir,
  // until the next daughter bounding box, taking step_limit into account.
  __host__ __device__ static Precision ApproachNextVolume(Vector3D const &localpoint, Vector3D const &localdir,
                                                          Precision step_limit, vecgeom::NavStateIndex const &in_state)
  {
    Precision step                 = step_limit;
    VPlacedVolumePtr_t pvol        = in_state.Top();
    VPlacedVolumePtr_t last_exited = in_state.GetLastExited();

    if (pvol->GetDaughters().size() > 0) {
      auto bvh = vecgeom::BVHManager::GetBVH(pvol->GetLogicalVolume()->id());

      // id is an uint, however we use a long in order to be able to fit the full uint range, and -1 in case there is no
      // last exited volume in the navigation state.
      long last_exited_id = -1;
      //if (last_exited != nullptr) last_exited_id = last_exited->id();

      bvh->ApproachNextDaughter<BVHNavigator>(localpoint, localdir, step, last_exited_id);
      // Make sure we don't "step" on next boundary
      step -= 10 * vecgeom::kTolerance;
    }

    if (step == vecgeom::kInfLength && step_limit > 0) return 0;

    // Is geometry further away than physics step?
    if (step > step_limit) {
      // Then this is a phyics step and we don't need to do anything.
      return step_limit;
    }

    if (step < 0) {
      step = 0;
    }

    return step;
  }

public:
  // Computes the isotropic safety from the globalpoint.
  __host__ __device__ static Precision ComputeSafety(Vector3D const &globalpoint, vecgeom::NavStateIndex const &state)
  {
    VPlacedVolumePtr_t pvol = state.Top();
    vecgeom::Transformation3D m;
    state.TopMatrix(m);
    Vector3D localpoint = m.Transform(globalpoint);

    // need to calc DistanceToOut first
    Precision safety = pvol->SafetyToOut(localpoint);

    if (safety > 0 && pvol->GetDaughters().size() > 0) {
      auto bvh = vecgeom::BVHManager::GetBVH(pvol->GetLogicalVolume()->id());
      safety   = bvh->ComputeSafety<BVHNavigator>(localpoint, safety);
    }

    return safety;
  }

  // Computes a step from the globalpoint (which must be in the current volume)
  // into globaldir, taking step_limit into account. If a volume is hit, the
  // function calls out_state.SetBoundaryState(true) and relocates the state to
  // the next volume.
  __host__ __device__ static Precision ComputeStepAndPropagatedState(Vector3D const &globalpoint,
                                                                     Vector3D const &globaldir, Precision step_limit,
                                                                     vecgeom::NavStateIndex const &in_state,
                                                                     vecgeom::NavStateIndex &out_state,
                                                                     Precision push = 0)
  {
    // If we are on the boundary, push a bit more.
    if (in_state.IsOnBoundary()) {
      push += kBoundaryPush;
    }
    if (step_limit < push) {
      // Go as far as the step limit says, assuming there is no boundary.
      // TODO: Does this make sense?
      in_state.CopyTo(&out_state);
      out_state.SetBoundaryState(false);
      return step_limit;
    }
    step_limit -= push;

    // calculate local point/dir from global point/dir
    Vector3D localpoint;
    Vector3D localdir;
    // Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);
    vecgeom::Transformation3D m;
    in_state.TopMatrix(m);
    localpoint = m.Transform(globalpoint);
    localdir   = m.TransformDirection(globaldir);
    // The user may want to move point from boundary before computing the step
    localpoint += push * localdir;

    VPlacedVolumePtr_t hitcandidate = nullptr;
    Precision step = ComputeStepAndHit(localpoint, localdir, step_limit, in_state, out_state, hitcandidate);
    step += push;

    if (out_state.IsOnBoundary()) {
      // Relocate the point after the step to refine out_state.
      localpoint += (step + kBoundaryPush) * localdir;

      if (!hitcandidate) {
        // We didn't hit a daughter but instead we're exiting the current volume.
        RelocatePoint(localpoint, out_state);
      } else {
        // Otherwise check if we're directly entering other daughters transitively.
        localpoint = hitcandidate->GetTransformation()->Transform(localpoint);
        LocatePointIn(hitcandidate, localpoint, out_state, false);
      }

      if (out_state.Top() != nullptr) {
        while (out_state.Top()->IsAssembly() || out_state.GetNavIndex() == in_state.GetNavIndex()) {
          out_state.Pop();
        }
        assert(!out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
      }
    }

    return step;
  }

  // Computes a step from the globalpoint (which must be in the current volume)
  // into globaldir, taking step_limit into account. If a volume is hit, the
  // function calls out_state.SetBoundaryState(true) and
  //  - removes all volumes from out_state if the current volume is left, or
  //  - adds the hit daughter volume to out_state if one is hit.
  // However the function does _NOT_ relocate the state to the next volume,
  // that is entering multiple volumes that share a boundary.
  __host__ __device__ static Precision ComputeStepAndNextVolume(Vector3D const &globalpoint, Vector3D const &globaldir,
                                                                Precision step_limit,
                                                                vecgeom::NavStateIndex const &in_state,
                                                                vecgeom::NavStateIndex &out_state, Precision push = 0)
  {
    // If we are on the boundary, push a bit more.
    if (in_state.IsOnBoundary()) {
      push += kBoundaryPush;
    }
    if (step_limit < push) {
      // Go as far as the step limit says, assuming there is no boundary.
      // TODO: Does this make sense?
      in_state.CopyTo(&out_state);
      out_state.SetBoundaryState(false);
      return step_limit;
    }
    step_limit -= push;

    // calculate local point/dir from global point/dir
    Vector3D localpoint;
    Vector3D localdir;
    // Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);
    vecgeom::Transformation3D m;
    in_state.TopMatrix(m);
    localpoint = m.Transform(globalpoint);
    localdir   = m.TransformDirection(globaldir);
    // The user may want to move point from boundary before computing the step
    localpoint += push * localdir;

    VPlacedVolumePtr_t hitcandidate = nullptr;
    Precision step = ComputeStepAndHit(localpoint, localdir, step_limit, in_state, out_state, hitcandidate);
    step += push;

    if (out_state.IsOnBoundary()) {
      if (!hitcandidate) {
        vecgeom::VPlacedVolume const *currentmother = out_state.Top();
        Vector3D transformed                        = localpoint;
        // Push the point inside the next volume.
        transformed += (step + kBoundaryPush) * localdir;

        do {
          out_state.SetLastExited();
          out_state.Pop();
          transformed   = currentmother->GetTransformation()->InverseTransform(transformed);
          currentmother = out_state.Top();
        } while (currentmother && (currentmother->IsAssembly() || !currentmother->UnplacedContains(transformed)));
      } else {
        out_state.Push(hitcandidate);
      }
    }

    return step;
  }

  // Computes a step from the globalpoint (which must be in the current volume)
  // into globaldir, taking step_limit into account.
  __host__ __device__ static Precision ComputeStepToApproachNextVolume(Vector3D const &globalpoint,
                                                                       Vector3D const &globaldir, Precision step_limit,
                                                                       vecgeom::NavStateIndex const &in_state)
  {
    // calculate local point/dir from global point/dir
    Vector3D localpoint;
    Vector3D localdir;
    // Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);
    vecgeom::Transformation3D m;
    in_state.TopMatrix(m);
    localpoint = m.Transform(globalpoint);
    localdir   = m.TransformDirection(globaldir);

    Precision step = ApproachNextVolume(localpoint, localdir, step_limit, in_state);

    return step;
  }

  // Relocate a state that was returned from ComputeStepAndNextVolume: It
  // recursively locates the pushed point in the containing volume.
  __host__ __device__ static void RelocateToNextVolume(Vector3D const &globalpoint, Vector3D const &globaldir,
                                                       vecgeom::NavStateIndex &state)
  {
    // Push the point inside the next volume.
    Vector3D pushed = globalpoint + kBoundaryPush * globaldir;

    // Calculate local point from global point.
    vecgeom::Transformation3D m;
    state.TopMatrix(m);
    Vector3D localpoint = m.Transform(pushed);

    VPlacedVolumePtr_t pvol = state.Top();

    state.Pop();
    LocatePointIn(pvol, localpoint, state, false, state.GetLastExited());

    if (state.Top() != nullptr) {
      while (state.Top()->IsAssembly()) {
        state.Pop();
      }
      assert(!state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
    }
  }
};

} // End namespace COPCORE_IMPL
#endif // RT_LOOP_NAVIGATOR_H_