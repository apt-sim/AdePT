// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file LoopNavigator.h
 * @brief Navigation methods for geometry.
 */

#ifndef RT_LOOP_NAVIGATOR_H_
#define RT_LOOP_NAVIGATOR_H_

#include <list>
#include <VecGeom/base/Global.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavStateIndex.h>

#include <CopCore/Global.h>

#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

inline namespace COPCORE_IMPL {

class LoopNavigator {

public:
  using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const *;

  VECCORE_ATT_HOST_DEVICE
  static VPlacedVolumePtr_t LocateGlobalPoint(vecgeom::VPlacedVolume const *vol,
                                              vecgeom::Vector3D<vecgeom::Precision> const &point,
                                              vecgeom::NavStateIndex &path, bool top)
  {
    vecgeom::VPlacedVolume const *candvolume = vol;
    vecgeom::Vector3D<vecgeom::Precision> currentpoint(point);
    if (top) {
      assert(vol != nullptr);
      if (!vol->UnplacedContains(point)) return nullptr;
    }
    path.Push(candvolume);
    vecgeom::LogicalVolume const *lvol                  = candvolume->GetLogicalVolume();
    vecgeom::Vector<vecgeom::Daughter> const *daughters = lvol->GetDaughtersp();

    bool godeeper = true;
    while (daughters->size() > 0 && godeeper) {
      godeeper = false;
      for (size_t i = 0; i < daughters->size(); ++i) {
        vecgeom::VPlacedVolume const *nextvolume = (*daughters)[i];
        vecgeom::Vector3D<vecgeom::Precision> transformedpoint;
        if (nextvolume->Contains(currentpoint, transformedpoint)) {
          path.Push(nextvolume);
          currentpoint = transformedpoint;
          candvolume   = nextvolume;
          daughters    = candvolume->GetLogicalVolume()->GetDaughtersp();
          godeeper     = true;
          break;
        }
      }
    }
    return candvolume;
  }

  VECCORE_ATT_HOST_DEVICE
  static VPlacedVolumePtr_t LocateGlobalPointExclVolume(vecgeom::VPlacedVolume const *vol,
                                                        vecgeom::VPlacedVolume const *excludedvolume,
                                                        vecgeom::Vector3D<vecgeom::Precision> const &point,
                                                        vecgeom::NavStateIndex &path, bool top)
  {
    vecgeom::VPlacedVolume const *candvolume = vol;
    vecgeom::Vector3D<vecgeom::Precision> currentpoint(point);
    if (top) {
      assert(vol != nullptr);
      candvolume = (vol->UnplacedContains(point)) ? vol : nullptr;
    }
    if (candvolume) {
      path.Push(candvolume);
      vecgeom::LogicalVolume const *lvol                  = candvolume->GetLogicalVolume();
      vecgeom::Vector<vecgeom::Daughter> const *daughters = lvol->GetDaughtersp();

      bool godeeper = true;
      while (daughters->size() > 0 && godeeper) {
        godeeper = false;
        // returns nextvolume; and transformedpoint; modified path
        for (size_t i = 0; i < daughters->size(); ++i) {
          vecgeom::VPlacedVolume const *nextvolume = (*daughters)[i];
          if (nextvolume != excludedvolume) {
            vecgeom::Vector3D<vecgeom::Precision> transformedpoint;
            if (nextvolume->Contains(currentpoint, transformedpoint)) {
              path.Push(nextvolume);
              currentpoint = transformedpoint;
              candvolume   = nextvolume;
              daughters    = candvolume->GetLogicalVolume()->GetDaughtersp();
              godeeper     = true;
              break;
            }
          } // end if excludedvolume
        }
      }
    }
    return candvolume;
  }

  VECCORE_ATT_HOST_DEVICE
  static VPlacedVolumePtr_t RelocatePointFromPathForceDifferent(vecgeom::Vector3D<vecgeom::Precision> const &localpoint,
                                                                vecgeom::NavStateIndex &path)
  {
    vecgeom::VPlacedVolume const *currentmother = path.Top();
    vecgeom::VPlacedVolume const *entryvol      = currentmother;
    vecgeom::Vector3D<vecgeom::Precision> transformed = localpoint;
    do {
      path.Pop();
      transformed   = currentmother->GetTransformation()->InverseTransform(transformed);
      currentmother = path.Top();
    } while (currentmother && (currentmother->IsAssembly() || !currentmother->UnplacedContains(transformed)));

    if (currentmother) {
      path.Pop();
      return LocateGlobalPointExclVolume(currentmother, entryvol, transformed, path, false);
    }
    return currentmother;
  }

  VECCORE_ATT_HOST_DEVICE
  static double ComputeStepAndPropagatedState(vecgeom::Vector3D<vecgeom::Precision> const &globalpoint,
                                              vecgeom::Vector3D<vecgeom::Precision> const &globaldir,
                                              vecgeom::Precision step_limit, vecgeom::NavStateIndex const &in_state,
                                              vecgeom::NavStateIndex &out_state)
  {
    // calculate local point/dir from global point/dir
    // call the static function for this provided/specialized by the Impl
    vecgeom::Vector3D<vecgeom::Precision> localpoint;
    vecgeom::Vector3D<vecgeom::Precision> localdir;
    // Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);
    vecgeom::Transformation3D m;
    in_state.TopMatrix(m);
    localpoint = m.Transform(globalpoint);
    localdir   = m.TransformDirection(globaldir);

    vecgeom::Precision step                    = step_limit;
    vecgeom::VPlacedVolume const *hitcandidate = nullptr;
    auto pvol                                  = in_state.Top();
    auto lvol                                  = pvol->GetLogicalVolume();

    // need to calc DistanceToOut first
    // step = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
    step = lvol->GetUnplacedVolume()->DistanceToOut(localpoint, localdir, step_limit);
    // step = pvol->DistanceToOut(localpoint, localdir, step_limit);

    if (step < 0) step = 0;
    // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
    //((Impl *)this)
    //    ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, &out_state, step, hitcandidate);
    auto *daughters = lvol->GetDaughtersp();
    auto ndaughters = daughters->size();
    for (decltype(ndaughters) d = 0; d < ndaughters; ++d) {
      auto daughter    = daughters->operator[](d);
      double ddistance = daughter->DistanceToIn(localpoint, localdir, step);

      // if distance is negative; we are inside that daughter and should relocate
      // unless distance is minus infinity
      const bool valid = (ddistance < step && !vecgeom::IsInf(ddistance)) &&
                         !((ddistance <= 0.) && in_state.GetLastExited() == daughter);
      hitcandidate = valid ? daughter : hitcandidate;
      step         = valid ? ddistance : step;
    }

    // fix state
    bool done;
    // step = Impl::PrepareOutState(in_state, out_state, step, step_limit, hitcandidate, done);
    // now we have the candidates and we prepare the out_state
    in_state.CopyTo(&out_state);
    done = false;
    if (step == vecgeom::kInfLength && step_limit > 0.) {
      step = vecgeom::kTolerance;
      out_state.SetBoundaryState(true);
      do {
        out_state.Pop();
      } while (out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
      done = true;
    } else {
      // is geometry further away than physics step?
      // this is a physics step
      if (step > step_limit) {
        // don't need to do anything
        step = step_limit;
        out_state.SetBoundaryState(false);
      } else {
        // otherwise it is a geometry step
        out_state.SetBoundaryState(true);
        if (hitcandidate) out_state.Push(hitcandidate);

        if (step < 0.) {
          // std::cerr << "WARNING: STEP NEGATIVE; NEXTVOLUME " << nexthitvolume << std::endl;
          // InspectEnvironmentForPointAndDirection( globalpoint, globaldir, currentstate );
          step = 0.;
        }
      }
    }

    if (done) {
      if (out_state.Top() != nullptr) {
        assert(!out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
      }
      return step;
    }
    // step was physics limited
    if (!out_state.IsOnBoundary()) return step;

    // otherwise if necessary do a relocation
    // try relocation to refine out_state to correct location after the boundary

    // ((Impl *)this)->Impl::Relocate(MovePointAfterBoundary(localpoint, localdir, step), in_state, out_state);
    localpoint += (step + 1.E-6) * localdir;

    if (out_state.Top() == in_state.Top()) {
      RelocatePointFromPathForceDifferent(localpoint, out_state);
    } else {
      // continue directly further down ( next volume should have been stored in out_state already )
      vecgeom::VPlacedVolume const *nextvol = out_state.Top();
      out_state.Pop();
      LoopNavigator::LocateGlobalPoint(nextvol, nextvol->GetTransformation()->Transform(localpoint), out_state, false);
    }

    if (out_state.Top() != nullptr) {
      while (out_state.Top()->IsAssembly()) {
        out_state.Pop();
      }
      assert(!out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
    }
    return step;
  }
};

} // End namespace COPCORE_IMPL
#endif // RT_LOOP_NAVIGATOR_H_
