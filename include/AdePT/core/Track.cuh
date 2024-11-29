// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRACK_CUH
#define ADEPT_TRACK_CUH

#include <AdePT/core/TrackData.h>
#include <AdePT/copcore/SystemOfUnits.h>
#include <AdePT/copcore/Ranluxpp.h>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>

// A data structure to represent a particle track. The particle type is implicit
// by the queue and not stored in memory.
struct Track {
  using Precision = vecgeom::Precision;

  int parentID{0}; // Stores the track id of the initial particle given to AdePT

  RanluxppDouble rngState;
  double eKin;
  double numIALeft[3];
  double initialRange;
  double dynamicRangeFactor;
  double tlimitMin;

  double globalTime{0};
  double localTime{0};
  double properTime{0};

  vecgeom::Vector3D<Precision> pos;   ///< track position
  vecgeom::Vector3D<Precision> dir;   ///< track direction
  vecgeom::Vector3D<float> safetyPos; ///< last position where the safety was computed
  float safety{0};                    ///< last computed safety value
  vecgeom::NavigationState navState;  ///< current navigation state
  
  /// @brief Get recomputed cached safety ay a given track position
  /// @param new_pos Track position
  /// @param accurate_limit Only return non-zero if the recomputed safety if larger than the accurate_limit
  /// @return Recomputed safety.
  __host__ __device__ VECGEOM_FORCE_INLINE 
  float GetSafety(vecgeom::Vector3D<Precision> const &new_pos, float accurate_limit = 0) const
  {
    float dsafe = safety - accurate_limit;
    if (dsafe <= 0) return 0;
    float distSq = (vecgeom::Vector3D<float>(new_pos) - safetyPos).Mag2();
    if (dsafe * dsafe < distSq) return 0.;
    return (safety - vecCore::math::Sqrt(distSq));
  }

  __host__ __device__ VECGEOM_FORCE_INLINE

  /// @brief Set Safety value computed in a new point
  /// @param new_pos Position where the safety is computed
  /// @param safe Safety value
  void SetSafety(vecgeom::Vector3D<Precision> const &new_pos, float safe)
  {
    safetyPos.Set(static_cast<float>(new_pos[0]), static_cast<float>(new_pos[1]), static_cast<float>(new_pos[2]));
    safety = vecCore::math::Max(safe, 0.f);
  }

  __host__ __device__ double Uniform() { return rngState.Rndm(); }

  __host__ __device__ void InitAsSecondary(const vecgeom::Vector3D<Precision> &parentPos,
                                           const vecgeom::NavigationState &parentNavState, double gTime)
  {
    // The caller is responsible to branch a new RNG state and to set the energy.
    this->numIALeft[0] = -1.0;
    this->numIALeft[1] = -1.0;
    this->numIALeft[2] = -1.0;

    this->initialRange       = -1.0;
    this->dynamicRangeFactor = -1.0;
    this->tlimitMin          = -1.0;

    // A secondary inherits the position of its parent; the caller is responsible
    // to update the directions.
    this->pos      = parentPos;
    this->navState = parentNavState;

    // The global time is inherited from the parent
    this->globalTime = gTime;
  }

  __host__ __device__ void CopyTo(adeptint::TrackData &tdata, int pdg)
  {
    tdata.pdg          = pdg;
    tdata.parentID     = parentID;
    tdata.position[0]  = pos[0];
    tdata.position[1]  = pos[1];
    tdata.position[2]  = pos[2];
    tdata.direction[0] = dir[0];
    tdata.direction[1] = dir[1];
    tdata.direction[2] = dir[2];
    tdata.eKin         = eKin;
    tdata.globalTime   = globalTime;
    tdata.localTime    = localTime;
    tdata.properTime   = properTime;
  }
};
#endif
