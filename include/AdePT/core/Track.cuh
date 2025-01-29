// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRACK_CUH
#define ADEPT_TRACK_CUH

#include <AdePT/core/TrackData.h>
#include <AdePT/copcore/SystemOfUnits.h>
#include <AdePT/copcore/Ranluxpp.h>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>

#ifdef USE_SPLIT_KERNELS
#include <G4HepEmRandomEngine.hh>
#endif

// A data structure to represent a particle track. The particle type is implicit
// by the queue and not stored in memory.
struct Track {
  using Precision = vecgeom::Precision;

  RanluxppDouble rngState;
  double eKin{0.};
  float numIALeft[4]{-1.f, -1.f, -1.f, -1.f};
  float initialRange{-1.f};
  float dynamicRangeFactor{-1.f};
  float tlimitMin{-1.f};

  double globalTime{0.};
  float localTime{0.f};
  float properTime{0.f};

  vecgeom::Vector3D<Precision> pos;   ///< track position
  vecgeom::Vector3D<Precision> dir;   ///< track direction
  vecgeom::Vector3D<float> safetyPos; ///< last position where the safety was computed
  float safety{0.f};                  ///< last computed safety value
  vecgeom::NavigationState navState;  ///< current navigation state

#ifdef USE_SPLIT_KERNELS
  // Variables used to store track info needed for scoring
  vecgeom::NavigationState nextState;
  vecgeom::NavigationState preStepNavState;
  vecgeom::Vector3D<Precision> preStepPos;
  vecgeom::Vector3D<Precision> preStepDir;
  RanluxppDouble newRNG;
  double preStepEKin{0};

  // Variables used to store navigation results
  double geometryStepLength{0};
  long hitsurfID{0};
#endif

  unsigned int eventId{0};
  int parentId{-1}; // Stores the track id of the initial particle given to AdePT
  short threadId{-1};
  unsigned short stepCounter{0};
  unsigned short looperCounter{0};

#ifdef USE_SPLIT_KERNELS
  bool propagated{false};

  // Variables used to store results from G4HepEM
  bool restrictedPhysicalStepLength{false};
  bool stopped{false};
#endif

  /// Construct a new track for GPU transport.
  /// NB: The navState remains uninitialised.
  __device__ Track(uint64_t rngSeed, double eKin, double globalTime, float localTime, float properTime,
                   double const position[3], double const direction[3], unsigned int eventId, int parentId,
                   short threadId)
      : eKin{eKin}, globalTime{globalTime}, localTime{localTime}, properTime{properTime}, eventId{eventId},
        parentId{parentId}, threadId{threadId}
  {
    rngState.SetSeed(rngSeed);

    pos = {position[0], position[1], position[2]};
    dir = {direction[0], direction[1], direction[2]};
  }

  /// Construct a secondary from a parent track.
  /// NB: The caller is responsible to branch a new RNG state.
  __device__ Track(RanluxppDouble const &rngState, double eKin, const vecgeom::Vector3D<Precision> &parentPos,
                   const vecgeom::Vector3D<Precision> &newDirection, const vecgeom::NavigationState &newNavState,
                   const Track &parentTrack)
      : rngState{rngState}, eKin{eKin}, globalTime{parentTrack.globalTime}, pos{parentPos}, dir{newDirection},
        navState{newNavState}, eventId{parentTrack.eventId}, parentId{parentTrack.parentId},
        threadId{parentTrack.threadId}
  {
  }

  /// @brief Get recomputed cached safety ay a given track position
  /// @param new_pos Track position
  /// @param accurate_limit Only return non-zero if the recomputed safety if larger than the accurate_limit
  /// @return Recomputed safety.
  __host__ __device__ VECGEOM_FORCE_INLINE float GetSafety(vecgeom::Vector3D<Precision> const &new_pos,
                                                           float accurate_limit = 0.f) const
  {
    float dsafe = safety - accurate_limit;
    if (dsafe <= 0.f) return 0.f;
    float distSq = (vecgeom::Vector3D<float>(new_pos) - safetyPos).Mag2();
    if (dsafe * dsafe < distSq) return 0.f;
    return (safety - vecCore::math::Sqrt(distSq));
  }

  /// @brief Set Safety value computed in a new point
  /// @param new_pos Position where the safety is computed
  /// @param safe Safety value
  __host__ __device__ VECGEOM_FORCE_INLINE void SetSafety(vecgeom::Vector3D<Precision> const &new_pos, float safe)
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
    this->numIALeft[3] = -1.0;

    this->initialRange       = -1.0;
    this->dynamicRangeFactor = -1.0;
    this->tlimitMin          = -1.0;

    // A secondary inherits the position of its parent; the caller is responsible
    // to update the directions.
    this->pos = parentPos;
    this->safetyPos.Set(0.f, 0.f, 0.f);
    this->safety   = 0.0f;
    this->navState = parentNavState;

    // The global time is inherited from the parent
    this->globalTime = gTime;
    this->localTime  = 0.;
    this->properTime = 0.;
  }

  __host__ __device__ void CopyTo(adeptint::TrackData &tdata, int pdg)
  {
    tdata.pdg          = pdg;
    tdata.parentId     = parentId;
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
    // FIXME missing navState here
  }
};
#endif
