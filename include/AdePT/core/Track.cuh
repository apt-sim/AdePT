// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRACK_CUH
#define ADEPT_TRACK_CUH

#include <AdePT/core/TrackData.h>
#include <AdePT/copcore/SystemOfUnits.h>
#include <AdePT/copcore/Ranluxpp.h>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>

#ifdef ADEPT_USE_SPLIT_KERNELS
#include <G4HepEmRandomEngine.hh>
#endif

// A data structure to represent a particle track. The particle type is implicit
// by the queue and not stored in memory.
struct Track {
  RanluxppDouble rngState;
  double eKin{0.};
  double globalTime{0.};

  float weight{0.};
#ifndef ADEPT_USE_SPLIT_KERNELS
  float numIALeft[4]{-1.f, -1.f, -1.f, -1.f};
  // default values taken from G4HepEmMSCTrackData.hh
  float initialRange{1.0e+21};
  float dynamicRangeFactor{0.04};
  float tlimitMin{1.0E-7};
#endif

  float localTime{0.f};
  float properTime{0.f};

  vecgeom::Vector3D<double> pos;      ///< track position
  vecgeom::Vector3D<double> dir;      ///< track direction
  vecgeom::Vector3D<float> safetyPos; ///< last position where the safety was computed
  // TODO: For better clarity in the split kernels, rename this to "stored safety" as opposed to the
  // safety we get from GetSafety(), which is computed in the moment
  float safety{0.f};                 ///< last computed safety value
  vecgeom::NavigationState navState; ///< current navigation state

#ifdef ADEPT_USE_SPLIT_KERNELS
  // Variables used to store track info needed for scoring
  vecgeom::NavigationState nextState;
  vecgeom::Vector3D<double> preStepPos;
  vecgeom::Vector3D<double> preStepDir;
  double preStepEKin{0};
  double preStepGlobalTime{0.};
  // Variables used to store navigation results
  double safeLength{0};
  long hitsurfID{0};
#endif

  uint64_t trackId{0};  ///< track id (non-consecutive, reproducible)
  uint64_t parentId{0}; // track id of the parent

  unsigned int eventId{0};
  short threadId{-1};
  unsigned short stepCounter{0};
  unsigned short looperCounter{0};
  unsigned short zeroStepCounter{0};

#ifdef ADEPT_USE_SPLIT_KERNELS
  bool propagated{false};
  bool hepEmTrackExists{false};

  // Variables used to store results from G4HepEM
  bool restrictedPhysicalStepLength{false};
  bool stopped{false};
#endif

  LeakStatus leakStatus{LeakStatus::NoLeak};

  __host__ __device__ Track(const Track &)            = default;
  __host__ __device__ Track &operator=(const Track &) = default;

  /// Construct a new track for GPU transport.
  __device__ Track(uint64_t rngSeed, double eKin, double globalTime, float localTime, float properTime, float weight,
                   double const position[3], double const direction[3], const vecgeom::NavigationState &newNavState,
                   unsigned int eventId, uint64_t trackId, uint64_t parentId, short threadId,
                   unsigned short stepCounter)
      : eKin{eKin}, weight{weight}, globalTime{globalTime}, localTime{localTime}, properTime{properTime},
        navState{newNavState}, eventId{eventId}, trackId{trackId}, parentId{parentId}, threadId{threadId},
        stepCounter{stepCounter}, looperCounter{0}, zeroStepCounter{0}
  {
    rngState.SetSeed(rngSeed);
    pos        = {position[0], position[1], position[2]};
    dir        = {direction[0], direction[1], direction[2]};
    leakStatus = LeakStatus::NoLeak;
  }

  /// Construct a secondary from a parent track.
  /// NB: The caller is responsible to branch a new RNG state.
  __device__ Track(RanluxppDouble const &rng_state, double eKin, const vecgeom::Vector3D<double> &parentPos,
                   const vecgeom::Vector3D<double> &newDirection, const vecgeom::NavigationState &newNavState,
                   const Track &parentTrack, const double globalTime)
      : rngState{rng_state}, eKin{eKin}, globalTime{globalTime}, pos{parentPos}, dir{newDirection},
        navState{newNavState}, trackId{rngState.IntRndm64()}, eventId{parentTrack.eventId},
        parentId{parentTrack.trackId}, threadId{parentTrack.threadId}, weight{parentTrack.weight}, stepCounter{0},
        looperCounter{0}, zeroStepCounter{0}, leakStatus{LeakStatus::NoLeak}
  {
  }

  __host__ __device__ VECGEOM_FORCE_INLINE bool Matches(int ievt, size_t itrack, size_t stepmin = 0,
                                                        size_t stepmax = 100000) const
  {
    bool match_event = (ievt < 0) || (ievt == eventId);
    return match_event && (itrack == trackId) && (stepCounter >= stepmin) && (stepCounter <= stepmax);
  }

  __host__ __device__ void Print(const char *label) const
  {
    printf("== evt %u parentId %lu %s id %lu step %d ekin %g MeV | pos {%.19f, %.19f, %.19f} dir {%.19f, %.19f, "
           "%.19f} remain_safe %g loop %u\n| | state: ",
           eventId, parentId, label, trackId, stepCounter, eKin / copcore::units::MeV, pos[0], pos[1], pos[2], dir[0],
           dir[1], dir[2], GetSafety(pos), looperCounter);
    navState.Print();
  }

  /// @brief Get recomputed cached safety ay a given track position
  /// @param new_pos Track position
  /// @param accurate_limit Only return non-zero if the recomputed safety if larger than the accurate_limit
  /// @return Recomputed safety.
  __host__ __device__ VECGEOM_FORCE_INLINE float GetSafety(vecgeom::Vector3D<double> const &new_pos,
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
  __host__ __device__ VECGEOM_FORCE_INLINE void SetSafety(vecgeom::Vector3D<double> const &new_pos, float safe)
  {
    safetyPos.Set(static_cast<float>(new_pos[0]), static_cast<float>(new_pos[1]), static_cast<float>(new_pos[2]));
    safety = vecCore::math::Max(safe, 0.f);
  }

  __host__ __device__ double Uniform() { return rngState.Rndm(); }
};
#endif
