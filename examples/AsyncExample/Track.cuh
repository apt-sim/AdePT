// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRACK_CUH
#define ADEPT_TRACK_CUH

#include <AdePT/base/MParray.h>
#include <AdePT/copcore/SystemOfUnits.h>
#include <AdePT/copcore/Ranluxpp.h>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>

// A data structure to represent a particle track. The particle type is implicit
// by the queue and not stored in memory.
struct Track {
  using Precision = vecgeom::Precision;
  RanluxppDouble rngState;
  double energy            = 0;
  float numIALeft[3]       = {-1., -1., -1.};
  float initialRange       = -1.f; // Only for e-?
  float dynamicRangeFactor = -1.f; // Only for e-?
  float tlimitMin          = -1.f; // Only for e-?

  double globalTime = 0.;
  float localTime   = 0.f; // Only for e-?
  float properTime  = 0.f; // Only for e-?

  vecgeom::Vector3D<Precision> pos;
  vecgeom::Vector3D<Precision> dir;
  vecgeom::NavigationState navState;
  unsigned int eventId{0};
  int parentId{-1};
  short threadId{-1};
  unsigned short stepCounter{0};
  unsigned short looperCounter{0};

  __host__ __device__ double Uniform() { return rngState.Rndm(); }

  /// Construct a new track for GPU transport.
  /// NB: The navState remains uninitialised.
  __device__ Track(uint64_t rngSeed, double eKin, double globalTime, float localTime, float properTime,
                   double const position[3], double const direction[3], unsigned int eventId, int parentId,
                   short threadId)
      : energy{eKin}, globalTime{globalTime}, localTime{localTime}, properTime{properTime}, eventId{eventId},
        parentId{parentId}, threadId{threadId}
  {
    rngState.SetSeed(rngSeed);

    pos = {position[0], position[1], position[2]};
    dir = {direction[0], direction[1], direction[2]};
  }

  /// Construct a secondary from a parent track.
  /// NB: The caller is responsible to branch a new RNG state.
  __device__ Track(RanluxppDouble const &rngState, double energy, const vecgeom::Vector3D<Precision> &parentPos,
                   const vecgeom::Vector3D<Precision> &newDirection, const vecgeom::NavigationState &newNavState,
                   const Track &parentTrack)
      : rngState{rngState}, energy{energy}, globalTime{parentTrack.globalTime}, pos{parentPos}, dir{newDirection},
        navState{newNavState}, eventId{parentTrack.eventId}, parentId{parentTrack.parentId},
        threadId{parentTrack.threadId}
  {
  }

  Track const &operator=(Track const &other) = delete;
};
#endif
