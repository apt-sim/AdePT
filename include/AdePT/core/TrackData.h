// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRACKDATA_H
#define ADEPT_TRACKDATA_H

#include <AdePT/base/MParray.h>

#include "VecGeom/navigation/NavigationState.h"

enum class LeakStatus : char { NoLeak, OutOfGPURegion, GammaNuclear, LeptonNuclear, FinishEventOnCPU };

namespace adeptint {

/// @brief Track data exchanged between Geant4 and AdePT
/// @details This struct is initialised from an AdePT Track, either in GPU or CPU, copied to
/// the destination, and used to reconstruct the track
struct TrackData {
  vecgeom::NavigationState navState;
  vecgeom::NavigationState originNavState;
  double position[3];
  double vertexPosition[3];
  double direction[3];
  double vertexMomentumDirection[3];
  double eKin{0};
  double vertexEkin{0};
  double globalTime{0};
  double localTime{0};
  double properTime{0};
  float weight{0};
  int pdg{0};
  uint64_t trackId{0};  ///< track id (non-consecutive, reproducible)
  uint64_t parentId{0}; // track id of the parent
  short creatorProcessId{-1};
  unsigned short stepCounter{0};

  LeakStatus leakStatus{LeakStatus::NoLeak};

  TrackData() = default;
  TrackData(int pdg_id, uint64_t trackId, uint64_t parentId, short creatorProcessId, double ene, double vertexEne,
            double x, double y, double z, double dirx, double diry, double dirz, double vertexX, double vertexY,
            double vertexZ, double vertexDirx, double vertexDiry, double vertexDirz, double gTime, double lTime,
            double pTime, float weight, unsigned short stepCounter, vecgeom::NavigationState &&state,
            vecgeom::NavigationState &&originState)
      : navState{std::move(state)}, originNavState{std::move(originState)}, position{x, y, z},
        vertexPosition{vertexX, vertexY, vertexZ}, direction{dirx, diry, dirz},
        vertexMomentumDirection{vertexDirx, vertexDiry, vertexDirz}, eKin{ene}, vertexEkin{vertexEne},
        globalTime{gTime}, localTime{lTime}, properTime{pTime}, weight{weight}, pdg{pdg_id}, trackId{trackId},
        creatorProcessId{creatorProcessId}, parentId{parentId}, stepCounter{stepCounter}
  {
  }

  // fixme: add include navigation state in operators?
  friend bool operator==(TrackData const &a, TrackData const &b) { return !(a < b && b < a); }
  friend bool operator!=(TrackData const &a, TrackData const &b) { return !(a == b); }
  inline bool operator<(TrackData const &t) const
  {
    if (pdg != t.pdg) return pdg < t.pdg;
    if (eKin != t.eKin) return eKin < t.eKin;
    if (position[0] != t.position[0]) return position[0] < t.position[0];
    if (position[1] != t.position[1]) return position[1] < t.position[1];
    if (position[2] != t.position[2]) return position[2] < t.position[2];
    if (direction[0] != t.direction[0]) return direction[0] < t.direction[0];
    if (direction[1] != t.direction[1]) return direction[1] < t.direction[1];
    if (direction[2] != t.direction[2]) return direction[2] < t.direction[2];
    return false;
  }
};
} // namespace adeptint

using MParrayTracks = adept::MParrayT<adeptint::TrackData>;
#endif
