// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRACKDATA_H
#define ADEPT_TRACKDATA_H

#include <AdePT/base/MParray.h>

#include "VecGeom/navigation/NavigationState.h"

#include <cstdint>
#include <utility>

namespace adeptint {

/// @brief Track data uploaded from Geant4 to AdePT for GPU transport.
struct TrackData {
  vecgeom::NavigationState navState;
  double position[3];
  double direction[3];
  double eKin{0};
  double globalTime{0};
  double localTime{0};
  double properTime{0};
  float weight{0};
  int pdg{0};
  uint64_t trackId{0};  ///< track id (non-consecutive, reproducible)
  uint64_t parentId{0}; // track id of the parent
  unsigned short stepCounter{0};
  unsigned int eventId{0};
  short threadId{-1};

  TrackData() = default;
  TrackData(int pdg_id, uint64_t trackId, uint64_t parentId, double ene, double x, double y, double z, double dirx,
            double diry, double dirz, double gTime, double lTime, double pTime, float weight,
            unsigned short stepCounter, vecgeom::NavigationState &&state, unsigned int eventId = 0, short threadId = -1)
      : navState{std::move(state)}, position{x, y, z}, direction{dirx, diry, dirz}, eKin{ene}, globalTime{gTime},
        localTime{lTime}, properTime{pTime}, weight{weight}, pdg{pdg_id}, trackId{trackId}, parentId{parentId},
        stepCounter{stepCounter}, eventId{eventId}, threadId{threadId}
  {
  }
};
} // namespace adeptint

using MParrayTracks = adept::MParrayT<adeptint::TrackData>;
#endif
