// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRACKDATA_H
#define ADEPT_TRACKDATA_H

#include <AdePT/base/MParray.h>

#include "VecGeom/navigation/NavigationState.h"

namespace adeptint {

/// @brief Track data exchanged between Geant4 and AdePT
/// @details This struct is initialised from an AdePT Track, either in GPU or CPU, copied to
/// the destination, and used to reconstruct the track
struct TrackData {
  vecgeom::NavigationState navState;
  double position[3];
  double direction[3];
  double eKin{0};
  double globalTime{0};
  double localTime{0};
  double properTime{0};
  int pdg{0};
  int parentID{0};

  TrackData() = default;
  TrackData(int pdg_id, int parentID, double ene, double x, double y, double z, double dirx, double diry, double dirz,
            double gTime, double lTime, double pTime, vecgeom::NavigationState state)
      : navState{state}, position{x, y, z}, direction{dirx, diry, dirz}, eKin{ene}, globalTime{gTime}, localTime{lTime},
        properTime{pTime}, pdg{pdg_id}, parentID{parentID}
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
