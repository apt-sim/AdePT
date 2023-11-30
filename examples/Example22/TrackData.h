// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRACKDATA_H
#define ADEPT_TRACKDATA_H

#include <AdePT/MParray.h>

namespace adeptint {

/// @brief Track data exchanged between Geant4 and AdePT
struct TrackData {
  double position[3];
  double direction[3];
  double energy{0};
  int pdg{0};

  TrackData() = default;
  TrackData(int pdg_id, double ene, double x, double y, double z, double dirx, double diry, double dirz)
      : position{x, y, z}, direction{dirx, diry, dirz}, energy{ene}, pdg{pdg_id}
  {
  }

  inline bool operator<(TrackData const &t)
  {
    if (pdg != t.pdg) return pdg < t.pdg;
    if (energy != t.energy) return energy < t.energy;
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
