// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_RETURNED_TRACK_DATA_HH
#define ADEPT_RETURNED_TRACK_DATA_HH

#include <AdePT/core/TrackData.h>

#include <cstdint>
#include <utility>

namespace AsyncAdePT {

struct TrackDataWithIDs : public adeptint::TrackData {
  unsigned int eventId{0};
  short threadId{-1};

  TrackDataWithIDs(int pdg_id, uint64_t trackId, uint64_t parentId, double ene, double x, double y, double z,
                   double dirx, double diry, double dirz, double gTime, double lTime, double pTime, float weight,
                   unsigned short stepCounter, vecgeom::NavigationState &&state, unsigned int eventId = 0,
                   short threadId = -1)
      : TrackData{pdg_id, trackId, parentId, ene,   x,     y,      z,           dirx,
                  diry,   dirz,    gTime,    lTime, pTime, weight, stepCounter, std::move(state)},
        eventId{eventId}, threadId{threadId}
  {
  }

  friend bool operator==(TrackDataWithIDs const &a, TrackDataWithIDs const &b)
  {
    return a.threadId != b.threadId || a.eventId != b.eventId || static_cast<adeptint::TrackData const &>(a) == b;
  }

  bool operator<(TrackDataWithIDs const &other)
  {
    if (threadId != other.threadId) return threadId < other.threadId;
    if (eventId != other.eventId) return eventId < other.eventId;
    return TrackData::operator<(other);
  }
};

} // namespace AsyncAdePT

#endif
