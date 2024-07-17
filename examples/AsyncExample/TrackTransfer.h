// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TRACK_TRANSFER_H
#define TRACK_TRANSFER_H

#include <AdePT/core/TrackData.h>

#include "ResourceManagement.h"

#include <atomic>
#include <array>
#include <mutex>
#include <shared_mutex>
#include <new>
#include <vector>

namespace AsyncAdePT {

struct TrackDataWithIDs : public adeptint::TrackData {
  unsigned int eventId{0};
  unsigned int trackId{0};
  short threadId{-1};

  TrackDataWithIDs(int pdg_id, double ene, double x, double y, double z, double dirx, double diry, double dirz,
                   double gTime, double lTime, double pTime, unsigned int eventId = 0, unsigned int trackId = 0,
                   short threadId = -1)
      : TrackData{pdg_id, ene, x, y, z, dirx, diry, dirz, gTime, lTime, pTime}, eventId{eventId}, trackId{trackId},
        threadId{threadId}
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

/// @brief Buffer holding input tracks to be transported on GPU and output tracks to be
/// re-injected in the Geant4 stack
struct TrackBuffer {
  struct alignas(64) ToDeviceBuffer {
    TrackDataWithIDs *tracks;
    unsigned int maxTracks;
    std::atomic_uint nTrack;
    mutable std::shared_mutex mutex;
  };
  std::array<ToDeviceBuffer, 2> toDeviceBuffer;
  std::atomic_short toDeviceIndex{0};

  std::vector<std::vector<TrackDataWithIDs>> fromDeviceBuffers;
  std::mutex fromDeviceMutex;

  TrackBuffer(TrackDataWithIDs *toDevice1, unsigned int maxTracks1, TrackDataWithIDs *toDevice2,
              unsigned int maxTracks2, unsigned short nThread)
      : toDeviceBuffer{{{toDevice1, maxTracks1, 0, {}}, {toDevice2, maxTracks2, 0, {}}}}, fromDeviceBuffers(nThread)
  {
  }

  ToDeviceBuffer &getActiveBuffer() { return toDeviceBuffer[toDeviceIndex]; }
  void swapToDeviceBuffers() { toDeviceIndex = (toDeviceIndex + 1) % 2; }

  /// A handle to access TrackData vectors while holding a lock
  struct TrackHandle {
    TrackDataWithIDs &track;
    std::shared_lock<std::shared_mutex> lock;
  };

  /// @brief Create a handle with lock for tracks that go to the device.
  /// Create a shared_lock and a reference to a track
  /// @return TrackHandle with lock and reference to track slot.
  TrackHandle createToDeviceSlot();

  struct FromDeviceHandle {
    std::vector<TrackDataWithIDs> &tracks;
    std::scoped_lock<std::mutex> lock;
  };

  /// @brief Create a handle with lock for tracks that return from the device.
  /// @return BufferHandle with lock and reference to track vector.
  FromDeviceHandle getTracksFromDevice(int threadId)
  {
    return {fromDeviceBuffers[threadId], std::scoped_lock{fromDeviceMutex}};
  }
};

} // namespace AsyncAdePT

#endif
