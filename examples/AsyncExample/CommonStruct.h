// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_INTEGRATION_COMMONSTRUCT_H
#define ADEPT_INTEGRATION_COMMONSTRUCT_H

#include "ResourceManagement.h"

#include <atomic>
#include <array>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>
#include <new>
#include <vector>

// Common data structures used by the integration with Geant4
namespace adeptint {

/// @brief Auxiliary logical volume data. This stores in the same structure the material-cuts couple index,
/// the sensitive volume handler index and the flag if the region is active for AdePT.
struct VolAuxData {
  int fSensIndex{-1}; ///< index of handler for sensitive volumes (-1 means non-sensitive)
  int fMCIndex{0};    ///< material-cut cuple index in G4HepEm
  int fGPUregion{0};  ///< GPU region index (currently 1 or 0, meaning tracked on GPU or not)
};

/// @brief Track data exchanged between Geant4 and AdePT
struct TrackData {
  double position[3];
  double direction[3];
  double energy{0};
  int pdg{0};
  int threadId{-1};
  int eventId{-1};
  unsigned int trackId{0};

  TrackData() = default;
  TrackData(int aThreadId, int aEventId, unsigned int aTrackId, int pdg_id, double ene, double x, double y, double z,
            double dirx, double diry, double dirz)
      : position{x, y, z}, direction{dirx, diry, dirz}, energy{ene}, pdg{pdg_id}, threadId{aThreadId},
        eventId{aEventId}, trackId{aTrackId}
  {
  }

  inline bool operator<(TrackData const &t) const
  {
    if (threadId != t.threadId) return threadId < t.threadId;
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

/// @brief Buffer holding input tracks to be transported on GPU and output tracks to be
/// re-injected in the Geant4 stack
struct TrackBuffer {
#ifdef __cpp_lib_hardware_interference_size
  using std::hardware_destructive_interference_size;
#else
  static constexpr size_t hardware_destructive_interference_size = 64;
#endif
  struct alignas(hardware_destructive_interference_size) ToDeviceBuffer {
    TrackData *tracks;
    unsigned int maxTracks;
    std::atomic_uint nTrack;
    std::shared_mutex mutex;
  };
  std::array<ToDeviceBuffer, 2> toDeviceBuffer;
  std::atomic_short toDeviceIndex{0};
  std::atomic_bool flushRequested{false};

  std::vector<std::vector<TrackData>> fromDeviceBuffers;
  std::mutex fromDeviceMutex;

  std::condition_variable_any cv_newTracks;
  std::condition_variable cv_fromDevice;

  TrackBuffer(TrackData *toDevice1, unsigned int maxTracks1, TrackData *toDevice2, unsigned int maxTracks2,
              unsigned short nThread)
      : toDeviceBuffer{{{toDevice1, maxTracks1, 0, {}}, {toDevice2, maxTracks2, 0, {}}}}, fromDeviceBuffers(nThread)
  {
  }

  ToDeviceBuffer &getActiveBuffer() { return toDeviceBuffer[toDeviceIndex]; }
  void swapToDeviceBuffers()
  {
    toDeviceIndex = (toDeviceIndex + 1) % 2;
    flushRequested.store(false, std::memory_order_release);
  }

  /// A handle to access TrackData vectors while holding a lock
  struct TrackHandle {
    TrackData &track;
    std::shared_lock<std::shared_mutex> lock;
  };

  /// @brief Create a handle with lock for tracks that go to the device.
  /// Create a shared_lock and a reference to a track
  /// @return TrackHandle with lock and reference to track slot.
  TrackHandle createToDeviceSlot();

  struct FromDeviceHandle {
    std::vector<TrackData> &tracks;
    std::scoped_lock<std::mutex> lock;
  };

  /// @brief Create a handle with lock for tracks that return from the device.
  /// @return BufferHandle with lock and reference to track vector.
  FromDeviceHandle getTracksFromDevice(int threadId)
  {
    return {fromDeviceBuffers[threadId], std::scoped_lock{fromDeviceMutex}};
  }
};

} // end namespace adeptint
#endif
