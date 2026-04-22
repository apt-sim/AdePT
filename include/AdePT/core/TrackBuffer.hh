// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRACK_BUFFER_HH
#define ADEPT_TRACK_BUFFER_HH

#include <AdePT/base/ResourceManagement.hh>
#include <AdePT/core/ReturnedTrackData.hh>

#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>

namespace AsyncAdePT {

/// @brief Buffer holding input tracks to be transported on GPU.
struct TrackBuffer {
  struct alignas(64) ToDeviceBuffer {
    TrackDataWithIDs *tracks;
    unsigned int maxTracks;
    std::atomic_uint nTrack;
    mutable std::shared_mutex mutex;
  };

  std::array<ToDeviceBuffer, 2> toDeviceBuffer;
  std::atomic_short toDeviceIndex{0};

  unsigned int fNumToDevice{0}; ///< number of slots in the toDevice buffer
  unique_ptr_cuda<TrackDataWithIDs, CudaHostDeleter<TrackDataWithIDs>>
      toDevice_host;                              ///< Tracks to be transported to the device
  unique_ptr_cuda<TrackDataWithIDs> toDevice_dev; ///< toDevice buffer of tracks

  TrackBuffer(unsigned int numToDevice);

  ToDeviceBuffer &getActiveBuffer() { return toDeviceBuffer[toDeviceIndex.load()]; }
  void swapToDeviceBuffers() { toDeviceIndex.store((toDeviceIndex + 1) % 2); }

  /// A handle to access TrackData vectors while holding a lock.
  struct TrackHandle {
    TrackDataWithIDs &track;
    std::shared_lock<std::shared_mutex> lock;
  };

  /// @brief Create a handle with lock for tracks that go to the device.
  /// Create a shared_lock and a reference to a track.
  /// @return TrackHandle with lock and reference to track slot.
  TrackHandle createToDeviceSlot()
  {
    bool warningIssued = false;
    while (true) {
      auto idx       = toDeviceIndex.load();
      auto &toDevice = toDeviceBuffer[idx];

      {
        std::shared_lock lock{toDevice.mutex};

        if (toDeviceIndex.load() != idx) {
          continue;
        }
        const auto slot = toDevice.nTrack.fetch_add(1, std::memory_order_relaxed);

        if (slot < toDevice.maxTracks) {
          return TrackHandle{toDevice.tracks[slot], std::move(lock)};
        }
      }

      if (!warningIssued) {
        std::cerr << __FILE__ << ':' << __LINE__ << " Contention in to-device queue; thread sleeping" << std::endl;
        warningIssued = true;
      }
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(1ms);
    }
  }
};

} // namespace AsyncAdePT

#endif
