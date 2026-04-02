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
#include <vector>

namespace AsyncAdePT {

/// @brief Buffer holding input tracks to be transported on GPU and output tracks to be
/// re-injected in the Geant4 stack.
struct TrackBuffer {
  struct alignas(64) ToDeviceBuffer {
    TrackDataWithIDs *tracks;
    unsigned int maxTracks;
    std::atomic_uint nTrack;
    mutable std::shared_mutex mutex;
  };

  std::array<ToDeviceBuffer, 2> toDeviceBuffer;
  std::atomic_short toDeviceIndex{0};

  unsigned int fNumToDevice{0};   ///< number of slots in the toDevice buffer
  unsigned int fNumFromDevice{0}; ///< number of slots in the fromDevice buffer
  unsigned int fNumLeaksTransferred{
      0}; ///< Used to keep track of the number of tracks transferred from device during extraction
  unique_ptr_cuda<TrackDataWithIDs, CudaHostDeleter<TrackDataWithIDs>>
      toDevice_host;                              ///< Tracks to be transported to the device
  unique_ptr_cuda<TrackDataWithIDs> toDevice_dev; ///< toDevice buffer of tracks
  unique_ptr_cuda<TrackDataWithIDs, CudaHostDeleter<TrackDataWithIDs>> fromDevice_host; ///< Tracks from device
  unique_ptr_cuda<TrackDataWithIDs> fromDevice_dev;                                     ///< fromDevice buffer of tracks
  unique_ptr_cuda<unsigned int, CudaHostDeleter<unsigned int>>
      nFromDevice_host; ///< Number of tracks collected on device
  unique_ptr_cuda<unsigned int, CudaHostDeleter<unsigned int>>
      nRemainingLeaks_host; ///< Number of tracks still left to transfer from device during extraction

  std::vector<std::vector<TrackDataWithIDs>> fromDeviceBuffers;
  std::mutex fromDeviceMutex;

  TrackBuffer(unsigned int numToDevice, unsigned int numFromDevice, unsigned short nThread);

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
