// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file TrackManager.h
 * @brief A track manager using a circular buffer with slot compression capability.
 * @details The track manager is templated by the track type and holds a circular
 *          buffer of pre-allocated tracks. It also holds two MParray with track
 *          slots for the current and next iteration.
 *
 *          The compression of the sparse active slots is triggered when the sum of the
 *          number of used slots and two times the number of inflight tracks becomes larger
 *          than a user-defined fraction of the total buffer size. The tracks pointed by the
 *          next iteration array are copied cotiguously by a kernel just after the last used
 *          slot, before making this slot the start index and swapping the current/next index
 *          arrays.
 *
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_TRACKMANAGER_H_
#define ADEPT_TRACKMANAGER_H_

#include <type_traits>
#include <CopCore/CopCore.h>
#include <AdePT/Atomic.h>

namespace adept {

template <typename Track>
struct TrackManager;

namespace device_impl_trackmgr {

template <typename Track>
__global__ void construct_trackmanager(void *addr, size_t capacity, adept::MParray *activeSlots,
                                       adept::MParray *nextSlots, Track *buffer)
{
  // Invoke inplace TrackManager constructor
  auto mgr           = new (addr) TrackManager<Track>(capacity);
  mgr->fActiveTracks = activeSlots;
  mgr->fNextTracks   = nextSlots;
  mgr->fBuffer       = buffer;
  // construct the MParray objects inplace
  adept::MParray::MakeInstanceAt(capacity, activeSlots);
  adept::MParray::MakeInstanceAt(capacity, nextSlots);
}

template <typename Track>
__global__ void clear_trackmanager(TrackManager<Track> *mgr)
{
  mgr->clear();
}

template <typename Track>
__global__ void swap_active(TrackManager<Track> *mgr)
{
  mgr->swap();
}

template <typename Track>
__global__ void defragment_buffer(TrackManager<Track> *mgr, int nactive, int where)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nactive; i += blockDim.x * gridDim.x) {
    const int slot_src     = (*mgr->fNextTracks)[i];
    const int slot_dst     = (where + i) % mgr->fCapacity;
    mgr->fBuffer[slot_dst] = mgr->fBuffer[slot_src];
    // fActiveTracks must be cleared before starting this kernel
    mgr->fActiveTracks->push_back(slot_dst);
  }
}

template <typename Track>
__global__ void adjust_indices(TrackManager<Track> *mgr, int start_new, int next_free)
{
  mgr->fStats.fStart = start_new;
  mgr->fNextFree.store(next_free);
  mgr->fNextTracks->clear();
}

} // End namespace device_impl_trackmgr

/// @brief A track manager working with a circular buffer.
template <typename Track>
struct TrackManager {

  static_assert(std::is_copy_constructible<Track>::value, "TrackManager: The track type must be copy constructible");
  /// @brief Statistics of the track manager to be updated and copied to host between iterations
  struct Stats {
    int fStart{0};     ///< Index of first used track in the buffer
    int fNextStart{0}; ///< Index where the tracks will be compacted
    int fInFlight{0};  ///< Number of tracks sill in flight

    int GetNused() { return fNextStart - fStart; }
  };

  Stats fStats;                       ///< Current statistics
  int fCapacity{0};                   ///< Maximum number of elements
  adept::Atomic_t<int> fNextFree;     ///< Index of last used index in the buffer
  TrackManager *fInstance_d{nullptr}; ///< Device instance

  adept::MParray *fActiveTracks{nullptr}; ///< Array of active (input) track slots (device pointer)
  adept::MParray *fNextTracks{nullptr};   ///< Array of rack slots for the next iteration (device pointer)
  Track *fBuffer{nullptr};                ///< Storage for the circular buffer of tracks (device pointer)

  /// @brief Construction done on host but holding device pointers.
  __host__ __device__ TrackManager(size_t capacity) : fCapacity(capacity) { fNextFree.store(0); }

  /// Construct a device instance and attach it to this instance on host
  TrackManager<Track> *ConstructOnDevice()
  {
    const size_t QueueSize  = adept::MParray::SizeOfInstance(fCapacity);
    const size_t TracksSize = sizeof(Track) * fCapacity;
    COPCORE_CUDA_CHECK(cudaMalloc(&fInstance_d, sizeof(TrackManager<Track>)));
    COPCORE_CUDA_CHECK(cudaMalloc(&fActiveTracks, QueueSize));
    COPCORE_CUDA_CHECK(cudaMalloc(&fNextTracks, QueueSize));
    COPCORE_CUDA_CHECK(cudaMalloc(&fBuffer, TracksSize));

    device_impl_trackmgr::construct_trackmanager<Track>
        <<<1, 1>>>(fInstance_d, fCapacity, fActiveTracks, fNextTracks, fBuffer);
    return fInstance_d;
  }

  void FreeFromDevice()
  {
    COPCORE_CUDA_CHECK(cudaFree(fBuffer));
    COPCORE_CUDA_CHECK(cudaFree(fActiveTracks));
    COPCORE_CUDA_CHECK(cudaFree(fNextTracks));
    COPCORE_CUDA_CHECK(cudaFree(fInstance_d));
  }

  /// @brief Swap active and next track slots. Compact if the fill percentage is higher than the threshold.
  /// @details Must be called after the stats were updated on host.
  template <typename Stream>
  bool SwapAndCompact(float compact_threshold, Stream stream)
  {
    // check if the compacting threshold is hit
    int used     = fStats.GetNused();
    int inFlight = fStats.fInFlight;
    assert(used >= 0 && used < fCapacity);
    // Cannot compress any more if the destination region overlaps the used one
    bool can_compress = (used + inFlight) < fCapacity;
    if (!can_compress)
      std::cout << "TrackManager::SwapAndCompact  ALERT: not enough space left to compress " << inFlight
                << " tracks from " << used << " used slots. Consider increasing TrackManager capacity.\n";

    // Estimate maximum space needed if we DON'T compress now
    int needed = used + 2 * inFlight;
    if (needed < compact_threshold * fCapacity || !can_compress) {
      device_impl_trackmgr::swap_active<Track><<<1, 1, 0, stream>>>(fInstance_d);
      return false;
    }

    constexpr int maxBlocks = 1024;
    constexpr int threads   = 32;

    int blocks = (inFlight + threads - 1) / threads;
    blocks     = min(blocks, maxBlocks);

    fStats.fStart = fStats.fNextStart % fCapacity;
    int next_free = fStats.fStart + inFlight;

    // printf("compacting %d / %d -> %d at slot: %d, next_free: %d\n", used, fCapacity, inFlight, fStats.fStart,
    // next_free % fCapacity);
    device_impl_trackmgr::defragment_buffer<Track>
        <<<blocks, threads, 0, stream>>>(fInstance_d, inFlight, fStats.fStart);
    device_impl_trackmgr::adjust_indices<Track><<<1, 1, 0, stream>>>(fInstance_d, fStats.fStart, next_free);
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
  }

  /// @brief Host static call to clear the container.
  template <typename Stream>
  void Clear(Stream stream)
  {
    fStats.fStart     = 0;
    fStats.fNextStart = 0;
    fStats.fInFlight  = 0;
    device_impl_trackmgr::clear_trackmanager<Track><<<1, 1, 0, stream>>>(fInstance_d);
  }

  /// @brief Device function to clear the container
  __device__ __forceinline__ void clear()
  {
    fStats.fStart     = 0;
    fStats.fNextStart = 0;
    fStats.fInFlight  = 0;
    fNextFree.store(0);
    fActiveTracks->clear();
    fNextTracks->clear();
  }

  /// @brief Refresh the statistics of the track manager and clear the processed active queue.
  __device__ __forceinline__ void refresh_stats()
  {
    if (fNextTracks->size() == 0) {
      clear();
    } else {
      // fStats.fStart is not modified during the transport loop
      fStats.fNextStart = fNextFree.load();
      fStats.fInFlight  = fNextTracks->size();
      fActiveTracks->clear();
    }
  }

  /// @brief Index operator works on device only
  __device__ __forceinline__ Track &operator[](int slot) { return fBuffer[slot]; }

  /// @brief This swaps active with next slots.
  __device__ __forceinline__ void swap()
  {
    auto tmp      = fActiveTracks;
    fActiveTracks = fNextTracks;
    fNextTracks   = tmp;
  }

  /// @brief Get next free slot.
  __device__ __forceinline__ int NextSlot()
  {
    int next = fNextFree.fetch_add(1);
    assert(next >= fStats.fStart);
    if ((next - fStats.fStart) >= fCapacity) return -1;
    return next % fCapacity;
  }

  /// @brief Main interface to get the next unused track on device.
  __device__ __forceinline__ Track &NextTrack()
  {
    int slot = NextSlot();
    if (slot == -1) {
      COPCORE_EXCEPTION("No slot available in TrackManager");
    }
    assert(slot < fCapacity);
    fNextTracks->push_back(slot);
    return fBuffer[slot];
  }

}; // End struct TrackManager

} // End namespace adept

#endif // ADEPT_TRACKMANAGER_H_
