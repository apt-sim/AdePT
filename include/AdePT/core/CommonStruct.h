// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_INTEGRATION_COMMONSTRUCT_H
#define ADEPT_INTEGRATION_COMMONSTRUCT_H

#include <vector>
#include <AdePT/base/MParray.h>
#include <AdePT/core/TrackData.h>

#include <AdePT/base/ResourceManagement.hh>

#include <atomic>
#include <array>
#include <mutex>
#include <shared_mutex>
#include <new>
#include <vector>
#include <thread>

// Common data structures used by the integration with Geant4
namespace adeptint {

/// @brief Common configuration data for AdePT transport
struct CommonConfig {
  int fDebugLevel; ///< Debug level

  static CommonConfig &GetInstance()
  {
    static CommonConfig theConfig;
    return theConfig;
  }
};

/// @brief Auxiliary logical volume data. This stores in the same structure the material-cuts couple index,
/// the sensitive volume handler index and the flag if the region is active for AdePT.
struct VolAuxData {
  int fSensIndex{-1}; ///< index of handler for sensitive volumes (-1 means non-sensitive)
  int fMCIndex{0};    ///< material-cut cuple index in G4HepEm
  int fGPUregion{0};  ///< GPU region index (currently 1 or 0, meaning tracked on GPU or not)
};

/// @brief Structure holding the arrays of auxiliary volume data on host and device
struct VolAuxArray {
  int fNumVolumes{0};
  VolAuxData *fAuxData{nullptr};     ///< array of auxiliary volume data on host
  VolAuxData *fAuxData_dev{nullptr}; ///< array of auxiliary volume data on device

  static VolAuxArray &GetInstance()
  {
    static VolAuxArray theAuxArray;
    return theAuxArray;
  }
};

/// @brief Buffer holding input tracks to be transported on GPU and output tracks to be
/// re-injected in the Geant4 stack
struct TrackBuffer {
  std::vector<TrackData> toDevice;    ///< Tracks to be transported on the device
  std::vector<TrackData> fromDevice;  ///< Tracks coming from device to be transported on the CPU.
                                      ///< Initialized from "fromDeviceBuff" after the copy
  TrackData *fromDeviceBuff{nullptr}; ///< Buffer of leaked tracks from device.
                                      ///< Used as the destination for the Cuda copy
  int buffSize{0};                    ///< Size of buffer collecting tracks from device
  int eventId{-1};                    ///< Index of current transported event
  int startTrack{0};                  ///< Track counter for the current event
  int nelectrons{0};                  ///< Number of electrons in the input buffer
  int npositrons{0};                  ///< Number of positrons in the input buffer
  int ngammas{0};                     ///< Number of gammas in the input buffer

  void Clear()
  {
    toDevice.clear();
    fromDevice.clear();
    nelectrons = npositrons = ngammas = 0;
  }
};

} // end namespace adeptint

namespace AsyncAdePT {

struct TrackDataWithIDs : public adeptint::TrackData {
  unsigned int eventId{0};
  unsigned int trackId{0};
  short threadId{-1};

  TrackDataWithIDs(int pdg_id, int parentID, double ene, double x, double y, double z, double dirx, double diry,
                   double dirz, double gTime, double lTime, double pTime, vecgeom::NavigationState &&state,
                   unsigned int eventId = 0, unsigned int trackId = 0, short threadId = -1)
      : TrackData{pdg_id, parentID, ene, x, y, z, dirx, diry, dirz, gTime, lTime, pTime, std::move(state)},
        eventId{eventId}, trackId{trackId}, threadId{threadId}
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

  unsigned int fNumToDevice{0};   ///< number of slots in the toDevice buffer
  unsigned int fNumFromDevice{0}; ///< number of slots in the fromDevice buffer
  unique_ptr_cuda<TrackDataWithIDs, CudaHostDeleter<TrackDataWithIDs>>
      toDevice_host;                              ///< Tracks to be transported to the device
  unique_ptr_cuda<TrackDataWithIDs> toDevice_dev; ///< toDevice buffer of tracks
  unique_ptr_cuda<TrackDataWithIDs, CudaHostDeleter<TrackDataWithIDs>> fromDevice_host; ///< Tracks from device
  unique_ptr_cuda<TrackDataWithIDs> fromDevice_dev;                                     ///< fromDevice buffer of tracks
  unique_ptr_cuda<unsigned int, CudaHostDeleter<unsigned int>>
      nFromDevice_host; ///< Number of tracks collected on device

  std::vector<std::vector<TrackDataWithIDs>> fromDeviceBuffers;
  std::mutex fromDeviceMutex;

  TrackBuffer(unsigned int numToDevice, unsigned int numFromDevice, unsigned short nThread);

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
  // TrackHandle createToDeviceSlot();
  TrackHandle createToDeviceSlot()
  {
    bool warningIssued = false;
    while (true) {
      auto &toDevice = getActiveBuffer();
      std::shared_lock lock{toDevice.mutex};
      const auto slot = toDevice.nTrack.fetch_add(1, std::memory_order_relaxed);

      if (slot < toDevice.maxTracks)
        return TrackHandle{toDevice.tracks[slot], std::move(lock)};
      else {
        if (!warningIssued) {
          std::cerr << __FILE__ << ':' << __LINE__ << " Contention in to-device queue; thread sleeping" << std::endl;
          warningIssued = true;
        }
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(1ms);
      }
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
