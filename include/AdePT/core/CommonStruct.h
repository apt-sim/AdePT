// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_INTEGRATION_COMMONSTRUCT_H
#define ADEPT_INTEGRATION_COMMONSTRUCT_H

#include <AdePT/base/MParray.h>
#include <AdePT/core/TrackData.h>

#include <AdePT/base/ResourceManagement.hh>
#include <AdePT/copcore/Ranluxpp.h>

#include <G4HepEmRandomEngine.hh>

#include "G4RegionStore.hh"
#include "G4Region.hh"

#include <atomic>
#include <array>
#include <mutex>
#include <shared_mutex>
#include <new>
#include <vector>
#include <thread>

#ifdef __CUDA_ARCH__
// Define inline implementations of the RNG methods for the device.
// (nvcc ignores the __device__ attribute in definitions, so this is only to
// communicate the intent.)
inline __device__ double G4HepEmRandomEngine::flat()
{
  return ((RanluxppDouble *)fObject)->Rndm();
}

inline __device__ void G4HepEmRandomEngine::flatArray(const int size, double *vect)
{
  for (int i = 0; i < size; i++) {
    vect[i] = ((RanluxppDouble *)fObject)->Rndm();
  }
}
#endif

// Common data structures used by the integration with Geant4
namespace adeptint {

/// @brief Auxiliary logical volume data. This stores in the same structure the material-cuts couple index,
/// the sensitive volume handler index and the flag if the region is active for AdePT.
struct VolAuxData {
  int fSensIndex{-1};   ///< index of handler for sensitive volumes (-1 means non-sensitive)
  int fMCIndex{0};      ///< material-cut couple index in G4HepEm
  int fGPUregionId{-1}; ///< GPU region index, corresponds to G4Region.instanceID if tracked on GPU, -1 otherwise
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

// Woodcock tracking helper structures

/// @brief Root Volume data of a Woodcock tracking region: Navigation index + G4HepEm material cut couple index// Raw
/// per-root entry
struct WDTRoot {
  vecgeom::NavigationState root; // NavState of the root placed volume
  int hepemIMC;                  // G4HepEm mat-cut index for this root
};

// Compact region header
struct WDTRegion {
  int offset;    // first index in roots[]
  int count;     // number of roots for this region
  float ekinMin; // kinetic energy threshold
};

// Device view pointing to the GPU data plus the number of roots and regions, required for access
struct WDTDeviceView {
  const WDTRoot *roots;     // [nRoots]
  const WDTRegion *regions; // [nRegions] (only WDT-enabled regions)
  const int *regionToWDT;   // [regionToWDTLen], regionId -> bucket (index into regions[]) or -1
  int nRoots;
  int nRegions;
};

// Temporary, sparse collection built during geometry traversal.
// - `roots` holds every discovered root volume in a Woodcock tracking region + the material index
// - `regionToRootIndices[rid]` maps from a region index to the list of indices of the Root volumes (each region can
// have multiple root volumes)
// - `ekinMin` is the global Woodcock minimum kinetic energy for Woodcock tracking
struct WDTHostRaw {
  // mapping: regionId -> list of indices into `roots`
  std::unordered_map<int, std::vector<int>> regionToRootIndices;
  // found during geometry visit (one entry per root placed volume)
  std::vector<WDTRoot> roots; // List of all Roots. Access for the roots for a given region via the regionToRootIndices
  float ekinMin{0.f};
};

// Compact, upload-ready representation.
// - `roots` is packed so that each region's roots are contiguous.
// - `regions[w]` points to the slice (offset,count) for region index `w`.
// - `regionToWDT[regionId]` returns `w` (or -1 if that region has no WDT).
struct WDTHostPacked {
  std::vector<WDTRoot> roots;     // packed per-region contiguous
  std::vector<WDTRegion> regions; // one per WDT region
  std::vector<int> regionToWDT;   // dense by regionId (size = number of G4 regions)
};

// Owned device buffers to manage lifetime of Woodcock tracking data
struct WDTDeviceBuffers {
  WDTRoot *d_roots     = nullptr;
  WDTRegion *d_regions = nullptr;
  int *d_map           = nullptr;
};

/// @brief This packs the Woodcock data from the original map to arrays that can be copied to the GPU
/// @param raw raw WDT data, stored in a map
/// @return packed, dense WDT data, ready to be copied to the GPU
inline WDTHostPacked PackWDT(const WDTHostRaw &raw)
{
  WDTHostPacked packed;

  // Build dense regionId -> bucket index
  int maxRegionId = -1;
  for (auto *r : *G4RegionStore::GetInstance())
    if (r) maxRegionId = std::max(maxRegionId, r->GetInstanceID());

  packed.regionToWDT.assign(maxRegionId + 1, -1);

  packed.roots.reserve(raw.roots.size());
  packed.regions.reserve(raw.regionToRootIndices.size());

  int runningOffset = 0;
  for (const auto &kv : raw.regionToRootIndices) {
    const int rid    = kv.first;
    const auto &idxs = kv.second;

    packed.regionToWDT[rid] = (int)packed.regions.size();
    packed.regions.push_back(WDTRegion{runningOffset, (int)idxs.size(), raw.ekinMin});

    for (int idx : idxs) {
      packed.roots.push_back(raw.roots[idx]); // preserve order per region
    }
    runningOffset += (int)idxs.size();
  }

  return packed;
}

} // end namespace adeptint

namespace AsyncAdePT {

struct TrackDataWithIDs : public adeptint::TrackData {
  unsigned int eventId{0};
  short threadId{-1};

  TrackDataWithIDs(int pdg_id, uint64_t trackId, uint64_t parentId, double ene, double x, double y, double z,
                   double dirx, double diry, double dirz, double gTime, double lTime, double pTime, float weight,
                   unsigned short stepCounter, vecgeom::NavigationState &&state, vecgeom::NavigationState &&originState,
                   unsigned int eventId = 0, short threadId = -1)
      : TrackData{pdg_id,
                  trackId,
                  parentId,
                  ene,
                  x,
                  y,
                  z,
                  dirx,
                  diry,
                  dirz,
                  gTime,
                  lTime,
                  pTime,
                  weight,
                  stepCounter,
                  std::move(state),
                  std::move(originState)},
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
