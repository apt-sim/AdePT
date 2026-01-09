// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef HOSTTRACKDATAMAPPER_H
#define HOSTTRACKDATAMAPPER_H

#include "G4VUserTrackInformation.hh"
#include "G4VProcess.hh"

#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <memory>

// Forward declaration
class G4PrimaryParticle;

/// @brief A helper struct to store the data that is stored exclusively on the CPU
struct HostTrackData {
  int g4id                               = 0; // the Geant4 track ID
  int g4parentid                         = 0; // the Geant4 parent ID
  uint64_t gpuId                         = 0; // the GPU’s 64-bit track ID
  G4PrimaryParticle *primary             = nullptr;
  G4VProcess *creatorProcess             = nullptr;
  G4VUserTrackInformation *userTrackInfo = nullptr;
  G4LogicalVolume *logicalVolumeAtVertex = nullptr;
  G4ThreeVector vertexPosition;
  G4ThreeVector vertexMomentumDirection;
  G4double vertexKineticEnergy = 0.0;
  unsigned char particleType;

  HostTrackData() = default;

  HostTrackData(HostTrackData &&) noexcept            = default;
  HostTrackData &operator=(HostTrackData &&) noexcept = default;

  HostTrackData(HostTrackData const &)            = delete;
  HostTrackData &operator=(HostTrackData const &) = delete;
};

// This class provides a mapping between G4 id's (int) and AdePT id's (uint64_t).
// Furthermore it holds a vector to all the information that must be kept on the CPU
// such as the pointer to the creator process, the G4 primary particle, and the G4VUserTrackInformation
class HostTrackDataMapper {
public:
  ~HostTrackDataMapper()
  {
    // Clear all memory upon destruction
    gpuToIndex.clear();
    gpuToIndex.rehash(0);
    g4idToGpuId.clear();
    g4idToGpuId.rehash(0);
    std::vector<HostTrackData>().swap(hostDataVec);
  }

  // Using a hash map to find the correct index for a given GPU id and then a vector for all the CPU-only data
  /// Call once at the start of each event, so we can clear and reserve
  void beginEvent(int eventID, size_t expectedTracks = 1'000'000)
  {
    if (eventID != currentEventID) {
      currentEventID = eventID;

      // for debugging:
      // should be 0 unless there are no GPU regions, in that case tracks that are created on the GPU
      // and die on the CPU will still be listed in g4idToGpuId
      // std::cout << " CLEARING HostTrackDataMapper size of gpuToIndex " << gpuToIndex.size() << " size of g4idToGPUid
      // " << g4idToGpuId.size() << " size of hostDataVec " << hostDataVec.size() << std::endl;

      gpuToIndex.clear();
      gpuToIndex.max_load_factor(0.5f);
      gpuToIndex.reserve(expectedTracks);

      g4idToGpuId.clear();
      g4idToGpuId.max_load_factor(0.25f);
      g4idToGpuId.reserve(expectedTracks);

      hostDataVec.clear();
      hostDataVec.reserve(expectedTracks);
      currentGpuReturnG4ID = std::numeric_limits<int>::max();
    }
  }

  /// HOT PATH: 1 hash + bucket probe, returns a reference into the table

  // Assumes caller knows entry exists
  HostTrackData &get(uint64_t gpuId) noexcept
  {
    auto it = gpuToIndex.find(gpuId); // guaranteed to exist here
    assert(it != gpuToIndex.end());
    return hostDataVec[it->second];
  }

  // creates new entry in HostTrackData map
  HostTrackData &create(uint64_t gpuId, bool useNewId = true)
  {
    const int idx = static_cast<int>(hostDataVec.size());
    hostDataVec.emplace_back();            // 1) add the element
    HostTrackData &d = hostDataVec.back(); // 2) take reference

    d.gpuId = gpuId;
    d.g4id  = useNewId ? currentGpuReturnG4ID-- : static_cast<int>(gpuId);

    gpuToIndex.emplace(gpuId, idx); // 3) map gpuId -> slot
    // Note: this is a hot path and increases run time significantly, but is needed for correct re-mapping of tracks
    // that go from GPU to CPU back to GPU, as they need to be assigned the same ID on the GPU
    g4idToGpuId.emplace(d.g4id, gpuId); // 4) reverse map

    return d;
  }

  /// Ensure a live HostTrackData slot exists for this GPU track. If the slot was
  /// retired (or never existed), create it and bind it to the given G4 id.
  /// Preserves reproducibility by keeping the original g4id.
  /// @return reference to the active HostTrackData.
  inline HostTrackData &activateForGPU(uint64_t gpuId, int g4id, bool haveReverse) noexcept
  {
    // Single lookup/insert for gpuToIndex:
    auto [it, inserted] = gpuToIndex.try_emplace(gpuId, static_cast<int>(hostDataVec.size()));
    if (!inserted) return hostDataVec[it->second];

    // We reserved in beginEvent(), so emplace_back() should not reallocate.
    hostDataVec.emplace_back();
    HostTrackData &d = hostDataVec.back();
    d.gpuId          = gpuId;
    d.g4id           = g4id; // preserve CPU id for reproducibility

    // Reverse map: only touch if missing
    if (!haveReverse) {
      g4idToGpuId.emplace(g4id, gpuId);
    }
#ifdef DEBUG
    else {
      auto rit = g4idToGpuId.find(g4id);
      if (rit == g4idToGpuId.end() || rit->second != gpuId) {
        std::cerr << "HostTrackDataMapper: inconsistent reverse map for g4id=" << g4id << std::endl;
        std::abort();
      }
    }
#endif
    return d;
  }

  /// @brief Sets the gpuid by reference and returns whether the entry already existed
  /// @param g4id int G4 id that is checked
  /// @param gpuid uint64 gpu id that is returned
  /// @return whether the GPU id already existed
  bool getGPUId(int g4id, uint64_t &gpuid)
  {
    auto it = g4idToGpuId.find(g4id);
    if (it == g4idToGpuId.end()) {
      gpuid = static_cast<uint64_t>(g4id);
      return false;
    }
    gpuid = it->second;
    return true;
  }

  /// Call when a track (with given gpuId) is completely done:
  void removeTrack(uint64_t gpuId)
  {
    auto it = gpuToIndex.find(gpuId);
    if (it == gpuToIndex.end()) return; // already gone
    int idx = it->second;

    // As the data of the userTrackInfo is owned by AdePT, it has to be deleted here
    if (hostDataVec[idx].userTrackInfo) {
      delete hostDataVec[idx].userTrackInfo;
      hostDataVec[idx].userTrackInfo = nullptr;
    }

    int last = int(hostDataVec.size()) - 1;

    // unused g4 id
    const int g4idToErase = hostDataVec[idx].g4id;
    if (idx != last) {
      // move last element into idx
      std::swap(hostDataVec[idx], hostDataVec[last]);
      // update its map entry
      gpuToIndex[hostDataVec[idx].gpuId] = idx;
    }
    hostDataVec.pop_back();
    gpuToIndex.erase(it);
    // second part of deletion of g4 ids
    g4idToGpuId.erase(g4idToErase);
  }

  // Free the big struct + index, keep g4id->gpuId for possible future reuse
  // This is needed for reproducibility. Since the track ID is used to seed the RNG
  // if a particle from the GPU goes to the CPU and then back to the GPU, we must ensure that it gets the same GPU id.
  // therefore, when we retire to the CPU, we must not delete the g4idToGpuId to keep the link!
  void retireToCPU(uint64_t gpuId)
  {
    auto it = gpuToIndex.find(gpuId);
    if (it == gpuToIndex.end()) return;
    int idx  = it->second;
    int last = int(hostDataVec.size()) - 1;

    if (idx != last) {
      std::swap(hostDataVec[idx], hostDataVec[last]);
      gpuToIndex[hostDataVec[idx].gpuId] = idx;
    }
    hostDataVec.pop_back();
    gpuToIndex.erase(it);
    // NOTE: intentionally *do not* erase g4idToGpuId here
  }

  /// @brief Whether an entry exists in the GPU to Index map for the given GPU id
  /// @param gpuId GPU id to be checked
  /// @return true if the value exists
  bool contains(uint64_t gpuId) const { return gpuToIndex.find(gpuId) != gpuToIndex.end(); }

private:
  std::unordered_map<uint64_t, int> gpuToIndex;  // key→slot in hostDataVec
  std::unordered_map<int, uint64_t> g4idToGpuId; // geant4 id to GPU id, needed for reverse lookup
  std::vector<HostTrackData> hostDataVec;        // contiguous array of all data

  int currentGpuReturnG4ID = std::numeric_limits<int>::max();
  int currentEventID       = -1;
};

#endif
