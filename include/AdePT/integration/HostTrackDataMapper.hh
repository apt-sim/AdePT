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
  char particleType;

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
  // Using a hash map to find the correct index for a given GPU id and then a vector for all the CPU-only data
  /// Call once at the start of each event, so we can clear and reserve
  void beginEvent(int eventID, size_t expectedTracks = 1'000'000)
  {
    if (eventID != currentEventID) {
      currentEventID = eventID;

      // for debugging:
      // should be 0 unless there are no GPU regions, in that case tracks that are created on the GPU
      // and die on the CPU will still be in the mapper.
      // std::cout << " CLEARING HostTrackDataMapper OF SIZE " << gpuToIndex.size() << std::endl;

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
    // assert(it != gpuToIndex.end());
    return hostDataVec[it->second];
  }

  /// @brief Sets the gpuid by reference and returns whether the entry already existed
  /// @param g4id int G4 id that is checked
  /// @param gpuid uint64 gpu id that is returned
  /// @return whether the GPU id already existed
  bool tryGetGPUId(int g4id, uint64_t &gpuid)
  {
    auto it = g4idToGpuId.find(g4id);
    if (it == g4idToGpuId.end()) {
      gpuid = static_cast<uint64_t>(g4id);
      return false;
    }
    gpuid = it->second;
    return true;
  }

  HostTrackData &getOrCreate(uint64_t gpuId, bool useNewId = true)
  {
    auto [it, inserted] = gpuToIndex.emplace(gpuId, -1);
    if (!inserted) return hostDataVec[it->second];

    const int idx = static_cast<int>(hostDataVec.size());
    hostDataVec.emplace_back(); // in-place default construction
    it->second = idx;

    auto &d             = hostDataVec.back();
    d.gpuId             = gpuId;
    d.g4id              = useNewId ? currentGpuReturnG4ID-- : static_cast<int>(gpuId);
    g4idToGpuId[d.g4id] = gpuId;
    return d;
  }

  HostTrackData &create(uint64_t gpuId, bool useNewId = true)
  {
    const int idx     = static_cast<int>(hostDataVec.size());
    gpuToIndex[gpuId] = idx;
    hostDataVec.emplace_back(); // in-place default construction

    auto &d             = hostDataVec.back();
    d.gpuId             = gpuId;
    d.g4id              = useNewId ? currentGpuReturnG4ID-- : static_cast<int>(gpuId);
    g4idToGpuId[d.g4id] = gpuId; // required for CPU↔GPU↔CPU ping-pong
    return d;
  }

  /// Call when a track (with given gpuId) is completely done:
  void removeTrack(uint64_t gpuId)
  {
    auto it = gpuToIndex.find(gpuId);
    if (it == gpuToIndex.end()) return; // already gone
    int idx  = it->second;
    int last = int(hostDataVec.size()) - 1;
    // optional: delete unused G4 ids. However, this is problematic as the IDs of killed tracks might require lookup
    // when the parent id is set. Therefore unused for now int g4idToErase = hostDataVec[idx].g4id; // optional: delete
    // unused g4 ids
    if (idx != last) {
      // move last element into idx
      std::swap(hostDataVec[idx], hostDataVec[last]);
      // update its map entry
      gpuToIndex[hostDataVec[idx].gpuId] = idx;
    }
    hostDataVec.pop_back();
    gpuToIndex.erase(it);
    // second part of deletion of g4 ids
    // g4idToGpuId.erase(g4idToErase);
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
