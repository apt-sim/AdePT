// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef HOSTTRACKDATAMAPPER_H
#define HOSTTRACKDATAMAPPER_H

#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <limits>

#include "G4VUserTrackInformation.hh"
#include "G4VProcess.hh"

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
  HostTrackData &getOrCreate(uint64_t gpuId, bool useNewId = true)
  {
    auto it = gpuToIndex.find(gpuId);
    if (it != gpuToIndex.end()) {
      // already have a slot
      return hostDataVec[it->second];
    }
    // new track -> assign next slot
    int idx           = static_cast<int>(hostDataVec.size());
    gpuToIndex[gpuId] = idx;
    hostDataVec.push_back({});
    auto &d             = hostDataVec.back();
    d.gpuId             = gpuId;
    d.g4id              = useNewId ? currentGpuReturnG4ID-- : static_cast<int>(gpuId);
    g4idToGpuId[d.g4id] = gpuId;
    return d;
  }

  // Assumes caller knows entry exists
  HostTrackData &get(uint64_t gpuId) { return hostDataVec[gpuToIndex.at(gpuId)]; }

  // Assumes caller knows entry exists
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

  // Assumes caller knows entry does not exist yet
  HostTrackData &create(uint64_t gpuId, bool useNewId = true)
  {
    int idx           = static_cast<int>(hostDataVec.size());
    gpuToIndex[gpuId] = idx;
    hostDataVec.push_back({});
    auto &d = hostDataVec.back();
    d.gpuId = gpuId;
    d.g4id  = useNewId ? currentGpuReturnG4ID-- : static_cast<int>(gpuId);
    // Note: this is a hot path and increases run time significantly, but is needed for correct re-mapping of tracks
    // that go from GPU to CPU back to GPU, as they need to be assigned the same ID on the GPU
    g4idToGpuId[d.g4id] = gpuId;
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
