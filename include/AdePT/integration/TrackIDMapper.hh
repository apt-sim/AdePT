// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TRACKIDMAPPER_H
#define TRACKIDMAPPER_H

#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <limits>

#include "G4VUserTrackInformation.hh"
#include "G4VProcess.hh"

// #include "robin_hood.h"

// Forward declaration
class G4PrimaryParticle;

/// @brief A helper struct to store the data that is stored exclusively on the CPU
struct HostTrackData {
  int g4id;       // the Geant4 track ID
  int g4parentid; // the Geant4 parent ID
  uint64_t gpuId; // the GPU’s 64-bit track ID
  G4PrimaryParticle *primary             = nullptr;
  G4VProcess *creatorProcess             = nullptr;
  G4VUserTrackInformation *userTrackInfo = nullptr;
  G4LogicalVolume *logicalVolumeAtVertex = nullptr;
  G4ThreeVector vertexPosition;
  G4ThreeVector vertexMomentumDirection;
  G4double vertexKineticEnergy;
};

// This class provides a mapping between G4 id's (int) and AdePT id's (uint64_t).
// Furthermore it holds a map to all the information that must be kept on the CPU
// such as the pointer to the creator process, the G4 primary particle, and the G4VUserTrackInformation
class TrackIDMapper {
public:
  // V2: HASH MAP + then VECTOR FOR DATA
  /// Call once at the start of each event, so we can clear and reserve
  void beginEvent(int eventID, size_t expectedTracks = 1'000'000)
  {
    if (eventID != currentEventID) {
      currentEventID = eventID;
      gpuToIndex.clear();
      gpuToIndex.max_load_factor(0.5f);

      gpuToIndex.reserve(expectedTracks);
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
    auto &d = hostDataVec.back();
    d.gpuId = gpuId;
    d.g4id  = useNewId ? currentGpuReturnG4ID-- : static_cast<int>(gpuId);
    return d;
  }

  // Assumes caller knows entry exists
  HostTrackData &get(uint64_t gpuId) { return hostDataVec[gpuToIndex.at(gpuId)]; }

  // Assumes caller knows entry does not exist yet
  HostTrackData &create(uint64_t gpuId, bool useNewId = true)
  {
    int idx           = static_cast<int>(hostDataVec.size());
    gpuToIndex[gpuId] = idx;
    hostDataVec.push_back({});
    auto &d = hostDataVec.back();
    d.gpuId = gpuId;
    d.g4id  = useNewId ? currentGpuReturnG4ID-- : static_cast<int>(gpuId);
    return d;
  }

  void removeTrack(uint64_t gpuId)
  {
    auto it = gpuToIndex.find(gpuId);
    if (it == gpuToIndex.end()) return; // already gone
    int idx  = it->second;
    int last = int(hostDataVec.size()) - 1;
    if (idx != last) {
      // move last element into idx
      std::swap(hostDataVec[idx], hostDataVec[last]);
      // update its map entry
      gpuToIndex[hostDataVec[idx].gpuId] = idx;
    }
    hostDataVec.pop_back();
    gpuToIndex.erase(it);
  }
  // end V2

  // V1: one big hash map
  // /// Call once at the start of each event, so we can clear and reserve
  // void beginEvent(int eventId, size_t expectedTracks = 20'000'000) {
  //   if (eventId != currentEventID) {
  //     std::cout << " SIZE BEFORE CLEARING " << gpuToHost.size() << std::endl;
  //     gpuToHost.clear();
  //     currentGpuReturnG4ID = std::numeric_limits<int>::max();
  //     currentEventID       = eventId;
  //     gpuToHost.max_load_factor(0.5f);
  //     gpuToHost.reserve(expectedTracks);
  //   }
  // }

  // /// HOT PATH: 1 hash + bucket probe, returns a reference into the table
  // HostTrackData& getOrCreate(uint64_t gpuId, bool useNewId=true) {
  //   auto [it, inserted] = gpuToHost.try_emplace(gpuId);
  //   if (inserted) {
  //     // first‐time initialization
  //     it->second.gpuId = gpuId;
  //     if (useNewId) {
  //       it->second.g4id  = currentGpuReturnG4ID--;
  //     } else {
  //       it->second.g4id = static_cast<int>(gpuId);
  //     }
  //   }

  //   std::cout << " g4id " << it->second.g4id << " inserted " << inserted << " useNewId " << useNewId <<  "
  //   currentsize " << gpuToHost.size() << std::endl;

  //   return it->second;
  // }
  // END V1

private:
  // V2:
  std::unordered_map<uint64_t, int> gpuToIndex; // key→slot in hostDataVec
  std::vector<HostTrackData> hostDataVec;       // contiguous array of all data
                                                // V1
  // std::unordered_map<uint64_t,HostTrackData> gpuToHost;

  int currentGpuReturnG4ID = std::numeric_limits<int>::max();
  int currentEventID       = -1;
};

#endif
