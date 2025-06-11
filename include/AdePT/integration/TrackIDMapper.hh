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
  uint64_t gpuId; // the GPU’s 64-bit track ID
  G4PrimaryParticle *primary             = nullptr;
  G4VProcess *creatorProcess             = nullptr;
  G4VUserTrackInformation *userTrackInfo = nullptr;
};

// This class provides a mapping between G4 id's (int) and AdePT id's (uint64_t).
// Furthermore it holds maps (to be unified) to all the information that must be kept on the CPU
// such as the pointer to the creator process, the G4 primary particle, and the G4VUserTrackInformation
class TrackIDMapper {
public:
  // V2: HASH MAP + then VECTOR FOR DATA
  void beginEvent(int eventID, size_t expectedTracks = 20'000'000)
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

  HostTrackData &getOrCreate(uint64_t gpuId, bool useNewId = true)
  {
    auto it = gpuToIndex.find(gpuId);
    if (it != gpuToIndex.end()) {
      // already have a slot
      return hostDataVec[it->second];
    }
    // new track → assign next slot
    int idx           = static_cast<int>(hostDataVec.size());
    gpuToIndex[gpuId] = idx;
    hostDataVec.push_back({});
    auto &d = hostDataVec.back();
    d.gpuId = gpuId;
    d.g4id  = useNewId ? currentGpuReturnG4ID-- : static_cast<int>(gpuId);
    return d;
  }

  // V1: one big hash map!
  // /// Call once at the start of each event, so we can clear and reserve cheaply.
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

  // OLD STUFF
  // /// Register a G4 track ID for a specific event and assign a unique uint64_t GPU ID.
  // uint64_t registerG4Track(int g4id, int eventId, G4PrimaryParticle *primary = nullptr, G4VProcess *creatorProcess =
  // nullptr, G4VUserTrackInformation *userTrackInfo = nullptr)
  // {
  //   if (eventId != currentEventID) {
  //     reset();
  //     currentEventID = eventId;
  //   }

  //   auto it = g4ToGpu.find(g4id);
  //   if (it != g4ToGpu.end()) return it->second.gpuID;

  //   HostTrackData hostData;
  //   hostData.gpuID = generateDeterministicID(g4id);
  //   hostData.primary = primary;
  //   hostData.creatorProcess = creatorProcess;
  //   hostData.userTrackInfo = userTrackInfo;

  //   g4ToGpu[g4id] = hostData;
  //   gpuToG4[hostData.gpuID] = g4id;

  //   return hostData.gpuID;
  // }

  // std::pair<int,HostTrackData&> mapGpuToG4Data(uint64_t gpuId)
  // {
  //   // 1) Try to insert gpuId→g4id (defaults to 0 if missing)
  //   auto [itG2G, insertedG2G] = gpuToG4.try_emplace(gpuId, 0);
  //   int  g4id = itG2G->second;
  //   if (insertedG2G) {
  //     // first time we see this GPU id → assign new G4 id
  //     g4id = currentGpuReturnG4ID--;
  //     itG2G->second = g4id;
  //   }

  //   // 2) Try to insert g4id→HostTrackData (value-initialized if missing)
  //   auto [itHTD, insertedHTD] = g4ToGpu.try_emplace(g4id);
  //   if (insertedHTD) {
  //     // on first insert, fill in the only required field
  //     itHTD->second.gpuID = gpuId;
  //   }

  //   // 3) Return both the G4 id and a reference to its HostTrackData
  //   return std::pair<int,HostTrackData&>(g4id, itHTD->second);
  // }

  // /// Map a GPU ID back to a G4 ID. If not known yet, assign a new G4 ID (counting down from INT_MAX).
  // int mapGpuToG4(uint64_t gpuId)
  // {
  //   auto it = gpuToG4.find(gpuId);
  //   if (it != gpuToG4.end()) return it->second;

  //   int newId      = currentGpuReturnG4ID--;
  //   gpuToG4[gpuId] = newId;
  //   HostTrackData hostData;
  //   hostData.gpuID = gpuId;
  //   g4ToGpu[newId] = hostData;

  //   return newId;
  // }

  /// Return the primary particle associated to a G4 ID (may be nullptr)
  // G4PrimaryParticle *getPrimaryForG4ID(int g4id) const
  // {
  //   auto it = g4ToGpu.find(g4id);
  //   return (it != g4ToGpu.end()) ? it->second.primary : nullptr;
  // }

  /// If not yet set, set the PrimaryParticle pointer for given G4 trackID
  // void setPrimaryForG4ID(int g4id, G4PrimaryParticle *primary) {
  //   if (g4ToGpu[g4id].primary == nullptr) g4ToGpu[g4id].primary = primary;
  // }

  /// Return the primary particle associated to a G4 ID (may be nullptr)
  // G4VUserTrackInformation *getUserTrackInfoForG4ID(int g4id) const
  // {
  //   auto it = g4ToGpu.find(g4id);
  //   return (it != g4ToGpu.end()) ? it->second.userTrackInfo : nullptr;
  // }

  /// If not yet set, set the UserTrackInfo pointer for given G4 trackID
  // void setUserTrackInfoForG4ID(int g4id, G4VUserTrackInformation *userTrackInfo)
  // {
  //   if (g4ToGpu[g4id].userTrackInfo == nullptr) g4ToGpu[g4id].userTrackInfo = userTrackInfo;
  // }

  // G4VProcess *getCreatorProcessForG4ID(int g4id) const
  // {
  //   auto it = g4ToGpu.find(g4id);
  //   return (it != g4ToGpu.end()) ? it->second.creatorProcess : nullptr;
  // }

  /// If not yet set, set the creator process pointer for given G4 trackID
  // void setCreatorProcessForG4ID(int g4id, G4VProcess *creatorProcess) {
  //   if (g4ToGpu[g4id].creatorProcess == nullptr) g4ToGpu[g4id].creatorProcess = creatorProcess; }

private:
  // V2:
  std::unordered_map<uint64_t, int> gpuToIndex; // key→slot in hostDataVec
  std::vector<HostTrackData> hostDataVec;       // contiguous array of all data
                                                // V1
  // std::unordered_map<uint64_t,HostTrackData> gpuToHost;

  // old version using 2 maps:
  // std::unordered_map<int, HostTrackData> g4ToGpu;
  // std::unordered_map<uint64_t, int> gpuToG4;

  int currentGpuReturnG4ID = std::numeric_limits<int>::max();
  int currentEventID       = -1;

  // void reset()
  // {
  //   g4ToGpu.clear();
  //   gpuToG4.clear();
  //   // g4PrimaryMap.clear();
  //   // g4UserTrackInfoMap.clear();
  //   // g4CreatorProcessMap.clear();
  //   currentGpuReturnG4ID = std::numeric_limits<int>::max();
  // }

  static uint64_t generateDeterministicID(int id) { return static_cast<uint64_t>(id); }
};

#endif
