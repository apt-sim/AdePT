#ifndef TRACKIDMAPPER_H
#define TRACKIDMAPPER_H

#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <limits>

#include "G4VUserTrackInformation.hh"
#include "G4VProcess.hh"

// Forward declaration
class G4PrimaryParticle;

// This class provides a mapping between G4 id's (int) and AdePT id's (uint64_t). 
// Furthermore it holds maps (to be unified) to all the information that must be kept on the CPU
// such as the pointer to the creator process, the G4 primary particle, and the G4VUserTrackInformation
class TrackIDMapper {
public:
  /// Register a G4 track ID for a specific event and assign a unique uint64_t GPU ID.
  uint64_t registerG4Track(int g4id, int eventId) {
    if (eventId != currentEventID) {
      reset();
      currentEventID = eventId;
    }

    auto it = g4ToGpu.find(g4id);
    if (it != g4ToGpu.end()) return it->second;

    uint64_t gpuId = generateDeterministicID(g4id);
    g4ToGpu[g4id] = gpuId;
    gpuToG4[gpuId] = g4id;

   return gpuId;
  }

  /// Map a GPU ID back to a G4 ID. If not known yet, assign a new G4 ID (counting down from INT_MAX).
  int mapGpuToG4(uint64_t gpuId) {
    auto it = gpuToG4.find(gpuId);
    if (it != gpuToG4.end()) return it->second;

    int newId = currentGpuReturnG4ID--;
    gpuToG4[gpuId] = newId;
    g4ToGpu[newId] = gpuId;

    return newId;
  }

  /// Get the corresponding GPU ID for a given G4 ID (or 0 if not found)
  uint64_t g4ToGpuID(int g4id) const {
    auto it = g4ToGpu.find(g4id);
    return (it != g4ToGpu.end()) ? it->second : 0;
  }

  /// Get the corresponding G4 ID for a given GPU ID (or -1 if not found)
  int gpuToG4ID(uint64_t gpuId) const {
    auto it = gpuToG4.find(gpuId);
    return (it != gpuToG4.end()) ? it->second : -1;
  }

  /// Return the primary particle associated to a G4 ID (may be nullptr)
  G4PrimaryParticle* getPrimaryForG4ID(int g4id) const {
    auto it = g4PrimaryMap.find(g4id);
    return (it != g4PrimaryMap.end()) ? it->second : nullptr;
  }

  /// Set the PrimaryParticle pointer for given G4 trackID
  void setPrimaryForG4ID(int g4id, G4PrimaryParticle* primary) {
    g4PrimaryMap[g4id] = primary;
  }

  /// Return the primary particle associated to a G4 ID (may be nullptr)
  G4VUserTrackInformation* getUserTrackInfoForG4ID(int g4id) const {
    auto it = g4UserTrackInfoMap.find(g4id);
    return (it != g4UserTrackInfoMap.end()) ? it->second : nullptr;
  }

    /// Set the UserTrackInfo pointer for given G4 trackID
  void setUserTrackInfoForG4ID(int g4id, G4VUserTrackInformation* userTrackInfo) {
    g4UserTrackInfoMap[g4id] = userTrackInfo;
  }

  G4VProcess* getCreatorProcessForG4ID(int g4id) const {
    auto it = g4CreatorProcessMap.find(g4id);
    return (it != g4CreatorProcessMap.end()) ? it->second : nullptr;
  }

    /// Set the creator process pointer for given G4 trackID
  void setCreatorProcessForG4ID(int g4id, G4VProcess* creatorProcess) {
    g4CreatorProcessMap[g4id] = creatorProcess;
  }

private:
  std::unordered_map<int, uint64_t> g4ToGpu;
  std::unordered_map<uint64_t, int> gpuToG4;
  std::unordered_map<int, G4PrimaryParticle*> g4PrimaryMap;
  std::unordered_map<int, G4VUserTrackInformation*> g4UserTrackInfoMap;
  std::unordered_map<int, G4VProcess*> g4CreatorProcessMap;

  int currentGpuReturnG4ID = std::numeric_limits<int>::max();
  int currentEventID = -1;

  void reset() {
    g4ToGpu.clear();
    gpuToG4.clear();
    g4PrimaryMap.clear();
    g4UserTrackInfoMap.clear();
    g4CreatorProcessMap.clear();
    currentGpuReturnG4ID = std::numeric_limits<int>::max();
  }

  static uint64_t generateDeterministicID(int id) {
    return static_cast<uint64_t>(id);
  }
};

#endif
