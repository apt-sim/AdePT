// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRANSPORT_INTERFACE_H
#define ADEPT_TRANSPORT_INTERFACE_H

#include "AdePT/integration/G4HepEmTrackingManagerSpecialized.hh"

#include "G4VPhysicalVolume.hh"
#include "VecGeom/navigation/NavigationState.h"
#include <G4HepEmConfig.hh>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

class AdePTTransportInterface {
public:
  virtual ~AdePTTransportInterface() {}

  /// @brief Adds a track to the buffer
  virtual void AddTrack(int pdg, uint64_t trackId, uint64_t parentId, double energy, double x, double y, double z,
                        double dirx, double diry, double dirz, double globalTime, double localTime, double properTime,
                        float weight, unsigned short stepCounter, int threadId, unsigned int eventId,
                        vecgeom::NavigationState &&state, vecgeom::NavigationState &&originState) = 0;

  /// @brief Set capacity of on-GPU track buffer.
  virtual void SetTrackCapacity(size_t capacity) = 0;
  /// @brief Set capacity of on-GPU leak buffer.
  virtual void SetLeakCapacity(size_t capacity) = 0;
  /// @brief Set Hit buffer capacity on GPU and Host
  virtual void SetHitBufferCapacity(size_t capacity) = 0;
  /// @brief Set maximum batch size
  virtual void SetMaxBatch(int npart) = 0;
  /// @brief Set buffer threshold
  virtual void SetBufferThreshold(int limit) = 0;
  /// @brief Set debug level for transport
  virtual void SetDebugLevel(int level) = 0;
  /// @brief Set whether AdePT should transport particles across the whole geometry
  virtual void SetTrackInAllRegions(bool trackInAllRegions) = 0;
  /// @brief Check whether AdePT should transport particles across the whole geometry
  virtual bool GetTrackInAllRegions() const = 0;
  /// @brief Set Geant4 region which are run on GPU
  virtual void SetGPURegionNames(std::vector<std::string> const *regionNames) = 0;
  /// @brief Set Geant4 region which are run on CPU
  virtual void SetCPURegionNames(std::vector<std::string> const *regionNames) = 0;
  virtual std::vector<std::string> const *GetGPURegionNames()                 = 0;
  virtual std::vector<std::string> const *GetCPURegionNames()                 = 0;
  /// @brief Set path to covfie magnetic field file
  /// @param fileName path to magnetic field file
  virtual void SetBfieldFileName(const std::string &fileName) = 0;
  virtual void SetCUDAStackLimit(int limit)                   = 0;
  virtual void SetCUDAHeapLimit(int limit)                    = 0;
  virtual void SetAdePTSeed(uint64_t seed)                    = 0;

  /// @brief Initialize service and copy geometry & physics data on device
  virtual void Initialize(G4HepEmTrackingManagerSpecialized *hepEmTM, bool common_data = false) = 0;
  /// @brief Interface for transporting a buffer of tracks in AdePT.
  virtual void Shower(int event, int threadId)            = 0;
  virtual void Cleanup(bool commonInitThread)             = 0;
  virtual void ProcessGPUSteps(int threadId, int eventId) = 0;
  /// @brief Setup function for the G4HepEmtrackingManager in the integration layer - used only in async AdePT
  /// @param threadId thread Id
  /// @param hepEmTM specialized G4HepEmTrackingManager
  virtual void SetHepEmTrackingManagerForThread(int threadId, G4HepEmTrackingManagerSpecialized *hepEmTM) = 0;
};

#endif
