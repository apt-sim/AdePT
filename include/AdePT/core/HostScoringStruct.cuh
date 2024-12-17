// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef HOSTSCORING_H
#define HOSTSCORING_H

#include <AdePT/core/ScoringCommons.hh>
// #include "VecGeom/navigation/NavigationState.h"
#include <AdePT/base/Atomic.h>
#include <G4ios.hh>


// Contains the necessary information for recording hits on GPU, and reconstructing them
// on the host, in order to call the user-defined sensitive detector code
struct HostScoring {

  /// @brief The data in this struct is copied from device to host after each iteration
  struct Stats {
    unsigned int fUsedSlots;   ///< Number of used hit slots
    unsigned int fNextFreeHit; ///< Index of last used hit slot in the buffer
    unsigned int fBufferStart; ///< Index of first used hit slot in the buffer
  };

  struct Iterator {
    std::size_t counter;
    std::size_t const modulus;
    GPUHit *const storage;

    GPUHit &operator*() { return storage[counter % modulus]; }
    Iterator &operator++()
    {
      counter++;
      return *this;
    }
    Iterator operator++(int)
    {
      Iterator result = *this;
      counter++;
      return result;
    }
    bool operator!=(Iterator const &other)
    {
      return counter != other.counter || modulus != other.modulus || storage != other.storage;
    }
  };

  HostScoring(unsigned int aBufferCapacity = 1024 * 1024, float aFlushLimit = 0.8)
      : fBufferCapacity(aBufferCapacity), fFlushLimit(aFlushLimit)
  {
    printf("Initializing scoring with buffer capacity: %d\n", aBufferCapacity);
    // Allocate the hits buffer on Host
    fGPUHitsBuffer_host = (GPUHit *)malloc(sizeof(GPUHit) * fBufferCapacity);
    // Allocate the global counters struct on host
    fGlobalCounters_host = (GlobalCounters *)malloc(sizeof(GlobalCounters));
  };

  ~HostScoring()
  {
    free(fGPUHitsBuffer_host);
    free(fGlobalCounters_host);
  }

  Iterator begin() const { return Iterator{fStats.fBufferStart, fBufferCapacity, fGPUHitsBuffer_host}; }
  Iterator end() const
  {
    return Iterator{fStats.fBufferStart + fStats.fUsedSlots, fBufferCapacity, fGPUHitsBuffer_host};
  }

  /// @brief Print scoring info
  void Print();

  // Data members
  unsigned int fBufferCapacity{0}; ///< Number of hits to be stored in the buffer
  float fFlushLimit{0};            ///< Proportion of the buffer that needs to be filled to trigger a flush to CPU
  unsigned int fBufferStart{0};    ///< Index of first used slot in the buffer
  GPUHit *fGPUHitsBuffer_dev{nullptr};
  GPUHit *fGPUHitsBuffer_host{nullptr};

  // Atomic variables used on GPU
  adept::Atomic_t<unsigned int> *fUsedSlots_dev;   ///< Number of used hit slots
  adept::Atomic_t<unsigned int> *fNextFreeHit_dev; ///< Index of last used hit slot in the buffer

  // Stats struct, used to transfer information about the state of the buffer
  Stats fStats;
  Stats *fStats_dev;

  // Used to get performance information
  unsigned int fStepsSinceLastFlush{0};

  // Used for comparison with Geant4
  GlobalCounters *fGlobalCounters_dev{nullptr};
  GlobalCounters *fGlobalCounters_host{nullptr};
};

using AdeptScoring = HostScoring;

#endif // HOSTSCORING_H