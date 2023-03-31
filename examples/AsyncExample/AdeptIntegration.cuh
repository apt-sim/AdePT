// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_INTEGRATION_CUH
#define ADEPT_INTEGRATION_CUH

#include "AdeptIntegration.h"

#include "Track.cuh"

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmRandomEngine.hh>

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

// A data structure to manage slots in the track storage.
class SlotManager {
  adept::Atomic_t<int> fNextSlot;
  const int fMaxSlot;

public:
  __host__ __device__ SlotManager(int maxSlot) : fMaxSlot(maxSlot) { fNextSlot = 0; }

  __host__ __device__ int NextSlot()
  {
    int next = fNextSlot.fetch_add(1);
    if (next >= fMaxSlot) return -1;
    return next;
  }
};

// A bundle of pointers to generate particles of an implicit type.
class ParticleGenerator {
  Track *fTracks;
  SlotManager *fSlotManager;
  adept::MParray *fActiveQueue;

public:
  __host__ __device__ ParticleGenerator(Track *tracks, SlotManager *slotManager, adept::MParray *activeQueue)
      : fTracks(tracks), fSlotManager(slotManager), fActiveQueue(activeQueue)
  {
  }

  __host__ __device__ Track &NextTrack()
  {
    int slot = fSlotManager->NextSlot();
    if (slot == -1) {
      COPCORE_EXCEPTION("No slot available in ParticleGenerator::NextTrack");
    }
    fActiveQueue->push_back(slot);
    return fTracks[slot];
  }

  void SetActiveQueue(adept::MParray *queue) { fActiveQueue = queue; }
};

struct LeakedTracks {
  Track *fTracks;
  adept::MParray *fLeakedQueue;
};

// A bundle of generators for the three particle types.
struct Secondaries {
  ParticleGenerator electrons;
  ParticleGenerator positrons;
  ParticleGenerator gammas;
};

struct AllLeaked {
  LeakedTracks leakedElectrons;
  LeakedTracks leakedPositrons;
  LeakedTracks leakedGammas;
};

// A bundle of queues per particle type:
//  * Two for active particles, one for the current iteration and the second for the next.
struct ParticleQueues {
  adept::MParray *currentlyActive;
  adept::MParray *nextActive;
  adept::MParray *leakedTracks;

  void SwapActive() { std::swap(currentlyActive, nextActive); }
};

struct ParticleType {
  Track *tracks;
  SlotManager *slotManager;
  ParticleQueues queues;
  cudaStream_t stream;
  cudaEvent_t event;

  enum {
    Electron = 0,
    Positron = 1,
    Gamma    = 2,

    NumParticleTypes,
  };
};

// A bundle of queues for the three particle types.
struct AllParticleQueues {
  ParticleQueues queues[ParticleType::NumParticleTypes];
};

// A data structure to transfer statistics after each iteration.
struct Stats {
  int inFlight[ParticleType::NumParticleTypes];
  int leakedTracks[ParticleType::NumParticleTypes];
};

struct GPUstate {
  using TrackData = adeptint::TrackData;

  ParticleType particles[ParticleType::NumParticleTypes];
  // Create a stream to synchronize kernels of all particle types.
  cudaStream_t stream;                ///< all-particle sync stream
  int fNumFromDevice{0};              ///< number of tracks in the fromDevice buffer
  TrackData *toDevice_dev{nullptr};   ///< toDevice buffer of tracks
  TrackData *fromDevice_dev{nullptr}; ///< fromDevice buffer of tracks
  Stats *stats_dev{nullptr};          ///< statistics object pointer on device
  Stats *stats{nullptr};              ///< statistics object pointer on host
};

// Constant data structures from G4HepEm accessed by the kernels.
// (defined in TestEm3.cu)
extern __constant__ __device__ struct G4HepEmParameters g4HepEmPars;
extern __constant__ __device__ struct G4HepEmData g4HepEmData;

// Pointer for array of volume auxiliary data on device
extern __constant__ __device__ adeptint::VolAuxData *gVolAuxData;

// constexpr float BzFieldValue = 0.1 * copcore::units::tesla;
extern __constant__ __device__ double BzFieldValue;
constexpr double kPush = 1.e-8 * copcore::units::cm;

#endif
