// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_INTEGRATION_CUH
#define ADEPT_INTEGRATION_CUH

#include "AdeptIntegration.h"

#include "Track.cuh"
#include "TrackTransfer.h"
#include "SlotManager.cuh"
#include "ResourceManagement.h"

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmRandomEngine.hh>

namespace AsyncAdePT {

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

// A bundle of pointers to generate particles of an implicit type.
struct ParticleGenerator {
  Track *fTracks;
  SlotManager *fSlotManager;
  adept::MParray *fActiveQueue;

public:
  __host__ __device__ ParticleGenerator(Track *tracks, SlotManager *slotManager, adept::MParray *activeQueue)
      : fTracks(tracks), fSlotManager(slotManager), fActiveQueue(activeQueue)
  {
  }

  __device__ Track &NextTrack()
  {
    const auto slot = fSlotManager->NextSlot();
    fActiveQueue->push_back(slot);
    return fTracks[slot];
  }

  void SetActiveQueue(adept::MParray *queue) { fActiveQueue = queue; }
};

struct LeakedTracks {
  Track *fTracks;
  adept::MParray *fLeakedQueue;
  adept::MParray *fLeakedQueueNext;
  SlotManager *fSlotManager;
};

struct GammaInteractions {
  enum Interaction : unsigned int { PairCreation = 0, ComptonScattering = 1, PhotoelectricProcess = 2, NInt };
  struct Data {
    double geometryStepLength;
    double PEmxSec; // Only used for photoelectric process
    unsigned int slot;
  };
  adept::MParrayT<Data> *queues[Interaction::NInt];
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
  adept::MParray *leakedTracksCurrent;
  adept::MParray *leakedTracksNext;

  void SwapActive() { std::swap(currentlyActive, nextActive); }
  void SwapLeakedQueue() { std::swap(leakedTracksCurrent, leakedTracksNext); }
};

struct ParticleType {
  Track *tracks;
  SlotManager slotManager_host{0, 0};
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
  static constexpr double relativeQueueSize[] = {0.35, 0.15, 0.5};
};

// Pointers to track storage for each particle type
struct TracksAndSlots {
  Track *const tracks[ParticleType::NumParticleTypes];
  SlotManager *const slotManagers[ParticleType::NumParticleTypes];
};

// A bundle of queues for the three particle types.
struct AllParticleQueues {
  ParticleQueues queues[ParticleType::NumParticleTypes];
};

// A data structure to transfer statistics after each iteration.
struct Stats {
  int inFlight[ParticleType::NumParticleTypes];
  int leakedTracks[ParticleType::NumParticleTypes];
  float queueFillLevel[ParticleType::NumParticleTypes];
  unsigned int perEventInFlight[AdeptIntegration::kMaxThreads];
  unsigned int perEventLeaked[AdeptIntegration::kMaxThreads];
  unsigned int hitBufferOccupancy;
};

struct QueueIndexPair {
  unsigned int slot;
  short queue;
};

struct GPUstate {
  ParticleType particles[ParticleType::NumParticleTypes];
  GammaInteractions gammaInteractions;

  std::vector<AsyncAdePT::unique_ptr_cuda<void>> allCudaPointers;
  // Create a stream to synchronize kernels of all particle types.
  cudaStream_t stream;                    ///< all-particle sync stream
  unsigned int fNumToDevice{8 * 16384};   ///< number of slots in the toDevice buffer
  unsigned int fNumFromDevice{4 * 16384}; ///< number of slots in the fromDevice buffer
  unique_ptr_cuda<TrackDataWithIDs> toDevice_host{nullptr, cudaHostDeleter}; ///< Tracks to be transported to the device
  unique_ptr_cuda<TrackDataWithIDs> toDevice_dev{nullptr, cudaDeleter};      ///< toDevice buffer of tracks
  unique_ptr_cuda<TrackDataWithIDs> fromDevice_host{nullptr, cudaHostDeleter}; ///< Tracks from device
  unique_ptr_cuda<TrackDataWithIDs> fromDevice_dev{nullptr, cudaDeleter};      ///< fromDevice buffer of tracks
  unique_ptr_cuda<unsigned int> nFromDevice{nullptr, cudaHostDeleter};         ///< Number of tracks collected on device

  Stats *stats_dev{nullptr}; ///< statistics object pointer on device
  Stats *stats{nullptr};     ///< statistics object pointer on host

  unique_ptr_cuda<PerEventScoring> fScoring_dev{nullptr, cudaDeleter}; ///< Device array for per-worker scoring data
  std::unique_ptr<HitScoring> fHitScoring;

  unique_ptr_cuda<adept::MParrayT<QueueIndexPair>> injectionQueue{nullptr, cudaDeleter};

  enum class InjectState { Idle, CreatingSlots, ReadyToEnqueue, Enqueueing };
  std::atomic<InjectState> injectState;
  enum class ExtractState { Idle, FreeingSlots, ReadyToCopy, CopyToHost };
  std::atomic<ExtractState> extractState;
  std::atomic_bool runTransport{true}; ///< Keep transport thread running
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

} // namespace AsyncAdePT

#endif
