// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNC_ADEPT_TRANSPORT_STRUCT_CUH
#define ASYNC_ADEPT_TRANSPORT_STRUCT_CUH

#include <AdePT/core/CommonStruct.h>
#include <AdePT/core/AsyncAdePTTransportStruct.hh>
// #include <AdePT/core/AsyncAdePTTransport.hh>
#include <AdePT/core/PerEventScoringImpl.cuh>
#include "AsyncTrack.cuh"
#include <AdePT/base/SlotManager.cuh>
#include <AdePT/base/ResourceManagement.cuh>

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

  /// Obtain a slot for a track, but don't enqueue.
  __device__ auto NextSlot() { return fSlotManager->NextSlot(); }

  /// Construct a track at the given location, forwarding all arguments to the constructor.
  template <typename... Ts>
  __device__ Track &InitTrack(SlotManager::value_type slot, Ts &&...args)
  {
    return *new (fTracks + slot) Track{std::forward<Ts>(args)...};
  }

  /// Obtain a slot and construct a track, forwarding args to the track constructor.
  template <typename... Ts>
  __device__ Track &NextTrack(Ts &&...args)
  {
    const auto slot = NextSlot();
    fActiveQueue->push_back(slot);
    auto &track = InitTrack(slot, std::forward<Ts>(args)...);
    return track;
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
    vecgeom::NavigationState preStepNavState;
    vecgeom::Vector3D<Precision> preStepPos;
    vecgeom::Vector3D<Precision> preStepDir;
    double preStepEnergy;
  };
  adept::MParrayT<Data> *queues[Interaction::NInt];
};

// A bundle of generators for the three particle types.
struct Secondaries {
  ParticleGenerator electrons;
  ParticleGenerator positrons;
  ParticleGenerator gammas;
};

// Holds the leaked track structs for all three particle types
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

// Holds all information needed to manage in-flight tracks of one type
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
  float slotFillLevel;
  unsigned int perEventInFlight[kMaxThreads];
  unsigned int perEventLeaked[kMaxThreads];
  unsigned int hitBufferOccupancy;
};

struct QueueIndexPair {
  unsigned int slot;
  short queue;
};

struct GPUstate {
  ParticleType particles[ParticleType::NumParticleTypes];
  GammaInteractions gammaInteractions;

  std::vector<void *> allCudaPointers;
  // Create a stream to synchronize kernels of all particle types.
  cudaStream_t stream; ///< all-particle sync stream

  static constexpr unsigned int nSlotManager_dev = 1;
  SlotManager slotManager_host;
  SlotManager *slotManager_dev{nullptr};
  Stats *stats_dev{nullptr}; ///< statistics object pointer on device
  Stats *stats{nullptr};     ///< statistics object pointer on host

  PerEventScoring *fScoring_dev; ///< Device array for per-worker scoring data
  std::unique_ptr<HitScoring> fHitScoring;

  adept::MParrayT<QueueIndexPair> *injectionQueue;

  enum class InjectState { Idle, CreatingSlots, ReadyToEnqueue, Enqueueing };
  std::atomic<InjectState> injectState;
  enum class ExtractState { Idle, FreeingSlots, ReadyToCopy, CopyToHost };
  std::atomic<ExtractState> extractState;
  std::atomic_bool runTransport{true}; ///< Keep transport thread running

  ~GPUstate()
  {
    if (stats) COPCORE_CUDA_CHECK(cudaFreeHost(stats));
    if (stream) COPCORE_CUDA_CHECK(cudaStreamDestroy(stream));

    for (ParticleType &particleType : particles) {
      if (particleType.stream) COPCORE_CUDA_CHECK(cudaStreamDestroy(particleType.stream));
      if (particleType.event) COPCORE_CUDA_CHECK(cudaEventDestroy(particleType.event));
    }
    for (void *ptr : allCudaPointers) {
      COPCORE_CUDA_CHECK(cudaFree(ptr));
    }
    allCudaPointers.clear();
  }
};

// Implementation of the GPUstate deleter
void GPUstateDeleter::operator()(GPUstate *ptr)
{
  delete ptr;
}

// Constant data structures from G4HepEm accessed by the kernels.
// (defined in TestEm3.cu)
extern __constant__ __device__ struct G4HepEmParameters g4HepEmPars;
extern __constant__ __device__ struct G4HepEmData g4HepEmData;

// Pointer for array of volume auxiliary data on device
extern __constant__ __device__ adeptint::VolAuxData *gVolAuxData;

// constexpr float BzFieldValue = 0.1 * copcore::units::tesla;
extern __constant__ __device__ double BzFieldValue;
extern __constant__ __device__ bool ApplyCuts;
constexpr double kPush = 1.e-8 * copcore::units::cm;

} // namespace AsyncAdePT

#endif
