// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNC_ADEPT_TRANSPORT_STRUCT_CUH
#define ASYNC_ADEPT_TRANSPORT_STRUCT_CUH

#include <AdePT/core/CommonStruct.h>
#include <AdePT/core/AsyncAdePTTransportStruct.hh>
// #include <AdePT/core/AsyncAdePTTransport.hh>
#include <AdePT/core/PerEventScoringImpl.cuh>
#include <AdePT/core/Track.cuh>
#include <AdePT/magneticfield/GeneralMagneticField.cuh>
#include <AdePT/magneticfield/UniformMagneticField.cuh>

#include <AdePT/base/SlotManager.cuh>
#include <AdePT/base/ResourceManagement.cuh>

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>

#ifdef ADEPT_USE_SPLIT_KERNELS
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmGammaTrack.hh>
#endif

namespace AsyncAdePT {

// A bundle of pointers to generate particles of an implicit type.
struct SpeciesParticleManager {
  Track *fTracks;
  Track *fLeakedTracks;
  SlotManager *fSlotManager;
  SlotManager *fSlotManagerLeaks;
  adept::MParray *fActiveQueue;
  adept::MParray *fNextActiveQueue;
  adept::MParray *fActiveLeaksQueue;

public:
  __host__ __device__ SpeciesParticleManager(Track *tracks, Track *leakedTracks, SlotManager *slotManager,
                                             SlotManager *slotManagerLeaks, adept::MParray *activeQueue,
                                             adept::MParray *nextActiveQueue, adept::MParray *activeLeaksQueue)
      : fTracks(tracks), fLeakedTracks(leakedTracks), fSlotManager(slotManager), fSlotManagerLeaks(slotManagerLeaks),
        fActiveQueue(activeQueue), fNextActiveQueue(nextActiveQueue), fActiveLeaksQueue(activeLeaksQueue)
  {
  }

  /// Obtain track and leaked track at given slot position
  __device__ __forceinline__ Track &TrackAt(SlotManager::value_type slot) { return fTracks[slot]; }
  __device__ __forceinline__ Track &LeakTrackAt(SlotManager::value_type slot) { return fLeakedTracks[slot]; }

  /// Obtain a slot for a track, but don't enqueue.
  __device__ auto NextSlot() { return fSlotManager->NextSlot(); }
  __device__ auto NextLeakSlot() { return fSlotManagerLeaks->NextSlot(); }

  // enqueue into next-active queue
  __device__ __forceinline__ bool EnqueueNext(SlotManager::value_type slot)
  {
    return fNextActiveQueue->push_back(slot);
  }

  // size of the active queue
  __device__ __forceinline__ int ActiveSize() const { return fActiveQueue->size(); }

  // read slot from active queue by index
  __device__ __forceinline__ SlotManager::value_type ActiveAt(int i) const { return (*fActiveQueue)[i]; }

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
    // next track is only visible in next GPU iteration, therefore pushed in the NextActiveQueue
    fNextActiveQueue->push_back(slot);
    auto &track = InitTrack(slot, std::forward<Ts>(args)...);
    return track;
  }

  /// Obtains a leak slot, copies the track from the source slot to the leaks, and marks the slot in the active queue
  /// for freeing
  __device__ __forceinline__ void CopyTrackToLeaked(SlotManager::value_type srcSlot)
  {
    // get a leak slot
    const auto leakSlot = NextLeakSlot();

    // Create and construct track from other track
    new (fLeakedTracks + leakSlot) Track{TrackAt(srcSlot)};

    // enqueue into leak queue
    const bool success = fActiveLeaksQueue->push_back(leakSlot);
    if (!success) {
      printf("ERROR: No space left in leaks queue.\n"
             "\tThe threshold for flushing the leak buffer may be too high\n"
             "\tThe space allocated to the leak buffer may be too small\n");
      asm("trap;");
    }

    // free the source slot
    fSlotManager->MarkSlotForFreeing(srcSlot);

    return;
  }
};

struct LeakedTracks {
  Track *fTracks;
  adept::MParray *fLeakedQueue;
  SlotManager *fSlotManager;
};

// Note: deprecated GammaInteractions for split gamma kernels
// struct GammaInteractions {
//   enum Interaction : unsigned int { PairCreation = 0, ComptonScattering = 1, PhotoelectricProcess = 2, NInt };
//   struct Data {
//     double geometryStepLength;
//     double PEmxSec; // Only used for photoelectric process
//     unsigned int slot;
//     vecgeom::NavigationState preStepNavState;
//     vecgeom::Vector3D<double> preStepPos;
//     vecgeom::Vector3D<double> preStepDir;
//     double preStepEnergy;
//   };
//   adept::MParrayT<Data> *queues[Interaction::NInt];
// };

// A bundle of generators for the three particle types.
struct ParticleManager {
  SpeciesParticleManager electrons;
  SpeciesParticleManager positrons;
  SpeciesParticleManager gammas;
  SpeciesParticleManager gammasWDT;
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
  /*
Gamma interactions:
0 - Conversion
1 - Compton
2 - Photoelectric
3 - Unused
4 - Relocation

Electron interactions:
0 - Ionization
1 - Bremsstrahlung
2 - Unused
3 - Unused
4 - Relocation

Positron interactions:
0 - Ionization
1 - Bremsstrahlung
2 - In flight annihilation
3 - Stopped annihilation
4 - Relocation

In-flight and stopped annihilation use different codes but may be merged to save space
in unused queues or if launching one kernel is faster than two smaller ones

It is not straightforward to allocate just the needed queues per particle type because
ParticleQueues needs to be passed by copy to the kernels, which means that we can't do
dynamic allocations
*/
  static constexpr char numInteractions = 5;
  adept::MParray *nextActive;
  adept::MParray *initiallyActive;
#ifdef ADEPT_USE_SPLIT_KERNELS
  adept::MParray *propagation;
  adept::MParray *interactionQueues[numInteractions];
#endif
  adept::MParray *leakedTracksCurrent;
  adept::MParray *leakedTracksNext;

  void SwapActive() { std::swap(initiallyActive, nextActive); }
  void SwapLeakedQueue() { std::swap(leakedTracksCurrent, leakedTracksNext); }
};

// Holds all information needed to manage in-flight tracks of one type
struct ParticleType {
  Track *tracks;
  Track *leaks;
  SlotManager *slotManager;
  SlotManager *slotManagerLeaks;
  ParticleQueues queues;
  cudaStream_t stream;
  cudaEvent_t event;

  enum {
    Electron = 0,
    Positron = 1,
    Gamma    = 2,

    NumParticleTypes,
    // alias for Woodcock tracking gammas:
    // as there is no explicit Woodcock tracking ParticleType, but the NumParticle types is used to loop over the
    // AllParticleQueues (which contain the Woodcock tracking gammas), an alias is used here to mark their access
    GammaWDT          = NumParticleTypes,
    NumParticleQueues = NumParticleTypes + 1
  };
  static constexpr double relativeQueueSize[] = {0.35, 0.15, 0.5};
};

#ifdef ADEPT_USE_SPLIT_KERNELS
struct HepEmBuffers {
  G4HepEmElectronTrack *electronsHepEm;
  G4HepEmElectronTrack *positronsHepEm;
  G4HepEmGammaTrack *gammasHepEm;
};

// A bundle of queues per interaction type
struct AllInteractionQueues {
  adept::MParray *queues[5];
};
#endif

// Pointers to track storage for each particle type
struct TracksAndSlots {
  Track *const tracks[ParticleType::NumParticleTypes];
  Track *const leaks[ParticleType::NumParticleTypes];
  SlotManager *const slotManagers[ParticleType::NumParticleTypes];
  SlotManager *const slotManagersLeaks[ParticleType::NumParticleTypes];
};

// A bundle of queues for the three particle types.
struct AllParticleQueues {
  // AllParticleQueues has queues for each particle type + one for Woodcock tracking
  ParticleQueues queues[ParticleType::NumParticleQueues];
};

struct AllSlotManagers {
  SlotManager slotManagers[ParticleType::NumParticleTypes];
  SlotManager slotManagersLeaks[ParticleType::NumParticleTypes];
};

// A data structure to transfer statistics after each iteration.
struct Stats {
  int inFlight[ParticleType::NumParticleTypes];
  int leakedTracks[ParticleType::NumParticleTypes];
  float queueFillLevel[ParticleType::NumParticleQueues];
  float slotFillLevel[ParticleType::NumParticleTypes];
  float slotFillLevelLeaks[ParticleType::NumParticleTypes];
  unsigned int perEventInFlight[kMaxThreads];         // Updated asynchronously
  unsigned int perEventInFlightPrevious[kMaxThreads]; // Used in transport kernels
  unsigned int perEventLeaked[kMaxThreads];
  unsigned int nLeakedCurrent[ParticleType::NumParticleTypes];
  unsigned int nLeakedNext[ParticleType::NumParticleTypes];
  unsigned int hitBufferOccupancy;
};

/// @brief Array of flags whether the event can be finished off
struct AllowFinishOffEventArray {
  unsigned short flags[kMaxThreads];

  __host__ __device__ unsigned short operator[](int idx) const { return flags[idx]; }
};

struct QueueIndexPair {
  unsigned int slot;
  short queue;
};

struct GPUstate {

#ifdef ADEPT_USE_EXT_BFIELD
  // If using a general magnetic field, GPUstate will store the host-side instance
  GeneralMagneticField magneticField;
#endif

  ParticleType particles[ParticleType::NumParticleTypes];
  // GammaInteractions gammaInteractions; // Note: deprecated gammaInteractions for split gamma kernels

  // particle queues for gammas doing woodcock tracking. Only the `initiallyActive` and `nextActive` queue are
  // allocated, for the leaks the normal gamma queues inside particles[ParticleType::Gamma] are used.
  ParticleQueues woodcockQueues;

  std::vector<void *> allCudaPointers;
  // Create a stream to synchronize kernels of all particle types.
  cudaStream_t stream; ///< all-particle sync stream

  static constexpr unsigned int nSlotManager_dev = 3;

  AllSlotManagers allmgr_h;                   // All host slot managers, statically allocated
  SlotManager *slotManager_dev{nullptr};      // All device slot managers
  SlotManager *slotManagerLeaks_dev{nullptr}; // All device leak slot managers

#ifdef ADEPT_USE_SPLIT_KERNELS
  HepEmBuffers hepEmBuffers_d; // All device buffers of hepem tracks
#endif

  Stats *stats_dev{nullptr}; ///< statistics object pointer on device
  Stats *stats{nullptr};     ///< statistics object pointer on host

  PerEventScoring *fScoring_dev; ///< Device array for per-worker scoring data
  std::unique_ptr<HitScoring> fHitScoring;

  adept::MParrayT<QueueIndexPair> *injectionQueue;

  enum class InjectState { Idle, CreatingSlots, ReadyToEnqueue, Enqueueing };
  std::atomic<InjectState> injectState;
  // ExtractState:
  // Idle: No flush has been requested
  // ExtractionRequested: An event requested a flush, waiting for transport to finish
  // TracksNeedTransfer: An event requested a flush, leak buffer on device has tracks to transfer
  // PreparingTracks: Tracks are being copied to the staging buffer
  // TracksReadyToCopy: Staging buffer is ready to be copied to host
  // CopyingTracks: Tracks are being copied to host
  // TracksOnHost: Some or all the tracks have been transferred from device to host and are waiting in the copy buffer
  // SavingTracks: Tracks are being copied to per-event queues
  // TracksSaved: Tracks have been moved from the copy buffer to their respective per-event queues
  enum class ExtractState {
    Idle,
    ExtractionRequested,
    TracksNeedTransfer,
    PreparingTracks,
    TracksReadyToCopy,
    CopyingTracks,
    TracksOnHost,
    SavingTracks,
    TracksSaved
  };
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

extern __constant__ __device__ adeptint::WDTDeviceView gWDTData;

constexpr double kPush = 1.e-8 * copcore::units::cm;
#ifdef ADEPT_USE_EXT_BFIELD
__device__ GeneralMagneticField *gMagneticField = nullptr;
#else
__device__ UniformMagneticField *gMagneticField = nullptr;
#endif

} // namespace AsyncAdePT

#endif
