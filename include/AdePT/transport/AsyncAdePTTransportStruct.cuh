// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNC_ADEPT_TRANSPORT_STRUCT_CUH
#define ASYNC_ADEPT_TRANSPORT_STRUCT_CUH

#include <AdePT/transport/AsyncAdePTTransportStruct.hh>
#include <AdePT/transport/geometry/GeometryAuxData.hh>
// #include <AdePT/transport/AsyncAdePTTransport.hh>
#include <AdePT/transport/steps/GPUStepRecording.cuh>
#include <AdePT/transport/steps/GPUStepTransferManager.cuh>
#include <AdePT/transport/tracks/Track.cuh>
#include <AdePT/transport/magneticfield/GeneralMagneticField.cuh>
#include <AdePT/transport/magneticfield/UniformMagneticField.cuh>

#include <AdePT/transport/containers/SlotManager.cuh>
#include <AdePT/transport/support/ResourceManagement.cuh>

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>

#ifdef ADEPT_USE_SPLIT_KERNELS
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmGammaTrack.hh>
#endif

namespace AsyncAdePT {

// A bundle of pointers to generate particles of an implicit type.
template <typename TrackT>
struct SpeciesParticleManager {
  TrackT *fTracks;
  SlotManager *fSlotManager;
  adept::MParray *fActiveQueue;
  adept::MParray *fNextActiveQueue;

public:
  __host__ __device__ SpeciesParticleManager(TrackT *tracks, SlotManager *slotManager, adept::MParray *activeQueue,
                                             adept::MParray *nextActiveQueue)
      : fTracks(tracks), fSlotManager(slotManager), fActiveQueue(activeQueue), fNextActiveQueue(nextActiveQueue)
  {
  }

  /// Obtain track at given slot position
  __device__ __forceinline__ TrackT &TrackAt(SlotManager::value_type slot) { return fTracks[slot]; }

  /// Obtain a slot for a track, but don't enqueue.
  __device__ auto NextSlot() { return fSlotManager->NextSlot(); }

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
  __device__ TrackT &InitTrack(SlotManager::value_type slot, Ts &&...args)
  {
    return *new (fTracks + slot) TrackT{std::forward<Ts>(args)...};
  }

  /// Obtain a slot and construct a track, forwarding args to the track constructor.
  template <typename... Ts>
  __device__ TrackT &NextTrack(Ts &&...args)
  {
    const auto slot = NextSlot();
    // next track is only visible in next GPU iteration, therefore pushed in the NextActiveQueue
    fNextActiveQueue->push_back(slot);
    auto &track = InitTrack(slot, std::forward<Ts>(args)...);
    return track;
  }
};

// A bundle of generators for the three particle types.
struct ParticleManager {
  SpeciesParticleManager<ChargedTrack> electrons;
  SpeciesParticleManager<ChargedTrack> positrons;
  SpeciesParticleManager<NeutralTrack> gammas;
  SpeciesParticleManager<NeutralTrack> gammasWDT;
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

  void SwapActive() { std::swap(initiallyActive, nextActive); }
};

/// @brief Named array-index enum for the per-species GPU state arrays in @ref SpeciesState.
///
/// Carries the three physical species plus Woodcock-tracking sentinels (GammaWDT,
/// NumParticleQueues).  For physics data (steps, tracks) use the free @ref ParticleType
/// enum class instead; a static_assert below guarantees the numeric values stay in sync.
enum GPUQueueIndex {
  Electron = 0,
  Positron = 1,
  Gamma    = 2,

  NumSpecies,
  // alias for Woodcock tracking gammas:
  // as there is no explicit Woodcock tracking species, but NumSpecies is used to loop over
  // AllParticleQueues (which contain the Woodcock tracking gammas), an alias is used here to mark their access
  GammaWDT          = NumSpecies,
  NumParticleQueues = NumSpecies + 1
};

static_assert(GPUQueueIndex::Electron == static_cast<int>(ParticleType::Electron),
              "GPUQueueIndex and ParticleType electron values must match");
static_assert(GPUQueueIndex::Positron == static_cast<int>(ParticleType::Positron),
              "GPUQueueIndex and ParticleType positron values must match");
static_assert(GPUQueueIndex::Gamma == static_cast<int>(ParticleType::Gamma),
              "GPUQueueIndex and ParticleType gamma values must match");

/// @brief Holds all GPU resources needed to manage in-flight tracks of one particle species:
///        track buffer, slot manager, interaction queues, CUDA stream and event.
template <typename TrackT>
struct SpeciesState {
  TrackT *tracks{nullptr};
  SlotManager *slotManager{nullptr};
  ParticleQueues queues{};
  ADEPT_DEVICE_API_SYMBOL(Stream_t) stream {};
  ADEPT_DEVICE_API_SYMBOL(Event_t) event {};
};

static constexpr double kRelativeQueueSize[GPUQueueIndex::NumSpecies] = {0.35, 0.15, 0.5};

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
  ChargedTrack *electrons;
  ChargedTrack *positrons;
  NeutralTrack *gammas;
  SlotManager *const slotManagers[GPUQueueIndex::NumSpecies];

  __device__ __forceinline__ short ThreadIdAt(unsigned int particleType, SlotManager::value_type slot) const
  {
    if (particleType == GPUQueueIndex::Electron) return electrons[slot].threadId;
    if (particleType == GPUQueueIndex::Positron) return positrons[slot].threadId;
    return gammas[slot].threadId;
  }
};

// A bundle of queues for the three particle types.
struct AllParticleQueues {
  // AllParticleQueues has queues for each particle type + one for Woodcock tracking
  ParticleQueues queues[GPUQueueIndex::NumParticleQueues];
};

struct AllSlotManagers {
  SlotManager slotManagers[GPUQueueIndex::NumSpecies];
};

// A data structure to transfer statistics after each iteration.
struct Stats {
  int inFlight[GPUQueueIndex::NumSpecies];
  float queueFillLevel[GPUQueueIndex::NumParticleQueues];
  float slotFillLevel[GPUQueueIndex::NumSpecies];
  unsigned int perEventInFlight[kMaxThreads];         // Updated asynchronously
  unsigned int perEventInFlightPrevious[kMaxThreads]; // Used in transport kernels
  unsigned int stepBufferOccupancy;
};

/// Host-only counters accumulating transport-loop stop/stall/flush action reasons across the full run.
/// These are incremented on the host transport thread and printed at shutdown when verbosity >= 1.
struct TransportLoopCounters {
  unsigned long long totalIterations{0};               ///< Total transport iterations executed
  unsigned long long leakExtractionByQueuePressure{0}; ///< Iterations where leak queue exceeded 50% threshold
  unsigned long long leakExtractionByEventFlush{0};    ///< Iterations where an event flush requested leak extraction
  unsigned long long leakExtractionBlocked{0};         ///< Times transport stalled waiting for in-progress extraction
  unsigned long long eventDrainedToStepFlush{0};      ///< Events that transitioned to RequestStepFlush (queues drained)
  unsigned long long stepBufferSwaps{0};              ///< Total step-buffer swaps performed
  unsigned long long stepBufferSwapByOccupancy{0};    ///< Swaps triggered by occupancy >= half capacity
  unsigned long long stepBufferSwapByOccupancy10k{0}; ///< Swaps triggered by occupancy >= 10000
  unsigned long long stepBufferSwapByPressure{0};     ///< Swaps triggered by nextStepMightFail (overflow risk)
  unsigned long long stepBufferSwapByEventFlush{0};   ///< Swaps triggered by event RequestStepFlush
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

  SpeciesState<ChargedTrack> electrons;
  SpeciesState<ChargedTrack> positrons;
  SpeciesState<NeutralTrack> gammas;

  // particle queues for gammas doing woodcock tracking. Only the `initiallyActive` and `nextActive` queues are
  // allocated.
  ParticleQueues woodcockQueues;

  std::vector<void *> allCudaPointers;
  // Create a stream to synchronize kernels of all particle types.
  ADEPT_DEVICE_API_SYMBOL(Stream_t) stream; ///< all-particle sync stream

  static constexpr unsigned int nSlotManager_dev = 3;

  AllSlotManagers allmgr_h;              // All host slot managers, statically allocated
  SlotManager *slotManager_dev{nullptr}; // All device slot managers

#ifdef ADEPT_USE_SPLIT_KERNELS
  HepEmBuffers hepEmBuffers_d; // All device buffers of hepem tracks
#endif

  Stats *stats_dev{nullptr}; ///< statistics object pointer on device
  Stats *stats{nullptr};     ///< statistics object pointer on host

  std::unique_ptr<GPUStepTransferManager> fGPUStepTransferManager;

  adept::MParrayT<QueueIndexPair> *injectionQueue;

  enum class InjectState { Idle, CreatingSlots, ReadyToEnqueue, Enqueueing };
  std::atomic<InjectState> injectState;
  std::atomic_bool runTransport{true}; ///< Keep transport thread running

  ~GPUstate()
  {
    try {
      if (stats) ADEPT_DEVICE_API_CALL(FreeHost(stats));
      if (stream) ADEPT_DEVICE_API_CALL(StreamDestroy(stream));

      auto destroySpeciesSync = [](auto &particleType) {
        if (particleType.stream) ADEPT_DEVICE_API_CALL(StreamDestroy(particleType.stream));
        if (particleType.event) ADEPT_DEVICE_API_CALL(EventDestroy(particleType.event));
      };
      destroySpeciesSync(electrons);
      destroySpeciesSync(positrons);
      destroySpeciesSync(gammas);
      for (void *ptr : allCudaPointers) {
        ADEPT_DEVICE_API_CALL(Free(ptr));
      }
    } catch (const std::exception &e) {
      std::cerr << "\033[31m" << "GPUstate::~GPUstate : Error during device API call" << "\033[0m" << std::endl;
      std::cerr << "\033[31m" << e.what() << "\033[0m" << std::endl;
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
