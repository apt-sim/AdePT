// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNC_ADEPT_TRANSPORT_CUH
#define ASYNC_ADEPT_TRANSPORT_CUH

#include <AdePT/core/AsyncAdePTTransportStruct.cuh>
#include <AdePT/core/AsyncAdePTTransportStruct.hh>
#include <AdePT/core/CommonStruct.h>
#include <AdePT/core/AdePTConfiguration.hh>
#include <AdePT/base/Atomic.h>
#include <AdePT/base/MParray.h>
#include <AdePT/copcore/Global.h>
#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/copcore/Ranluxpp.h>

#ifdef USE_SPLIT_KERNELS
#include <AdePT/kernels/electrons_split.cuh>
#include <AdePT/kernels/gammas_split.cuh>
#else
#include <AdePT/kernels/electrons.cuh>
#include <AdePT/kernels/gammas.cuh>
#endif
#include <AdePT/core/TrackDebug.cuh>
// deprecated kernels that split the gamma interactions:
// #include <AdePT/kernels/electrons_async.cuh>
// #include <AdePT/kernels/gammas_async.cuh>

#include <AdePT/navigation/BVHNavigator.h>
#include <AdePT/integration/AdePTGeant4Integration.hh>

// #include <AdePT/benchmarking/NVTX.h>

#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif
#ifdef ADEPT_USE_SURF
#include <VecGeom/surfaces/cuda/BrepCudaManager.h>
#endif

#include <G4HepEmData.hh>
#include <G4HepEmState.hh>
#include <G4HepEmStateInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmParametersInit.hh>
#include <G4HepEmMaterialInit.hh>
#include <G4HepEmElectronInit.hh>
#include <G4HepEmGammaInit.hh>

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <vector>
#include <numeric>
// #include <cuda/std/numeric>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <cstdlib>
#include <cstdint>

using namespace AsyncAdePT;

/// Communication with the hit processing thread.
struct HitProcessingContext {
  cudaStream_t hitTransferStream;
  std::condition_variable cv{};
  std::mutex mutex{};
  std::atomic_bool keepRunning = true;
};

/// @brief Init track id for debugging to the number read from ADEPT_DEBUG_TRACK environment variable
/// The default debug step range is 0 - 10000, and can be changed via ADEPT_DEBUG_MINSTEP/ADEPT_DEBUG_MAXSTEP
/// @return Debugging enabled
bool InitializeTrackDebug()
{
  TrackDebug debug;
  char *end;
  const char *env_evt = std::getenv("ADEPT_DEBUG_EVENT");
  if (env_evt) {
    debug.event_id = std::strtoull(env_evt, &end, 0);
    if (*end != '\0') debug.event_id = -1;
  }
  const char *env_trk = std::getenv("ADEPT_DEBUG_TRACK");
  if (env_trk == nullptr) return false;
  debug.track_id = std::strtoull(env_trk, &end, 0);
  if (*end != '\0') return false;
  if (debug.track_id == 0) return false;
  debug.active = true;

  const char *env_minstep = std::getenv("ADEPT_DEBUG_MINSTEP");
  if (env_minstep) {
    debug.min_step = std::strtol(env_minstep, &end, 0);
    if (*end != '\0') debug.min_step = 0;
  }

  const char *env_maxstep = std::getenv("ADEPT_DEBUG_MAXSTEP");
  if (env_maxstep) {
    debug.max_step = std::strtol(env_maxstep, &end, 0);
    if (*end != '\0') debug.max_step = 10000;
  }

  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(gTrackDebug, &debug, sizeof(TrackDebug)));
#if ADEPT_DEBUG_TRACK > 0
  printf("=== Track debugging enabled: event %d track %lu steps %ld - %ld\n", debug.event_id, debug.track_id,
         debug.min_step, debug.max_step);
#endif
  return true;
}

// Kernel to initialize the set of queues per particle type.
__global__ void InitParticleQueues(ParticleQueues queues, size_t CapacityTransport, size_t CapacityLeaked)
{
  adept::MParray::MakeInstanceAt(CapacityTransport, queues.currentlyActive);
  adept::MParray::MakeInstanceAt(CapacityTransport, queues.nextActive);
  adept::MParray::MakeInstanceAt(CapacityTransport, queues.reachedInteraction);
  for (int i = 0; i < ParticleQueues::numInteractions; i++) {
    adept::MParray::MakeInstanceAt(CapacityTransport, queues.interactionQueues[i]);
  }
  adept::MParray::MakeInstanceAt(CapacityLeaked, queues.leakedTracksCurrent);
  adept::MParray::MakeInstanceAt(CapacityLeaked, queues.leakedTracksNext);
}

// Init a queue at the designated location
template <typename T>
__global__ void InitQueue(adept::MParrayT<T> *queue, size_t Capacity)
{
  adept::MParrayT<T>::MakeInstanceAt(Capacity, queue);
}

// Use the 64-bit MurmurHash3 finalizer for avalanche behaviour so that every input bit can influence every output bit
__device__ inline uint64_t murmur3_64(uint64_t x)
{
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

/// Generate a “random” uint64_t ID on the device by combining:
///   • base = initialSeed * eventId + trackId
///   • eKin (double)
///   • globalTime (double)
__device__ inline uint64_t GenerateSeed(uint64_t base, double eKin, double globalTime)
{
  // 1) grab the raw bit‐patterns of the doubles
  uint64_t eb = __double_as_longlong(eKin);
  uint64_t tb = __double_as_longlong(globalTime);

  // 2) mix them in boost::hash_combine style
  uint64_t seed = base;
  seed ^= eb + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
  seed ^= tb + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);

  // 3) final avalanche
  return murmur3_64(seed);
}

// Kernel function to initialize tracks comming from a Geant4 buffer
__global__ void InitTracks(AsyncAdePT::TrackDataWithIDs *trackinfo, int ntracks, Secondaries secondaries,
                           const vecgeom::VPlacedVolume *world, adept::MParrayT<QueueIndexPair> *toBeEnqueued,
                           uint64_t initialSeed)
{
  // constexpr double tolerance = 10. * vecgeom::kTolerance;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ntracks; i += blockDim.x * gridDim.x) {
    ParticleGenerator *generator = nullptr;
    const auto &trackInfo        = trackinfo[i];
    short queueIndex             = -1;
    switch (trackInfo.pdg) {
    case 11:
      generator  = &secondaries.electrons;
      queueIndex = ParticleType::Electron;
      break;
    case -11:
      generator  = &secondaries.positrons;
      queueIndex = ParticleType::Positron;
      break;
    case 22:
      generator  = &secondaries.gammas;
      queueIndex = ParticleType::Gamma;
    };
    assert(generator != nullptr && "Unsupported pdg type");

    // TODO: Delay when not enough slots?
    const auto slot = generator->NextSlot();
    // we need to scramble the initial seed with some more trackinfo to generate a unique seed. otherwise, if a particle
    // returns from the device and is injected again, it would have the same random number state
    auto seed = GenerateSeed(initialSeed * trackInfo.eventId + trackInfo.trackId, trackInfo.eKin, trackInfo.globalTime);
    Track &track = generator->InitTrack(
        slot, seed, trackInfo.eKin, trackInfo.vertexEkin, trackInfo.globalTime, static_cast<float>(trackInfo.localTime),
        static_cast<float>(trackInfo.properTime), trackInfo.weight, trackInfo.position, trackInfo.direction,
        trackInfo.vertexPosition, trackInfo.vertexMomentumDirection, trackInfo.eventId, trackInfo.parentId,
        trackInfo.threadId);
    track.navState.Clear();
    track.navState       = trackinfo[i].navState;
    track.originNavState = trackinfo[i].originNavState;
    toBeEnqueued->push_back(QueueIndexPair{slot, queueIndex});

    // FIXME KEEP OLD IMPLEMENTATION SINCE THIS HAS NOT BEEN TESTED THOROUGLY, THEN REMOVE
    //     // We locate the pushed point because we run the risk that the
    //     // point is not located in the GPU region
    // #ifdef NDEBUG
    //     constexpr int maxAttempt = 2;
    // #else
    //     constexpr int maxAttempt = 10;
    // #endif
    //     for (int attempt = 1; attempt < maxAttempt; ++attempt) {
    //       const auto amount = attempt < 5 ? attempt : (attempt - 5) * -1;
    //       track.navState.Clear();
    //       const auto pushedPosition = track.pos + amount * tolerance * track.dir;
    //       BVHNavigator::LocatePointIn(world, pushedPosition, track.navState, true);
    //       // The track must be on boundary at this point
    //       track.navState.SetBoundaryState(true);
    //       // nextState is initialized as needed.
    // #ifndef NDEBUG
    //       const vecgeom::VPlacedVolume *volume = track.navState.Top();
    //       int lvolID                           = volume->GetLogicalVolume()->id();
    //       adeptint::VolAuxData const &auxData  = AsyncAdePT::gVolAuxData[lvolID];
    //       if (auxData.fGPUregion && attempt == 1) {
    //         break;
    //       } else {
    //         printf("Error [%d, %d]: ev=%d track=%d thread=%d: GPUregion=%d volID=%d "
    //                "x=(%18.15f, %18.15f, %18.15f) dir=(%f, %f, %f) "
    //                "Safety=%17.15f DistanceToOut=%f shiftAmount=%d\n",
    //                blockIdx.x, threadIdx.x, trackInfo.eventId, trackInfo.trackId, trackInfo.threadId,
    //                auxData.fGPUregion, volume->id(), pushedPosition[0], pushedPosition[1], pushedPosition[2],
    //                track.dir[0], track.dir[1], track.dir[2], BVHNavigator::ComputeSafety(pushedPosition,
    //                track.navState), volume->DistanceToOut(track.pos, track.dir), amount);
    //         track.navState.Print();
    //         if (auxData.fGPUregion) {
    //           printf("Success in attempt %d shiftAmount %d\n", attempt, amount);
    //           break;
    //         }
    //       }
    // #endif
    //     }
  }
}

__global__ void EnqueueTracks(AllParticleQueues allQueues, adept::MParrayT<QueueIndexPair> *toBeEnqueued)
{
  const auto end = toBeEnqueued->size();
  for (unsigned int i = threadIdx.x; i < end; i += blockDim.x) {
    const auto [slotNumber, particleType] = (*toBeEnqueued)[i];
    allQueues.queues[particleType].nextActive->push_back(slotNumber);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    toBeEnqueued->clear();
  }
}

__device__ unsigned int nFromDevice_dev;
__device__ unsigned int nRemainingLeaks_dev;

// Copy particles leaked from the GPU region into a compact buffer
__global__ void FillFromDeviceBuffer(AllLeaked all, AsyncAdePT::TrackDataWithIDs *fromDevice,
                                     unsigned int maxFromDeviceBuffer, unsigned int nAlreadyTransferred)
{
  const auto numElectrons = all.leakedElectrons.fLeakedQueue->size();
  const auto numPositrons = all.leakedPositrons.fLeakedQueue->size();
  const auto numGammas    = all.leakedGammas.fLeakedQueue->size();
  const auto total        = numElectrons + numPositrons + numGammas - nAlreadyTransferred;
  const auto nToCopy      = total < maxFromDeviceBuffer ? total : maxFromDeviceBuffer;

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // Update the number of particles that will be copied in this iteration
    nFromDevice_dev     = nToCopy;
    nRemainingLeaks_dev = total - nToCopy;
  }

  for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x + nAlreadyTransferred;
       i < nAlreadyTransferred + maxFromDeviceBuffer; i += blockDim.x * gridDim.x) {
    LeakedTracks *leakedTracks = nullptr;
    unsigned int queueSlot     = 0;
    int pdg                    = 0;

    if (i < numGammas) {
      leakedTracks = &all.leakedGammas;
      queueSlot    = i;
      pdg          = 22;
    } else if (i < numGammas + numElectrons) {
      leakedTracks = &all.leakedElectrons;
      queueSlot    = i - numGammas;
      pdg          = 11;
    } else {
      leakedTracks = &all.leakedPositrons;
      queueSlot    = i - numGammas - numElectrons;
      pdg          = -11;
    }

    const auto trackSlot     = (*leakedTracks->fLeakedQueue)[queueSlot];
    Track const *const track = leakedTracks->fTracks + trackSlot;

    if (nAlreadyTransferred <= i && i < nAlreadyTransferred + nToCopy) {
      // Offset i by nAlreadyTransferred to get the index in the fromDevice buffer
      auto idx = i - nAlreadyTransferred;

      // NOTE: Sync transport copies data into trackData structs during transport.
      // Async transport stores the slots and copies to trackdata structs for transfer to
      // host here. These approaches should be unified.
      fromDevice[idx].position[0]      = track->pos[0];
      fromDevice[idx].position[1]      = track->pos[1];
      fromDevice[idx].position[2]      = track->pos[2];
      fromDevice[idx].direction[0]     = track->dir[0];
      fromDevice[idx].direction[1]     = track->dir[1];
      fromDevice[idx].direction[2]     = track->dir[2];
      fromDevice[idx].eKin             = track->eKin;
      fromDevice[idx].globalTime       = track->globalTime;
      fromDevice[idx].localTime        = track->localTime;
      fromDevice[idx].properTime       = track->properTime;
      fromDevice[idx].weight           = track->weight;
      fromDevice[idx].pdg              = pdg;
      fromDevice[idx].eventId          = track->eventId;
      fromDevice[idx].threadId         = track->threadId;
      fromDevice[idx].navState         = track->navState;
      fromDevice[idx].originNavState   = track->originNavState;
      fromDevice[idx].leakStatus       = track->leakStatus;
      fromDevice[idx].parentId         = track->parentId;
      fromDevice[idx].trackId          = track->trackId;
      fromDevice[idx].creatorProcessId = track->creatorProcessId;
      fromDevice[idx].stepCounter      = track->stepCounter;

      leakedTracks->fSlotManager->MarkSlotForFreeing(trackSlot);
    }
  }
}

template <typename... Ts>
__global__ void FreeSlots1(Ts... slotManagers)
{
  (slotManagers->FreeMarkedSlotsStage1(), ...);
}

template <typename... Ts>
__global__ void FreeSlots2(Ts... slotManagers)
{
  (slotManagers->FreeMarkedSlotsStage2(), ...);
}

// Finish iteration: clear queues and fill statistics.
__global__ void FinishIteration(AllParticleQueues all, Stats *stats, TracksAndSlots tracksAndSlots)
//, GammaInteractions gammaInteractions) // Note: deprecated gammaInteractions
{
  if (blockIdx.x == 0) {
    // Clear queues and write statistics
    for (int i = threadIdx.x; i < ParticleType::NumParticleTypes; i += blockDim.x) {
      all.queues[i].currentlyActive->clear();
      all.queues[i].reachedInteraction->clear();
      for (int j = 0; j < ParticleQueues::numInteractions; j++) {
        all.queues[i].interactionQueues[j]->clear();
      }
      stats->inFlight[i]       = all.queues[i].nextActive->size();
      stats->leakedTracks[i]   = all.queues[i].leakedTracksCurrent->size() + all.queues[i].leakedTracksNext->size();
      stats->queueFillLevel[i] = float(all.queues[i].nextActive->size()) / all.queues[i].nextActive->max_size();
    }
  } else if (blockIdx.x == 1 && threadIdx.x == 0) {
    // Assert that there are enough slots allocated:
    unsigned int particlesInFlight = 0;
    unsigned int occupiedSlots     = 0;

    for (int i = 0; i < ParticleType::NumParticleTypes; ++i) {
      particlesInFlight += all.queues[i].nextActive->size();
      occupiedSlots += tracksAndSlots.slotManagers[i]->OccupiedSlots();
      stats->slotFillLevel[i] = tracksAndSlots.slotManagers[i]->FillLevel();
    }
    if (particlesInFlight > occupiedSlots) {
      printf("Error: %d in flight while %d slots allocated\n", particlesInFlight, occupiedSlots);
      asm("trap;");
    }
  } else if (blockIdx.x == 2) {
    // Note: deprecated gammainteractions
    // if (threadIdx.x < gammaInteractions.NInt) {
    //   gammaInteractions.queues[threadIdx.x]->clear();
    // }
    if (threadIdx.x == 0) {
      // Note: hitBufferOccupancy gives the maximum occupancy of all threads combined
      stats->hitBufferOccupancy = AsyncAdePT::gHitScoringBuffer_dev.GetMaxSlotCount();
    }
  }

#if false
  if (blockIdx.x == 3) {
    for (unsigned int i = threadIdx.x; i < 3; ++i) {
      if (tracksAndSlots.slotManagers[i]->OccupiedSlots() < all.queues[i].nextActive->size())
        printf("Error: For particle %d, %d slots are allocated for %ld in flight\n", i,
               tracksAndSlots.slotManagers[i]->OccupiedSlots(), all.queues[i].nextActive->size());
    }

    if (threadIdx.x == 4) {
      printf("In flight (kernel): %ld %ld %ld  %ld\tslots: %d %d %d\n", all.queues[0].nextActive->size(),
             all.queues[1].nextActive->size(), all.queues[2].nextActive->size(),
             all.queues[0].nextActive->size() + all.queues[1].nextActive->size() + all.queues[2].nextActive->size(),
             tracksAndSlots.slotManagers[0]->OccupiedSlots(), tracksAndSlots.slotManagers[1]->OccupiedSlots(),
             tracksAndSlots.slotManagers[2]->OccupiedSlots());
    }
  }
#endif
}

__global__ void ZeroEventCounters(Stats *stats)
{
  constexpr auto size = std::extent<decltype(stats->perEventInFlight)>::value;
  for (unsigned int i = threadIdx.x; i < size; i += blockDim.x) {
    stats->perEventInFlight[i] = 0;
    stats->perEventLeaked[i]   = 0;
  }
  for (unsigned int i = threadIdx.x; i < ParticleType::NumParticleTypes; i += blockDim.x) {
    stats->nLeakedCurrent[i] = 0;
    stats->nLeakedNext[i]    = 0;
  }
}

/**
 * Count how many tracks are currently in flight for each event.
 */
__global__ void CountCurrentPopulation(AllParticleQueues all, Stats *stats, TracksAndSlots tracksAndSlots)
{
  constexpr unsigned int N = kMaxThreads;
  __shared__ unsigned int sharedCount[N];

  for (unsigned int particleType = blockIdx.x; particleType < ParticleType::NumParticleTypes;
       particleType += gridDim.x) {
    Track const *const tracks   = tracksAndSlots.tracks[particleType];
    adept::MParray const *queue = all.queues[particleType].currentlyActive;

    for (unsigned int i = threadIdx.x; i < N; i += blockDim.x)
      sharedCount[i] = 0;

    __syncthreads();

    const auto end = queue->size();
    for (unsigned int i = threadIdx.x; i < end; i += blockDim.x) {
      const auto slot     = (*queue)[i];
      const auto threadId = tracks[slot].threadId;
      atomicAdd(sharedCount + threadId, 1u);
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < N; i += blockDim.x)
      atomicAdd(stats->perEventInFlight + i, sharedCount[i]);

    __syncthreads();
  }
}

/**
 * Count tracks both in the current and the future queue of leaked particles.
 */
__global__ void CountLeakedTracks(AllParticleQueues all, Stats *stats, TracksAndSlots tracksAndSlots)
{
  constexpr auto nQueue = 2 * ParticleType::NumParticleTypes;
  // One block processes each queue
  for (unsigned int queueIndex = blockIdx.x; queueIndex < nQueue; queueIndex += gridDim.x) {
    const auto particleType =
        queueIndex < ParticleType::NumParticleTypes ? queueIndex : queueIndex - ParticleType::NumParticleTypes;
    Track const *const tracks = tracksAndSlots.tracks[particleType];
    auto const queue = queueIndex < ParticleType::NumParticleTypes ? all.queues[particleType].leakedTracksCurrent
                                                                   : all.queues[particleType].leakedTracksNext;
    const auto size  = queue->size();
    for (unsigned int i = threadIdx.x; i < size; i += blockDim.x) {
      const auto slot     = (*queue)[i];
      const auto threadId = tracks[slot].threadId;
      atomicAdd(stats->perEventLeaked + threadId, 1u);
    }

    // Update the global usage
    if (threadIdx.x == 0) {
      queueIndex < ParticleType::NumParticleTypes ? stats->nLeakedCurrent[particleType] = size
                                                  : stats->nLeakedNext[particleType]    = size;
    }
  }
}

template <typename... Args>
__global__ void ClearQueues(Args *...queue)
{
  (queue->clear(), ...);
}

__global__ void ClearAllQueues(AllParticleQueues all)
{
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    all.queues[i].currentlyActive->clear();
    all.queues[i].nextActive->clear();
    all.queues[i].leakedTracksCurrent->clear();
    all.queues[i].leakedTracksNext->clear();
  }
}

__global__ void InitSlotManagers(SlotManager *mgr, std::size_t N)
{
  for (int i = 0; i < N; ++i) {
    mgr[i].Clear();
  }
}

/// WIP: Free functions implementing the CUDA parts
namespace async_adept_impl {
void CopySurfaceModelToGPU()
{
// Copy surface data to GPU
#ifdef ADEPT_USE_SURF
#ifdef ADEPT_USE_SURF_SINGLE
  using SurfData        = vgbrep::SurfData<float>;
  using BrepCudaManager = vgbrep::BrepCudaManager<float>;
#else
  using SurfData        = vgbrep::SurfData<double>;
  using BrepCudaManager = vgbrep::BrepCudaManager<double>;
#endif
  BrepCudaManager::Instance().TransferSurfData(SurfData::Instance());
  printf("== Surface data transferred to GPU\n");
#endif
}

G4HepEmState *InitG4HepEm(G4HepEmConfig *hepEmConfig)
{
  // here we call everything from InitG4HepEmState, as we need to provide the parameters from the G4HepEmConfig and do
  // not want to initialize to the default values
  auto state = new G4HepEmState;

  // Use the config-provided parameters
  state->fParameters = hepEmConfig->GetG4HepEmParameters();

  // Initialize data and fill each subtable using its initialize function
  state->fData = new G4HepEmData;
  InitG4HepEmData(state->fData);
  InitMaterialAndCoupleData(state->fData, state->fParameters);

  // electrons, positrons, gamma
  InitElectronData(state->fData, state->fParameters, true);
  InitElectronData(state->fData, state->fParameters, false);
  InitGammaData(state->fData, state->fParameters);

  G4HepEmMatCutData *cutData = state->fData->fTheMatCutData;
  G4cout << "fNumG4MatCuts = " << cutData->fNumG4MatCuts << ", fNumMatCutData = " << cutData->fNumMatCutData << G4endl;

  // Copy to GPU.
  CopyG4HepEmDataToGPU(state->fData);
  CopyG4HepEmParametersToGPU(state->fParameters);

  // Create G4HepEmParameters with the device pointer
  G4HepEmParameters parametersOnDevice        = *state->fParameters;
  parametersOnDevice.fParametersPerRegion     = state->fParameters->fParametersPerRegion_gpu;
  parametersOnDevice.fParametersPerRegion_gpu = nullptr;

  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmPars, &parametersOnDevice, sizeof(G4HepEmParameters)));

  // Create G4HepEmData with the device pointers.
  G4HepEmData dataOnDevice;
  dataOnDevice.fTheMatCutData   = state->fData->fTheMatCutData_gpu;
  dataOnDevice.fTheMaterialData = state->fData->fTheMaterialData_gpu;
  dataOnDevice.fTheElementData  = state->fData->fTheElementData_gpu;
  dataOnDevice.fTheElectronData = state->fData->fTheElectronData_gpu;
  dataOnDevice.fThePositronData = state->fData->fThePositronData_gpu;
  dataOnDevice.fTheSBTableData  = state->fData->fTheSBTableData_gpu;
  dataOnDevice.fTheGammaData    = state->fData->fTheGammaData_gpu;
  // The other pointers should never be used.
  dataOnDevice.fTheMatCutData_gpu   = nullptr;
  dataOnDevice.fTheMaterialData_gpu = nullptr;
  dataOnDevice.fTheElementData_gpu  = nullptr;
  dataOnDevice.fTheElectronData_gpu = nullptr;
  dataOnDevice.fThePositronData_gpu = nullptr;
  dataOnDevice.fTheSBTableData_gpu  = nullptr;
  dataOnDevice.fTheGammaData_gpu    = nullptr;

  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmData, &dataOnDevice, sizeof(G4HepEmData)));

  return state;
}

template <typename FieldType>
bool InitializeBField(FieldType &magneticField)
{

  // Allocate and copy the FieldType instance (not the field array itself), and set the global device pointer
  FieldType *dMagneticFieldInstance = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&dMagneticFieldInstance, sizeof(FieldType)));
  COPCORE_CUDA_CHECK(cudaMemcpy(dMagneticFieldInstance, &magneticField, sizeof(FieldType), cudaMemcpyHostToDevice));
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(gMagneticField, &dMagneticFieldInstance, sizeof(FieldType *)));

  return true;
}

template <typename FieldType>
void FreeBField()
{
  FieldType *dMagneticFieldInstance = nullptr;

  // Retrieve the global device pointer from the symbol
  COPCORE_CUDA_CHECK(cudaMemcpyFromSymbol(&dMagneticFieldInstance, gMagneticField, sizeof(FieldType *)));

  if (dMagneticFieldInstance) {
    // Free the device memory and reset global device pointer
    COPCORE_CUDA_CHECK(cudaFree(dMagneticFieldInstance));
    FieldType *nullPtr = nullptr;
    COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(gMagneticField, &nullPtr, sizeof(FieldType *)));
  }
}

void FlushScoring(AdePTScoring &scoring)
{
  scoring.CopyToHost();
  scoring.ClearGPU();
}

/// Allocate memory on device, as well as streams and cuda events to synchronise kernels.
/// If successful, this will initialise the member fGPUState.
/// If memory allocation fails, an exception is thrown. In this case, the caller has to
/// try again after some wait time or with less transport slots.
std::unique_ptr<GPUstate, GPUstateDeleter> InitializeGPU(int trackCapacity, int leakCapacity, int scoringCapacity,
                                                         int numThreads, TrackBuffer &trackBuffer,
                                                         std::vector<AdePTScoring> &scoring, double CPUCapacityFactor,
                                                         double CPUCopyFraction)
{
  auto gpuState_ptr  = std::unique_ptr<GPUstate, GPUstateDeleter>(new GPUstate());
  GPUstate &gpuState = *gpuState_ptr;

  // Allocate structures to manage tracks of an implicit type:
  //  * memory to hold the actual Track elements,
  //  * objects to manage slots inside the memory,
  //  * queues of slots to remember active particle and those needing relocation,
  //  * a stream and an event for synchronization of kernels.

  auto gpuMalloc = [&gpuState](auto &devPtr, std::size_t N, bool emplaceForAutoDelete = true) {
    std::size_t size = N;
    using value_type = std::remove_pointer_t<std::remove_reference_t<decltype(devPtr)>>;
    if constexpr (std::is_object_v<value_type>) {
      size *= sizeof(*devPtr);
    }

    const auto result = cudaMalloc(&devPtr, size);
    if (result != cudaSuccess) {
      std::size_t free, total;
      cudaMemGetInfo(&free, &total);
      std::stringstream msg;
      msg << "Not enough space to allocate " << size / 1024. / 1024 << " MB for " << N << " objects of size "
          << size / N << ". Free memory: " << free / 1024. / 1024 << " MB"
          << " Occupied memory: " << (total - free) / 1024. / 1024 << " MB\n";
      throw std::invalid_argument{msg.str()};
    }
    if (emplaceForAutoDelete) gpuState.allCudaPointers.push_back(devPtr);
  };

  // Create a stream to synchronize kernels of all particle types.
  COPCORE_CUDA_CHECK(cudaStreamCreate(&gpuState.stream));

  // Allocate all slot managers on device
  gpuState.slotManager_dev = nullptr;
  gpuMalloc(gpuState.slotManager_dev, gpuState.nSlotManager_dev);
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    // Number of slots allocated computed based on the proportions set in ParticleType::relativeQueueSize
    const size_t nSlot              = trackCapacity * ParticleType::relativeQueueSize[i];
    const size_t sizeOfQueueStorage = adept::MParray::SizeOfInstance(nSlot);
    const size_t nLeakSlots         = leakCapacity;
    const size_t sizeOfLeakQueue    = adept::MParray::SizeOfInstance(nLeakSlots);

    // Initialize all host slot managers (This call allocates GPU memory)
    gpuState.allmgr_h.slotManagers[i] =
        SlotManager{static_cast<SlotManager::value_type>(nSlot), static_cast<SlotManager::value_type>(nSlot)};
    // Initialize dev slotmanagers by copying the host data
    COPCORE_CUDA_CHECK(cudaMemcpy(&gpuState.slotManager_dev[i], &gpuState.allmgr_h.slotManagers[i], sizeof(SlotManager),
                                  cudaMemcpyDefault));

    // Allocate the queues where the active and leak indices are stored
    // * Current and next active track indices
    // * Current and next leaked track indices
    ParticleType &particleType = gpuState.particles[i];
    particleType.slotManager   = &gpuState.slotManager_dev[i];

    void *gpuPtr = nullptr;
    gpuMalloc(gpuPtr, sizeOfQueueStorage);
    particleType.queues.currentlyActive = static_cast<adept::MParray *>(gpuPtr);
    gpuMalloc(gpuPtr, sizeOfQueueStorage);
    particleType.queues.nextActive = static_cast<adept::MParray *>(gpuPtr);
    gpuMalloc(gpuPtr, sizeOfQueueStorage);
    particleType.queues.reachedInteraction = static_cast<adept::MParray *>(gpuPtr);
    for (int j = 0; j < ParticleQueues::numInteractions; j++) {
      gpuMalloc(gpuPtr, sizeOfQueueStorage);
      particleType.queues.interactionQueues[j] = static_cast<adept::MParray *>(gpuPtr);
    }
    gpuMalloc(gpuPtr, sizeOfLeakQueue);
    particleType.queues.leakedTracksCurrent = static_cast<adept::MParray *>(gpuPtr);
    gpuMalloc(gpuPtr, sizeOfLeakQueue);
    particleType.queues.leakedTracksNext = static_cast<adept::MParray *>(gpuPtr);
    InitParticleQueues<<<1, 1>>>(particleType.queues, nSlot, nLeakSlots);

    COPCORE_CUDA_CHECK(cudaStreamCreate(&particleType.stream));
    COPCORE_CUDA_CHECK(cudaEventCreate(&particleType.event));

    // Allocate the array where the tracks are stored
    // This is the largest allocation. If it does not fit, we need to try again:
    Track *trackStorage_dev = nullptr;
    gpuMalloc(trackStorage_dev, nSlot);

    gpuState.particles[i].tracks = trackStorage_dev;

    printf("%lu track slots allocated for particle type %d on GPU (%.2lf%% of %d total slots allocated)\n", nSlot, i,
           ParticleType::relativeQueueSize[i] * 100, trackCapacity);

#ifdef USE_SPLIT_KERNELS
    // Allocate an array of HepEm tracks per particle type
    switch (i) {
    case 0: // Electrons
      gpuMalloc(gpuState.hepEmBuffers_d.electronsHepEm, nSlot);
      break;
    case 1: // Positrons
      gpuMalloc(gpuState.hepEmBuffers_d.positronsHepEm, nSlot);
      break;
    case 2: // Gammas
      gpuMalloc(gpuState.hepEmBuffers_d.gammasHepEm, nSlot);
      break;
    default:
      printf("Error: Undefined particle type");
      break;
    }
#endif
  }

  // NOTE: deprecated GammaInteractions
  // init gamma interaction queues
  // for (unsigned int i = 0; i < GammaInteractions::NInt; ++i) {
  //   const auto capacity     = trackCapacity / 6;
  //   const auto instanceSize = adept::MParrayT<GammaInteractions::Data>::SizeOfInstance(capacity);
  //   void *gpuPtr            = nullptr;
  //   gpuMalloc(gpuPtr, instanceSize);
  //   gpuState.gammaInteractions.queues[i] = static_cast<adept::MParrayT<GammaInteractions::Data> *>(gpuPtr);
  //   InitQueue<GammaInteractions::Data><<<1, 1>>>(gpuState.gammaInteractions.queues[i], capacity);
  // }

  // initialize statistics
  gpuMalloc(gpuState.stats_dev, 1);
  COPCORE_CUDA_CHECK(cudaMallocHost(&gpuState.stats, sizeof(Stats)));

  // init scoring structures
  gpuMalloc(gpuState.fScoring_dev, numThreads);

  // initialize track debugging
  InitializeTrackDebug();

  scoring.clear();
  scoring.reserve(numThreads);
  for (unsigned int i = 0; i < numThreads; ++i) {
    scoring.emplace_back(gpuState.fScoring_dev + i);
  }
  gpuState.fHitScoring.reset(new HitScoring(scoringCapacity, numThreads, CPUCapacityFactor, CPUCopyFraction));

  const auto injectQueueSize = adept::MParrayT<QueueIndexPair>::SizeOfInstance(trackBuffer.fNumToDevice);
  void *gpuPtr               = nullptr;
  gpuMalloc(gpuPtr, injectQueueSize);
  gpuState.injectionQueue = static_cast<adept::MParrayT<QueueIndexPair> *>(gpuPtr);
  InitQueue<QueueIndexPair><<<1, 1>>>(gpuState.injectionQueue, trackBuffer.fNumToDevice);

  return gpuState_ptr;
}

void AdvanceEventStates(EventState oldState, EventState newState, std::vector<std::atomic<EventState>> &eventStates)
{
  for (auto &eventState : eventStates) {
    EventState expected = oldState;
    eventState.compare_exchange_strong(expected, newState, std::memory_order_release, std::memory_order_relaxed);
  }
}

// Atomically advances the Extract state
void AdvanceExtractState(GPUstate::ExtractState oldState, GPUstate::ExtractState newState,
                         std::atomic<GPUstate::ExtractState> &extractState)
{
  GPUstate::ExtractState expected = oldState;
  auto success =
      extractState.compare_exchange_strong(expected, newState, std::memory_order_release, std::memory_order_relaxed);
#ifndef NDEBUG
  if (!success)
    std::cerr << "Error: Extract state is different than expected. Expected: " << (uint)expected
              << " Found: " << (uint)extractState.load() << std::endl;
  assert(success);
#endif
}

__host__ void ReturnTracksToG4(TrackBuffer &trackBuffer, GPUstate &gpuState,
                               std::vector<std::atomic<EventState>> &eventStates)
{
  std::scoped_lock lock{trackBuffer.fromDeviceMutex};
  const auto &fromDevice                      = trackBuffer.fromDevice_host.get();
  TrackDataWithIDs const *const fromDeviceEnd = fromDevice + *trackBuffer.nFromDevice_host;

  for (TrackDataWithIDs *trackIt = fromDevice; trackIt < fromDeviceEnd; ++trackIt) {
    // TODO: Pass numThreads here, only used in debug mode however
    // assert(0 <= trackIt->threadId && trackIt->threadId <= numThreads);
    trackBuffer.fromDeviceBuffers[trackIt->threadId].push_back(*trackIt);
  }

#ifndef NDEBUG
  for (const auto &buffer : trackBuffer.fromDeviceBuffers) {
    if (buffer.empty()) continue;
    const auto eventId = buffer.front().eventId;
    assert(std::all_of(buffer.begin(), buffer.end(),
                       [eventId](const TrackDataWithIDs &track) { return eventId == track.eventId; }));
  }
#endif
}

void HitProcessingLoop(HitProcessingContext *const context, GPUstate &gpuState,
                       std::vector<std::atomic<EventState>> &eventStates, std::condition_variable &cvG4Workers,
                       int debugLevel)
{
  while (context->keepRunning) {
    std::unique_lock lock(context->mutex);
    context->cv.wait(lock);

    AdvanceEventStates(EventState::RequestHitFlush, EventState::FlushingHits, eventStates);
    gpuState.fHitScoring->TransferHitsToHost(context->hitTransferStream);
    const bool haveNewHits = gpuState.fHitScoring->ProcessHits(cvG4Workers, debugLevel);

    if (haveNewHits) {
      AdvanceEventStates(EventState::FlushingHits, EventState::HitsFlushed, eventStates);
    }

    // Notify all even without newHits because the HitProcessingThread could have been woken up because the Event
    // finished
    cvG4Workers.notify_all();
  }
}

void TransportLoop(int trackCapacity, int leakCapacity, int scoringCapacity, int numThreads, TrackBuffer &trackBuffer,
                   GPUstate &gpuState, std::vector<std::atomic<EventState>> &eventStates,
                   std::condition_variable &cvG4Workers, std::vector<AdePTScoring> &scoring, int adeptSeed,
                   int debugLevel, bool returnAllSteps, bool returnLastStep, unsigned short lastNParticlesOnCPU)
{
  // NVTXTracer tracer{"TransportLoop"};

  using InjectState                             = GPUstate::InjectState;
  using ExtractState                            = GPUstate::ExtractState;
  auto &cudaManager                             = vecgeom::cxx::CudaManager::Instance();
  const vecgeom::cuda::VPlacedVolume *world_dev = cudaManager.world_gpu();

  ParticleType &electrons = gpuState.particles[ParticleType::Electron];
  ParticleType &positrons = gpuState.particles[ParticleType::Positron];
  ParticleType &gammas    = gpuState.particles[ParticleType::Gamma];

  // Auxiliary struct used to keep track of the queues that need flushing
  AllLeaked allLeaked{nullptr, nullptr, nullptr};

  cudaEvent_t cudaEvent, cudaStatsEvent;
  cudaStream_t hitTransferStream, injectStream, extractStream, statsStream, interactionStream;
  COPCORE_CUDA_CHECK(cudaEventCreateWithFlags(&cudaEvent, cudaEventDisableTiming));
  COPCORE_CUDA_CHECK(cudaEventCreateWithFlags(&cudaStatsEvent, cudaEventDisableTiming));
  unique_ptr_cuda<cudaEvent_t> cudaEventCleanup{&cudaEvent};
  unique_ptr_cuda<cudaEvent_t> cudaStatsEventCleanup{&cudaStatsEvent};
  COPCORE_CUDA_CHECK(cudaStreamCreate(&hitTransferStream));
  COPCORE_CUDA_CHECK(cudaStreamCreate(&injectStream));
  COPCORE_CUDA_CHECK(cudaStreamCreate(&extractStream));
  COPCORE_CUDA_CHECK(cudaStreamCreate(&statsStream));
  COPCORE_CUDA_CHECK(cudaStreamCreate(&interactionStream));
  unique_ptr_cuda<cudaStream_t> cudaStreamCleanup{&hitTransferStream};
  unique_ptr_cuda<cudaStream_t> cudaInjectStreamCleanup{&injectStream};
  unique_ptr_cuda<cudaStream_t> cudaExtractStreamCleanup{&extractStream};
  unique_ptr_cuda<cudaStream_t> cudaStatsStreamCleanup{&statsStream};
  unique_ptr_cuda<cudaStream_t> cudaInteractionStreamCleanup{&interactionStream};
  auto waitForOtherStream = [&cudaEvent](cudaStream_t waitingStream, cudaStream_t streamToWaitFor) {
    COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, streamToWaitFor));
    COPCORE_CUDA_CHECK(cudaStreamWaitEvent(waitingStream, cudaEvent));
  };

  // needed for the HOTFIX below
  int injectIteration[numThreads];
  std::fill_n(injectIteration, numThreads, -1);

  std::unique_ptr<HitProcessingContext> hitProcessing{new HitProcessingContext{hitTransferStream}};
  std::thread hitProcessingThread{&HitProcessingLoop,    (HitProcessingContext *)hitProcessing.get(),
                                  std::ref(gpuState),    std::ref(eventStates),
                                  std::ref(cvG4Workers), std::ref(debugLevel)};

  auto computeThreadsAndBlocks = [](unsigned int nParticles) -> std::pair<unsigned int, unsigned int> {
    constexpr int TransportThreads             = 256;
    constexpr int LowOccupancyTransportThreads = 32;

    auto transportBlocks = nParticles / TransportThreads + 1;
    if (transportBlocks < 10) {
      transportBlocks = nParticles / LowOccupancyTransportThreads + 1;
      return {LowOccupancyTransportThreads, transportBlocks};
    }
    return {TransportThreads, transportBlocks};
  };

  std::chrono::steady_clock::time_point startTime;
  if (debugLevel >= 2) {
    static bool isInitialized = false;
    if (!isInitialized) {
      startTime     = std::chrono::steady_clock::now();
      isInitialized = true;
    }
  }

  while (gpuState.runTransport) {
    // NVTXTracer nvtx1{"Setup"}, nvtx2{"Setup2"};
    InitSlotManagers<<<80, 256, 0, gpuState.stream>>>(gpuState.slotManager_dev, gpuState.nSlotManager_dev);
    COPCORE_CUDA_CHECK(cudaMemsetAsync(gpuState.stats_dev, 0, sizeof(Stats), gpuState.stream));

    int inFlight                                                   = 0;
    unsigned int numLeaked                                         = 0;
    unsigned int particlesInFlight[ParticleType::NumParticleTypes] = {1, 1, 1};

    auto needTransport = [](std::atomic<EventState> const &state) {
      return state.load(std::memory_order_acquire) < EventState::LeakedTracksRetrieved;
    };
    // Wait for work from G4 workers:
    while (gpuState.runTransport && std::none_of(eventStates.begin(), eventStates.end(), needTransport)) {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(10ms);
    }

    if (debugLevel > 2) {
      G4cout << "GPU transport starting" << std::endl;
    }

    COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

    // #ifdef USE_NVTX
    //     std::map<AsyncAdePTTransport::EventState, std::string> stateMap{
    //         {EventState::NewTracksFromG4, "NewTracksFromG4"},
    //         {EventState::G4RequestsFlush, "G4RequestsFlush"},
    //         {EventState::Inject, "Inject"},
    //         {EventState::InjectionCompleted, "InjectionCompleted"},
    //         {EventState::Transporting, "Transporting"},
    //         {EventState::WaitingForTransportToFinish, "WaitingForTransportToFinish"},
    //         {EventState::RequestHitFlush, "RequestHitFlush"},
    //         {EventState::HitsFlushed, "HitsFlushed"},
    //         {EventState::FlushingTracks, "FlushingTracks"},
    //         {EventState::DeviceFlushed, "DeviceFlushed"},
    //         {EventState::LeakedTracksRetrieved, "LeakedTracksRetrieved"},
    //         {EventState::ScoringRetrieved, "ScoringRetrieved"}};
    // #endif

    for (unsigned int iteration = 0;
         inFlight > 0 || gpuState.injectState != InjectState::Idle || gpuState.extractState != ExtractState::Idle ||
         std::any_of(eventStates.begin(), eventStates.end(), needTransport);
         ++iteration) {
      // #ifdef USE_NVTX
      //       nvtx1.setTag(stateMap[eventStates[0].load()].data());
      //       nvtx2.setTag(stateMap[eventStates[1].load()].data());
      // #endif

      // Swap the queues for the next iteration.
      electrons.queues.SwapActive();
      positrons.queues.SwapActive();
      gammas.queues.SwapActive();

      const Secondaries secondaries = {
          .electrons = {electrons.tracks, electrons.slotManager, electrons.queues.nextActive},
          .positrons = {positrons.tracks, positrons.slotManager, positrons.queues.nextActive},
          .gammas    = {gammas.tracks, gammas.slotManager, gammas.queues.nextActive},
      };
      const AllParticleQueues allParticleQueues = {{electrons.queues, positrons.queues, gammas.queues}};
#ifdef USE_SPLIT_KERNELS
      const AllInteractionQueues allGammaInteractionQueues = {
          {gammas.queues.interactionQueues[0], gammas.queues.interactionQueues[1], gammas.queues.interactionQueues[2],
           nullptr, gammas.queues.interactionQueues[4]}};
      const AllInteractionQueues allElectronInteractionQueues = {{electrons.queues.interactionQueues[0],
                                                                  electrons.queues.interactionQueues[1], nullptr,
                                                                  nullptr, electrons.queues.interactionQueues[4]}};
      const AllInteractionQueues allPositronInteractionQueues = {
          {positrons.queues.interactionQueues[0], positrons.queues.interactionQueues[1],
           positrons.queues.interactionQueues[2], positrons.queues.interactionQueues[3],
           positrons.queues.interactionQueues[4]}};
#endif
      const TracksAndSlots tracksAndSlots = {{electrons.tracks, positrons.tracks, gammas.tracks},
                                             {electrons.slotManager, positrons.slotManager, gammas.slotManager}};
      // --------------------------
      // *** Particle injection ***
      // --------------------------
      if (gpuState.injectState == InjectState::Idle) {
        for (auto &eventState : eventStates) {
          if (const auto state = eventState.load(std::memory_order_acquire); state == EventState::G4RequestsFlush) {
            eventState = EventState::Inject;
          } else if (state == EventState::Inject) {
            eventState = EventState::InjectionCompleted;
          }
        }

        if (auto &toDevice = trackBuffer.getActiveBuffer(); toDevice.nTrack > 0) {
          gpuState.injectState = InjectState::CreatingSlots;

          trackBuffer.swapToDeviceBuffers();
          std::scoped_lock lock{toDevice.mutex};
          const auto nInject = std::min(toDevice.nTrack.load(), toDevice.maxTracks);
          toDevice.nTrack    = 0;

          if (debugLevel > 3) std::cout << "Injecting " << nInject << " to GPU\n";

          // copy buffer of tracks to device
          COPCORE_CUDA_CHECK(cudaMemcpyAsync(trackBuffer.toDevice_dev.get(), toDevice.tracks,
                                             nInject * sizeof(TrackDataWithIDs), cudaMemcpyHostToDevice, injectStream));
          // Mark end of copy operation:
          COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, injectStream));

          // Init AdePT tracks using the track buffer
          constexpr auto injectThreads = 128u;
          const auto injectBlocks      = (nInject + injectThreads - 1) / injectThreads;
          InitTracks<<<injectBlocks, injectThreads, 0, injectStream>>>(
              trackBuffer.toDevice_dev.get(), nInject, secondaries, world_dev, gpuState.injectionQueue, adeptSeed);
          COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
              injectStream,
              [](void *arg) { (*static_cast<decltype(GPUstate::injectState) *>(arg)) = InjectState::ReadyToEnqueue; },
              &gpuState.injectState));

          // Ensure that copy operation completed before releasing lock on to-device buffer
          COPCORE_CUDA_CHECK(cudaEventSynchronize(cudaEvent));
        } else {
          // No tracks in to-device buffer
          // Move tracks that requested a flush to InjectionCompleted
          gpuState.injectState = InjectState::Idle;
        }
      }

      // *** Enqueue particles that are ready on the device ***
      if (gpuState.injectState == InjectState::ReadyToEnqueue) {
        gpuState.injectState = InjectState::Enqueueing;
        EnqueueTracks<<<1, 256, 0, gpuState.stream>>>(allParticleQueues, gpuState.injectionQueue);
        // New injection has to wait until particles are enqueued:
        waitForOtherStream(injectStream, gpuState.stream);
      } else if (gpuState.injectState == InjectState::Enqueueing) {
        gpuState.injectState = InjectState::Idle;
      }

      // ------------------
      // *** Transport ***
      // ------------------

      AllowFinishOffEventArray allowFinishOffEvent;
      for (int i = 0; i < numThreads; ++i) {
        // if waiting for transport to finish, the last N particles may be finished on CPU
        if (eventStates[i].load(std::memory_order_acquire) == EventState::WaitingForTransportToFinish) {
          allowFinishOffEvent.flags[i] = lastNParticlesOnCPU;
        } else {
          allowFinishOffEvent.flags[i] = 0;
        }
      }

      // *** ELECTRONS ***
      {

        // wait for swapping of hit buffers
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(electrons.stream, gpuState.fHitScoring->getSwapDoneEvent(), 0));

        const auto [threads, blocks] = computeThreadsAndBlocks(particlesInFlight[ParticleType::Electron]);
#ifdef USE_SPLIT_KERNELS
        ElectronHowFar<true><<<blocks, threads, 0, electrons.stream>>>(
            electrons.tracks, gpuState.hepEmBuffers_d.electronsHepEm, electrons.queues.currentlyActive,
            electrons.queues.nextActive, electrons.queues.leakedTracksCurrent, gpuState.stats_dev, allowFinishOffEvent);
        ElectronPropagation<true><<<blocks, threads, 0, electrons.stream>>>(
            electrons.tracks, gpuState.hepEmBuffers_d.electronsHepEm, electrons.queues.currentlyActive,
            electrons.queues.leakedTracksCurrent);
        ElectronMSC<true><<<blocks, threads, 0, electrons.stream>>>(
            electrons.tracks, gpuState.hepEmBuffers_d.electronsHepEm, electrons.queues.currentlyActive);
        ElectronSetupInteractions<true, PerEventScoring><<<blocks, threads, 0, electrons.stream>>>(
            electrons.tracks, gpuState.hepEmBuffers_d.electronsHepEm, electrons.queues.currentlyActive, secondaries,
            electrons.queues.nextActive, electrons.queues.reachedInteraction, allElectronInteractionQueues,
            electrons.queues.leakedTracksCurrent, gpuState.fScoring_dev, returnAllSteps, returnLastStep);
        ElectronRelocation<true, PerEventScoring><<<blocks, threads, 0, electrons.stream>>>(
            electrons.tracks, gpuState.hepEmBuffers_d.electronsHepEm, secondaries, electrons.queues.nextActive,
            electrons.queues.interactionQueues[4], electrons.queues.leakedTracksCurrent, gpuState.fScoring_dev,
            returnAllSteps, returnLastStep);
        // ElectronInteractions<true, PerEventScoring><<<blocks, threads, 0, electrons.stream>>>(
        //     electrons.tracks, gpuState.hepEmBuffers_d.electronsHepEm, secondaries, electrons.queues.nextActive,
        //     electrons.queues.reachedInteraction, electrons.queues.leakedTracksCurrent, gpuState.fScoring_dev,
        //     returnAllSteps, returnLastStep);
        ElectronIonization<true, PerEventScoring><<<blocks, threads, 0, electrons.stream>>>(
            electrons.tracks, gpuState.hepEmBuffers_d.electronsHepEm, secondaries, electrons.queues.nextActive,
            electrons.queues.interactionQueues[0], electrons.queues.leakedTracksCurrent, gpuState.fScoring_dev,
            returnAllSteps, returnLastStep);
        ElectronBremsstrahlung<true, PerEventScoring><<<blocks, threads, 0, electrons.stream>>>(
            electrons.tracks, gpuState.hepEmBuffers_d.electronsHepEm, secondaries, electrons.queues.nextActive,
            electrons.queues.interactionQueues[1], electrons.queues.leakedTracksCurrent, gpuState.fScoring_dev,
            returnAllSteps, returnLastStep);
#else
        TransportElectrons<PerEventScoring><<<blocks, threads, 0, electrons.stream>>>(
            electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive,
            electrons.queues.leakedTracksCurrent, gpuState.fScoring_dev, gpuState.stats_dev, allowFinishOffEvent,
            returnAllSteps, returnLastStep);
#endif
        COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, electrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, electrons.event, 0));
      }

      // *** POSITRONS ***
      {

        // wait for swapping of hit buffers
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(positrons.stream, gpuState.fHitScoring->getSwapDoneEvent(), 0));

        const auto [threads, blocks] = computeThreadsAndBlocks(particlesInFlight[ParticleType::Positron]);
#ifdef USE_SPLIT_KERNELS
        ElectronHowFar<false><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, gpuState.hepEmBuffers_d.positronsHepEm, positrons.queues.currentlyActive,
            positrons.queues.nextActive, positrons.queues.leakedTracksCurrent, gpuState.stats_dev, allowFinishOffEvent);
        ElectronPropagation<false><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, gpuState.hepEmBuffers_d.positronsHepEm, positrons.queues.currentlyActive,
            positrons.queues.leakedTracksCurrent);
        ElectronMSC<false><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, gpuState.hepEmBuffers_d.positronsHepEm, positrons.queues.currentlyActive);
        ElectronSetupInteractions<false, PerEventScoring><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, gpuState.hepEmBuffers_d.positronsHepEm, positrons.queues.currentlyActive, secondaries,
            positrons.queues.nextActive, positrons.queues.reachedInteraction, allPositronInteractionQueues,
            positrons.queues.leakedTracksCurrent, gpuState.fScoring_dev, returnAllSteps, returnLastStep);
        ElectronRelocation<false, PerEventScoring><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, gpuState.hepEmBuffers_d.positronsHepEm, secondaries, positrons.queues.nextActive,
            positrons.queues.interactionQueues[4], positrons.queues.leakedTracksCurrent, gpuState.fScoring_dev,
            returnAllSteps, returnLastStep);
        // ElectronInteractions<false, PerEventScoring><<<blocks, threads, 0, positrons.stream>>>(
        //     positrons.tracks, gpuState.hepEmBuffers_d.positronsHepEm, secondaries, positrons.queues.nextActive,
        //     positrons.queues.reachedInteraction, positrons.queues.leakedTracksCurrent, gpuState.fScoring_dev,
        //     returnAllSteps, returnLastStep);
        ElectronIonization<false, PerEventScoring><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, gpuState.hepEmBuffers_d.positronsHepEm, secondaries, positrons.queues.nextActive,
            positrons.queues.interactionQueues[0], positrons.queues.leakedTracksCurrent, gpuState.fScoring_dev,
            returnAllSteps, returnLastStep);
        ElectronBremsstrahlung<false, PerEventScoring><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, gpuState.hepEmBuffers_d.positronsHepEm, secondaries, positrons.queues.nextActive,
            positrons.queues.interactionQueues[1], positrons.queues.leakedTracksCurrent, gpuState.fScoring_dev,
            returnAllSteps, returnLastStep);
        PositronAnnihilation<PerEventScoring><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, gpuState.hepEmBuffers_d.positronsHepEm, secondaries, positrons.queues.nextActive,
            positrons.queues.interactionQueues[2], positrons.queues.leakedTracksCurrent, gpuState.fScoring_dev,
            returnAllSteps, returnLastStep);
        PositronStoppedAnnihilation<PerEventScoring><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, gpuState.hepEmBuffers_d.positronsHepEm, secondaries, positrons.queues.nextActive,
            positrons.queues.interactionQueues[3], positrons.queues.leakedTracksCurrent, gpuState.fScoring_dev,
            returnAllSteps, returnLastStep);
#else

        TransportPositrons<PerEventScoring><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive,
            positrons.queues.leakedTracksCurrent, gpuState.fScoring_dev, gpuState.stats_dev, allowFinishOffEvent,
            returnAllSteps, returnLastStep);
#endif

        COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, positrons.event, 0));
      }

      // *** GAMMAS ***
      {

        // wait for swapping of hit buffers
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gammas.stream, gpuState.fHitScoring->getSwapDoneEvent(), 0));

        const auto [threads, blocks] = computeThreadsAndBlocks(particlesInFlight[ParticleType::Gamma]);
#ifdef USE_SPLIT_KERNELS
        GammaHowFar<<<blocks, threads, 0, gammas.stream>>>(
            gammas.tracks, gpuState.hepEmBuffers_d.gammasHepEm, gammas.queues.currentlyActive,
            gammas.queues.leakedTracksCurrent, gpuState.stats_dev, allowFinishOffEvent);
        GammaPropagation<<<blocks, threads, 0, gammas.stream>>>(gammas.tracks, gpuState.hepEmBuffers_d.gammasHepEm,
                                                                gammas.queues.currentlyActive);
        GammaSetupInteractions<PerEventScoring><<<blocks, threads, 0, gammas.stream>>>(
            gammas.tracks, gpuState.hepEmBuffers_d.gammasHepEm, gammas.queues.currentlyActive, secondaries,
            gammas.queues.nextActive, gammas.queues.reachedInteraction, allGammaInteractionQueues,
            gammas.queues.leakedTracksCurrent, gpuState.fScoring_dev, returnAllSteps, returnLastStep);
        GammaRelocation<PerEventScoring><<<blocks, threads, 0, gammas.stream>>>(
            gammas.tracks, gpuState.hepEmBuffers_d.gammasHepEm, secondaries, gammas.queues.nextActive,
            gammas.queues.interactionQueues[4], gammas.queues.leakedTracksCurrent, gpuState.fScoring_dev,
            returnAllSteps, returnLastStep);
        // Copying the number of interacting tracks back to host and using this information to adjust
        // the launch parameters of the interactions kernel is complicated and expensive due to a
        // required additional kernel launch and copy. Instead, launch the kernel with the same
        // parameters, the unneeded threads inmediately return and become free again.
        // GammaInteractions<PerEventScoring><<<blocks, threads, 0, gammas.stream>>>(
        //     gammas.tracks, gpuState.hepEmBuffers_d.gammasHepEm, secondaries, gammas.queues.nextActive,
        //     gammas.queues.reachedInteraction, gammas.queues.leakedTracksCurrent, gpuState.fScoring_dev,
        //     returnAllSteps, returnLastStep);

        GammaConversion<PerEventScoring><<<blocks, threads, 0, gammas.stream>>>(
            gammas.tracks, gpuState.hepEmBuffers_d.gammasHepEm, secondaries, gammas.queues.nextActive,
            gammas.queues.interactionQueues[0], gammas.queues.leakedTracksCurrent, gpuState.fScoring_dev,
            returnAllSteps, returnLastStep);
        GammaCompton<PerEventScoring><<<blocks, threads, 0, gammas.stream>>>(
            gammas.tracks, gpuState.hepEmBuffers_d.gammasHepEm, secondaries, gammas.queues.nextActive,
            gammas.queues.interactionQueues[1], gammas.queues.leakedTracksCurrent, gpuState.fScoring_dev,
            returnAllSteps, returnLastStep);
        GammaPhotoelectric<PerEventScoring><<<blocks, threads, 0, gammas.stream>>>(
            gammas.tracks, gpuState.hepEmBuffers_d.gammasHepEm, secondaries, gammas.queues.nextActive,
            gammas.queues.interactionQueues[2], gammas.queues.leakedTracksCurrent, gpuState.fScoring_dev,
            returnAllSteps, returnLastStep);
#else
        TransportGammas<PerEventScoring><<<blocks, threads, 0, gammas.stream>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive,
            gammas.queues.leakedTracksCurrent, gpuState.fScoring_dev, gpuState.stats_dev, allowFinishOffEvent,
            returnAllSteps, returnLastStep); //, gpuState.gammaInteractions);
#endif

        // constexpr unsigned int intThreads = 128;
        // ApplyGammaInteractions<PerEventScoring><<<dim3(20, 3, 1), intThreads, 0, gammas.stream>>>(
        //     gammas.tracks, secondaries, gammas.queues.nextActive, gpuState.fScoring_dev,
        //     gpuState.gammaInteractions);

        COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, gammas.event, 0));
      }

      // ---------------------------------------
      // *** Count detailed event statistics ***
      // ---------------------------------------
      {
        AdvanceEventStates(EventState::Transporting, EventState::WaitingForTransportToFinish, eventStates);
        AdvanceEventStates(EventState::InjectionCompleted, EventState::Transporting, eventStates);

        // Reset all counters count the currently flying population
        ZeroEventCounters<<<1, 256, 0, statsStream>>>(gpuState.stats_dev);
        CountCurrentPopulation<<<ParticleType::NumParticleTypes, 128, 0, statsStream>>>(
            allParticleQueues, gpuState.stats_dev, tracksAndSlots);
        // Count leaked tracks. Note that new tracks might be added while/after we count:
        CountLeakedTracks<<<2 * ParticleType::NumParticleTypes, 128, 0, statsStream>>>(
            allParticleQueues, gpuState.stats_dev, tracksAndSlots);

        waitForOtherStream(gpuState.stream, statsStream);

        // Copy the number of particles in flight to the previous one, which is used within the kernel
        COPCORE_CUDA_CHECK(cudaMemcpyAsync(gpuState.stats_dev->perEventInFlightPrevious,
                                           gpuState.stats_dev->perEventInFlight, kMaxThreads * sizeof(unsigned int),
                                           cudaMemcpyDeviceToDevice, statsStream));

        // Get results to host:
        COPCORE_CUDA_CHECK(
            cudaMemcpyAsync(gpuState.stats, gpuState.stats_dev, sizeof(Stats), cudaMemcpyDeviceToHost, statsStream));
        COPCORE_CUDA_CHECK(cudaEventRecord(cudaStatsEvent, statsStream));
      }

      // -------------------------
      // *** Collect particles ***
      // -------------------------

      // There are two reasons to transfers leaks back to the host:
      // - An event requested a flush
      // - The leak queue usage is too high
      //
      // If the queue usage is too high, but we are already extracting leaks, the GPU transport
      // thread needs to wait. It stops launching kernels and instead steers the leak extraction
      // loop until all have been transferred
      // This issue is very similar to the bottleneck encountered when extracting all steps
      // done in simple geometries. It doesn't appear to be a problem in complex geometries
      //
      // If an event has requested a flush, but we are already extracting leaks, we don't necessarily
      // need to wait. The GPU transport can continue as long as the current leak queue remains below
      // the usage threshold

      // Is any of the current leak queues over the usage threshold?
      bool leakQueueNeedsTransfer = false;
      for (int particleType = 0; particleType < ParticleType::NumParticleTypes; ++particleType) {
        // NOTE: This chek is done without synchronization with the stats counting and transfer, which
        // means that we might be seeing the usage during the previous iteration. We expect that this
        // will not be an issue in most situations, while allowing us to parallelize this work with
        // the stats counting
        if (gpuState.stats->nLeakedCurrent[particleType] > 0.5 * leakCapacity) {
          leakQueueNeedsTransfer = true;
          break;
        }
      }

      // Did an event request a flush?
      bool leakExtractionRequested = std::any_of(eventStates.begin(), eventStates.end(), [](const auto &eventState) {
        return eventState.load(std::memory_order_acquire) == EventState::HitsFlushed;
      });

      bool leakExtractionNeeded = leakQueueNeedsTransfer || leakExtractionRequested;

      // Leak Extraction
      // We always do one pass of this loop. If the leak queues are over the usage threshold but
      // an extraction is already in progress, the transport thread will stay in this loop until
      // it finishes and the queues can be swapped
      do {
        if (gpuState.extractState.load(std::memory_order_acquire) != ExtractState::Idle) {
          // If:
          // - A previous extraction is in progress
          // - The current leak queue usage is under the threshold
          if (!leakQueueNeedsTransfer) {
            // An event requested a flush, but the current leak queue usage is under the threshold,
            // transport can continue
            leakExtractionNeeded = false;
          } else {
            // Otherwise, the current leak queue usage is above the threshold. Transport needs to stop until the
            // transfer of these leaks can start
            if (debugLevel > 5) {
              printf("Leak extraction blocked. Transport will stop until current extraction ends\n");
            }
          }
        }

        // If not extracting tracks from a previous event, freeze the current leak queues and swap the next ones in
        if (gpuState.extractState.load(std::memory_order_acquire) == ExtractState::Idle && leakExtractionNeeded) {

          // if (gpuState.extractState.load(std::memory_order_acquire) == ExtractState::Idle &&
          //   std::any_of(eventStates.begin(), eventStates.end(), [](const auto &eventState) {
          //     return eventState.load(std::memory_order_acquire) == EventState::HitsFlushed;
          //   })) {

          // Transport can continue
          leakExtractionNeeded = false;

          AdvanceEventStates(EventState::HitsFlushed, EventState::FlushingTracks, eventStates);
          // Advance the extractState. This ensures that this code will not run again until this flush has been
          // completed, this is important to ensure that no events can enter the `FlushingTracks` state while an
          // extraction is already in progress
          AdvanceExtractState(ExtractState::Idle, ExtractState::ExtractionRequested, gpuState.extractState);

          if (debugLevel > 5) {
            for (unsigned short threadId = 0; threadId < eventStates.size(); ++threadId) {
              if (eventStates[threadId].load(std::memory_order_acquire) == EventState::FlushingTracks) {
                printf("\033[48;5;25mFlushing leaks for event %d\033[0m\n", threadId);
              }
            }
          }

          // We need to keep track of how many tracks have already been transferred to the host
          trackBuffer.fNumLeaksTransferred = 0;

          // This struct will hold the queues that need to be flushed
          allLeaked = {
              .leakedElectrons = {electrons.tracks, electrons.queues.leakedTracksCurrent, electrons.slotManager},
              .leakedPositrons = {positrons.tracks, positrons.queues.leakedTracksCurrent, positrons.slotManager},
              .leakedGammas    = {gammas.tracks, gammas.queues.leakedTracksCurrent, gammas.slotManager}};

          // Ensure that transport that's writing to the old queues finishes before collecting leaked tracks
          for (auto const &event : {electrons.event, positrons.event, gammas.event}) {
            COPCORE_CUDA_CHECK(cudaStreamWaitEvent(extractStream, event));
          }

          // Once transport has finished, we can start extracting the leaks
          COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
              extractStream,
              [](void *arg) {
                AdvanceExtractState(ExtractState::ExtractionRequested, ExtractState::TracksNeedTransfer,
                                    *static_cast<decltype(GPUstate::extractState) *>(arg));
              },
              &gpuState.extractState));

          // Swap host pointer to the leak queues. This freezes the current queues and starts filling the next
          electrons.queues.SwapLeakedQueue();
          positrons.queues.SwapLeakedQueue();
          gammas.queues.SwapLeakedQueue();
        }

        // When the leak queues are frozen, we can start copying the leaks to the host
        if (gpuState.extractState.load(std::memory_order_acquire) == ExtractState::TracksNeedTransfer) {

          // Update the state so that the staging buffer will not be modified again until tracks have been copied
          // gpuState.extractState = ExtractState::PreparingTracks;
          AdvanceExtractState(ExtractState::TracksNeedTransfer, ExtractState::PreparingTracks, gpuState.extractState);

          // // Populate the staging buffer and copy to host
          constexpr unsigned int block_size = 128;
          const unsigned int grid_size      = (trackBuffer.fNumFromDevice + block_size - 1) / block_size;
          FillFromDeviceBuffer<<<grid_size, block_size, 0, extractStream>>>(
              allLeaked, trackBuffer.fromDevice_dev.get(), trackBuffer.fNumFromDevice,
              // printtotal, allLeaked, trackBuffer.fromDevice_dev.get(), trackBuffer.fNumFromDevice,
              trackBuffer.fNumLeaksTransferred);

          // Copy the number of leaked tracks to host
          COPCORE_CUDA_CHECK(cudaMemcpyFromSymbolAsync(trackBuffer.nFromDevice_host.get(), nFromDevice_dev,
                                                       sizeof(unsigned int), 0, cudaMemcpyDeviceToHost, extractStream));
          // Copy the number of tracks remaining on GPU to host
          COPCORE_CUDA_CHECK(cudaMemcpyFromSymbolAsync(trackBuffer.nRemainingLeaks_host.get(), nRemainingLeaks_dev,
                                                       sizeof(unsigned int), 0, cudaMemcpyDeviceToHost, extractStream));

          // Update the state after the copy
          COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
              extractStream,
              [](void *arg) {
                AdvanceExtractState(ExtractState::PreparingTracks, ExtractState::TracksReadyToCopy,
                                    *static_cast<decltype(GPUstate::extractState) *>(arg));
              },
              &gpuState.extractState));
        }

        if (gpuState.extractState.load(std::memory_order_acquire) == ExtractState::TracksReadyToCopy) {
          AdvanceExtractState(ExtractState::TracksReadyToCopy, ExtractState::CopyingTracks, gpuState.extractState);

          // printf("COPYING: %d\n", *trackBuffer.nFromDevice_host);

          // Copy leaked tracks to host
          COPCORE_CUDA_CHECK(cudaMemcpyAsync(trackBuffer.fromDevice_host.get(), trackBuffer.fromDevice_dev.get(),
                                             (*trackBuffer.nFromDevice_host) * sizeof(TrackDataWithIDs),
                                             cudaMemcpyDeviceToHost, extractStream));
          // Update the state after the copy
          COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
              extractStream,
              [](void *arg) {
                AdvanceExtractState(ExtractState::CopyingTracks, ExtractState::TracksOnHost,
                                    *static_cast<decltype(GPUstate::extractState) *>(arg));
              },
              &gpuState.extractState));
        }

        if (gpuState.extractState.load(std::memory_order_acquire) == ExtractState::TracksOnHost) {
          AdvanceExtractState(ExtractState::TracksOnHost, ExtractState::SavingTracks, gpuState.extractState);
          // Update the number of tracks already transferred
          trackBuffer.fNumLeaksTransferred += *trackBuffer.nFromDevice_host;

          // Auxiliary struct to pass the necessary data to the callback
          struct CallbackData {
            TrackBuffer *trackBuffer;
            GPUstate *gpuState;
            std::vector<std::atomic<EventState>> *eventStates;
          };
          // Needs to be dynamically allocated, since the callback may execute after
          // the current scope has ended.
          CallbackData *data = new CallbackData{&trackBuffer, &gpuState, &eventStates};

          // Distribute the leaked tracks on host to the appropriate G4 workers
          COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
              extractStream,
              [](void *userData) {
                CallbackData *data = static_cast<CallbackData *>(userData);
                ReturnTracksToG4(*data->trackBuffer, *data->gpuState, *data->eventStates);
                AdvanceExtractState(ExtractState::SavingTracks, ExtractState::TracksSaved,
                                    data->gpuState->extractState);
                delete data;
              },
              data));
        }

        // Now we can re-use the host buffer for the next copy, if there are any remaining leaks on GPU
        if (gpuState.extractState.load(std::memory_order_acquire) == ExtractState::TracksSaved) {
          if (*trackBuffer.nRemainingLeaks_host == 0) {
            // Extraction finished, clear the queues and set to idle
            ClearQueues<<<1, 1, 0, extractStream>>>(allLeaked.leakedElectrons.fLeakedQueue,
                                                    allLeaked.leakedPositrons.fLeakedQueue,
                                                    allLeaked.leakedGammas.fLeakedQueue);
            // ExtraxtState is set to Idle, a new extraction can be started
            // The events that had requested a flush are guaranteed to have all their tracks available on host
            AdvanceExtractState(ExtractState::TracksSaved, ExtractState::Idle, gpuState.extractState);

            if (debugLevel > 5) {
              for (unsigned short threadId = 0; threadId < eventStates.size(); ++threadId) {
                if (eventStates[threadId].load(std::memory_order_acquire) == EventState::FlushingTracks) {
                  printf("\033[48;5;208mEvent %d flushed\033[0m\n", threadId);
                }
              }
            }

            AdvanceEventStates(EventState::FlushingTracks, EventState::DeviceFlushed, eventStates);

          } else {
            // There are still tracks left on device
            AdvanceExtractState(ExtractState::TracksSaved, ExtractState::TracksNeedTransfer, gpuState.extractState);
          }
        }
      } while (leakExtractionNeeded);

      // -------------------------
      // *** Finish iteration ***
      // -------------------------

      // This kernel needs to wait that all of the above work (except for asynchronous particle transfer) is done.
      // Don't forget to synchronise any of the transport or event counting with it.
      FinishIteration<<<4, 32, 0, gpuState.stream>>>(allParticleQueues, gpuState.stats_dev, tracksAndSlots);
      //, gpuState.gammaInteractions); // Note: deprecated gammainteractions

      // Try to free slots if one of the queues is half full
      if (gpuState.injectState != InjectState::CreatingSlots) {
        // NOTE: This is done before synchronizing with the stats copy. This means that the value we
        // see may not be up to date. This is acceptable in most situations
        for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
          if (gpuState.stats->slotFillLevel[i] > 0.5) {
            // Freeing of slots has to run exclusively
            // FIXME: Revise this code and make sure all three streams actually need to be synchronized
            // with gpuState.stream
            waitForOtherStream(gpuState.stream, hitTransferStream);
            waitForOtherStream(gpuState.stream, injectStream);
            waitForOtherStream(gpuState.stream, extractStream);
            static_assert(gpuState.nSlotManager_dev == ParticleType::NumParticleTypes,
                          "The below launches assume there is a slot manager per particle type.");
            FreeSlots1<<<10, 256, 0, gpuState.stream>>>(gpuState.slotManager_dev + i);
            FreeSlots2<<<1, 1, 0, gpuState.stream>>>(gpuState.slotManager_dev + i);
            waitForOtherStream(hitTransferStream, gpuState.stream);
            waitForOtherStream(injectStream, gpuState.stream);
            waitForOtherStream(extractStream, gpuState.stream);
          }
        }
      }

      // *** Synchronise all but transfer stream with the end of this iteration ***
      COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, gpuState.stream));
      for (auto stream : {electrons.stream, positrons.stream, gammas.stream, statsStream}) {
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, cudaEvent));
      }

      // ------------------------------------------
      // *** Take decisions for next iterations ***
      // ------------------------------------------

      // *** Count particles in flight ***
      {
        inFlight  = 0;
        numLeaked = 0;
        // FIXME: Synchronizing with the injectionStream here, along with the stats stream is not
        // needed by the logic. It is a temporary fix to a long-standing bug where some particles may
        // be injected after their events have already finished. This synchronization doesn't cause
        // any measurable slowdown, but it should be removed once the underlying cause of the bug is
        // understood and fixed
        COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, injectStream));
        // Synchronize with stats count before taking decisions
        cudaError_t result, injectResult;
        while ((result = cudaEventQuery(cudaStatsEvent)) == cudaErrorNotReady ||
               (injectResult = cudaEventQuery(cudaEvent)) == cudaErrorNotReady) {
          // Cuda uses a busy wait. This reduces CPU consumption by 50%:
          using namespace std::chrono_literals;
          std::this_thread::sleep_for(50us);
        }
        COPCORE_CUDA_CHECK(result);
        COPCORE_CUDA_CHECK(injectResult);

        for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
          inFlight += gpuState.stats->inFlight[i];
          numLeaked += gpuState.stats->leakedTracks[i];
          particlesInFlight[i] = gpuState.stats->inFlight[i];
        }

        for (unsigned short threadId = 0; threadId < numThreads; ++threadId) {
          const auto state = eventStates[threadId].load(std::memory_order_acquire);
          if (state == EventState::WaitingForTransportToFinish && gpuState.stats->perEventInFlight[threadId] == 0) {
            eventStates[threadId] = EventState::RequestHitFlush;
          }
          if (state >= EventState::RequestHitFlush && gpuState.stats->perEventInFlight[threadId] != 0) {
            std::cerr << "ERROR thread " << threadId << " is in state " << static_cast<unsigned int>(state)
                      << " and occupancy is " << gpuState.stats->perEventInFlight[threadId] << "\n";
          }
        }

        // *** Hit management ***
        if (!gpuState.fHitScoring->ReadyToSwapBuffers()) {
          hitProcessing->cv.notify_one();
        } else {
          if (gpuState.stats->hitBufferOccupancy >= gpuState.fHitScoring->HitCapacity() / numThreads / 2 ||
              gpuState.stats->hitBufferOccupancy >= 30000 ||
              std::any_of(eventStates.begin(), eventStates.end(), [](const auto &state) {
                return state.load(std::memory_order_acquire) == EventState::RequestHitFlush;
              })) {
            // Reset hitBufferOccupancy to 0 when we swap, as the delay of updating it could cause another unwanted swap
            COPCORE_CUDA_CHECK(
                cudaMemsetAsync(&(gpuState.stats_dev->hitBufferOccupancy), 0, sizeof(unsigned int), gpuState.stream));
            gpuState.fHitScoring->SwapDeviceBuffers(gpuState.stream);
            hitProcessing->cv.notify_one();
          }
        }
      }

      // *** Notify G4 workers if their events completed ***
      if (std::any_of(eventStates.begin(), eventStates.end(), [](const std::atomic<EventState> &state) {
            return state.load(std::memory_order_acquire) == EventState::DeviceFlushed;
          })) {
        // Notify HitProcessingThread to notify the workers. Do not notify workers directly, as this could bypass the
        // processing of hits
        hitProcessing->cv.notify_one();
      }

      if (debugLevel >= 3 && inFlight > 0 || (debugLevel >= 2 && iteration % 500 == 0)) {
        auto elapsedTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
        std::cerr << "Time elapsed: " << std::fixed << std::setprecision(6) << elapsedTime << "s ";
        std::cerr << inFlight << " in flight ";
        std::cerr << "(" << gpuState.stats->inFlight[ParticleType::Electron] << " "
                  << gpuState.stats->inFlight[ParticleType::Positron] << " "
                  << gpuState.stats->inFlight[ParticleType::Gamma] << "),\tqueues:(" << std::setprecision(3)
                  << gpuState.stats->queueFillLevel[ParticleType::Electron] << " "
                  << gpuState.stats->queueFillLevel[ParticleType::Positron] << " "
                  << gpuState.stats->queueFillLevel[ParticleType::Gamma] << ")";
        std::cerr << "\t slots [e-, e+, gamma]: [" << gpuState.stats->slotFillLevel[0] << ", "
                  << gpuState.stats->slotFillLevel[1] << ", " << gpuState.stats->slotFillLevel[2] << "], " << numLeaked
                  << " leaked."
                  << "\tInjectState: " << static_cast<unsigned int>(gpuState.injectState.load())
                  << "\tExtractState: " << static_cast<unsigned int>(gpuState.extractState.load())
                  << "\tHitBuffer: " << gpuState.stats->hitBufferOccupancy
                  << "\tHitBufferReadyToSwap: " << gpuState.fHitScoring->ReadyToSwapBuffers();
        gpuState.fHitScoring->PrintHostBufferState();
        gpuState.fHitScoring->PrintDeviceBufferStates();
        if (debugLevel >= 4) {
          std::cerr << "\n\tper event: ";
          for (unsigned int i = 0; i < numThreads; ++i) {
            std::cerr << i << ": " << gpuState.stats->perEventInFlight[i]
                      << " (s=" << static_cast<unsigned short>(eventStates[i].load(std::memory_order_acquire)) << ")\t";
          }
        }
        std::cerr << std::endl;
      }

#ifndef NDEBUG
      // *** Check slots ***
      if (false && iteration % 100 == 0 && gpuState.injectState != InjectState::CreatingSlots &&
          gpuState.extractState != ExtractState::PreparingTracks) {
        AssertConsistencyOfSlotManagers<<<120, 256, 0, gpuState.stream>>>(gpuState.slotManager_dev,
                                                                          gpuState.nSlotManager_dev);
        COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));
      }

#if false
      for (int i = 0; i < ParticleType::NumParticleTypes; ++i) {
        ParticleType &part = gpuState.particles[i];
        COPCORE_CUDA_CHECK(cudaMemcpyAsync(&part.slotManager_host, part.slotManager, sizeof(SlotManager),
                                           cudaMemcpyDefault, gpuState.stream));
      }
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));
      {
        unsigned int slotsUsed[3];
        unsigned int slotsMax[3];
        unsigned int slotsToFree[3];
        for (int i = 0; i < ParticleType::NumParticleTypes; ++i) {
          ParticleType &part = gpuState.particles[i];
          slotsUsed[i]       = part.slotManager_host.fSlotCounter - part.slotManager_host.fFreeCounter;
          slotsMax[i]        = part.slotManager_host.fSlotCounterMax;
          slotsToFree[i]     = part.slotManager_host.fFreeCounter;
        }
        std::cout << "SlotManager: (" << slotsUsed[0] << ", " << slotsUsed[1] << ", " << slotsUsed[2]
                  << ") slots used.\ttoFree: (" << slotsToFree[0] << ", " << slotsToFree[1] << ", " << slotsToFree[2]
                  << ")\tmax: (" << slotsMax[0] << ", " << slotsMax[1] << ", " << slotsMax[2] << ")\n ";
      }
#endif
#endif
    }

    AllParticleQueues queues = {{electrons.queues, positrons.queues, gammas.queues}};
    ClearAllQueues<<<1, 1, 0, gpuState.stream>>>(queues);
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

    if (debugLevel > 2) std::cout << "End transport loop.\n";
  }

  hitProcessing->keepRunning = false;
  hitProcessing->cv.notify_one();
  hitProcessingThread.join();
}

std::pair<GPUHit *, GPUHit *> GetGPUHitsFromBuffer(unsigned int threadId, unsigned int eventId, GPUstate &gpuState,
                                                   bool &dataOnBuffer)
{
  HitQueueItem *hitItem = gpuState.fHitScoring->GetNextHitsHandle(threadId, dataOnBuffer);
  if (hitItem) {
    return {hitItem->begin, hitItem->end};
  } else {
    return {nullptr, nullptr};
  }
}

void CloseGPUBuffer(unsigned int threadId, GPUstate &gpuState, GPUHit *begin, const bool dataOnBuffer)
{
  gpuState.fHitScoring->CloseHitsHandle(threadId, begin, dataOnBuffer);
}

// TODO: Make it clear that this will initialize and return the GPUState or make a
// separate init function that will compile here and be called from the .icc
std::thread LaunchGPUWorker(int trackCapacity, int leakCapacity, int scoringCapacity, int numThreads,
                            TrackBuffer &trackBuffer, GPUstate &gpuState,
                            std::vector<std::atomic<EventState>> &eventStates, std::condition_variable &cvG4Workers,
                            std::vector<AdePTScoring> &scoring, int adeptSeed, int debugLevel, bool returnAllSteps,
                            bool returnLastStep, unsigned short lastNParticlesOnCPU)
{
  return std::thread{&TransportLoop,
                     trackCapacity,
                     leakCapacity,
                     scoringCapacity,
                     numThreads,
                     std::ref(trackBuffer),
                     std::ref(gpuState),
                     std::ref(eventStates),
                     std::ref(cvG4Workers),
                     std::ref(scoring),
                     adeptSeed,
                     debugLevel,
                     returnAllSteps,
                     returnLastStep,
                     lastNParticlesOnCPU};
}

void FreeGPU(std::unique_ptr<AsyncAdePT::GPUstate, AsyncAdePT::GPUstateDeleter> &gpuState, G4HepEmState &g4hepem_state,
             std::thread &gpuWorker)
{
  gpuState->runTransport = false;
  gpuWorker.join();

  adeptint::VolAuxData *volAux = nullptr;
  COPCORE_CUDA_CHECK(cudaMemcpyFromSymbol(&volAux, AsyncAdePT::gVolAuxData, sizeof(adeptint::VolAuxData *)));
  COPCORE_CUDA_CHECK(cudaFree(volAux));

  // Free resources.
  gpuState.reset();

  // Free G4HepEm data
  FreeG4HepEmData(g4hepem_state.fData);
  FreeG4HepEmParametersOnGPU(g4hepem_state.fParameters);

  // Free magnetic field
#ifdef ADEPT_USE_EXT_BFIELD
  FreeBField<GeneralMagneticField>();
#else
  FreeBField<UniformMagneticField>();
#endif
}

// explicit instantiation
template bool InitializeBField<GeneralMagneticField>(GeneralMagneticField &);
template bool InitializeBField<UniformMagneticField>(UniformMagneticField &);

} // namespace async_adept_impl

namespace AsyncAdePT {
__constant__ __device__ struct G4HepEmParameters g4HepEmPars;
__constant__ __device__ struct G4HepEmData g4HepEmData;

__constant__ __device__ adeptint::VolAuxData *gVolAuxData = nullptr;

/// Transfer volume auxiliary data to GPU
void InitVolAuxArray(adeptint::VolAuxArray &array)
{
  using adeptint::VolAuxData;
  COPCORE_CUDA_CHECK(cudaMalloc(&array.fAuxData_dev, sizeof(VolAuxData) * array.fNumVolumes));
  COPCORE_CUDA_CHECK(
      cudaMemcpy(array.fAuxData_dev, array.fAuxData, sizeof(VolAuxData) * array.fNumVolumes, cudaMemcpyHostToDevice));
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(gVolAuxData, &array.fAuxData_dev, sizeof(VolAuxData *)));
}

/// Initialise the track buffers used to communicate between host and device.
TrackBuffer::TrackBuffer(unsigned int numToDevice, unsigned int numFromDevice, unsigned short nThread)
    : fNumToDevice{numToDevice}, fNumFromDevice{numFromDevice}, fromDeviceBuffers(nThread)
{
  TrackDataWithIDs *devPtr, *hostPtr;
  // Double buffer for lock-free host runs:
  COPCORE_CUDA_CHECK(cudaMallocHost(&hostPtr, 2 * numToDevice * sizeof(TrackDataWithIDs)));
  COPCORE_CUDA_CHECK(cudaMalloc(&devPtr, numToDevice * sizeof(TrackDataWithIDs)));

  toDevice_host.reset(hostPtr);
  toDevice_dev.reset(devPtr);

  COPCORE_CUDA_CHECK(cudaMallocHost(&hostPtr, numFromDevice * sizeof(TrackDataWithIDs)));
  COPCORE_CUDA_CHECK(cudaMalloc(&devPtr, numFromDevice * sizeof(TrackDataWithIDs)));

  fromDevice_host.reset(hostPtr);
  fromDevice_dev.reset(devPtr);

  unsigned int *nFromDevice = nullptr;
  COPCORE_CUDA_CHECK(cudaMallocHost(&nFromDevice, sizeof(unsigned int)));
  nFromDevice_host.reset(nFromDevice);
  unsigned int *nRemainingLeaks = nullptr;
  COPCORE_CUDA_CHECK(cudaMallocHost(&nRemainingLeaks, sizeof(unsigned int)));
  nRemainingLeaks_host.reset(nRemainingLeaks);

  toDeviceBuffer[0].tracks    = toDevice_host.get();
  toDeviceBuffer[0].maxTracks = numToDevice;
  toDeviceBuffer[0].nTrack    = 0;
  toDeviceBuffer[1].tracks    = toDevice_host.get() + numToDevice;
  toDeviceBuffer[1].maxTracks = numToDevice;
  toDeviceBuffer[1].nTrack    = 0;
}

} // namespace AsyncAdePT

#endif // ASYNC_ADEPT_TRANSPORT_CUH
