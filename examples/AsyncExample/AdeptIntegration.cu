// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "AdeptIntegration.h"
#include "AdeptIntegration.cuh"

#include "TrackTransfer.h"

#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

#include <AdePT/base/Atomic.h>
#include <AdePT/base/MParray.h>
#include <AdePT/navigation/BVHNavigator.h>
#include <AdePT/integration/AdePTGeant4Integration.hh>

#include <AdePT/copcore/Global.h>
#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/copcore/Ranluxpp.h>

#include <G4Threading.hh>
#include <G4TransportationManager.hh>
#include <G4UniformMagField.hh>
#include <G4FieldManager.hh>

#include <G4HepEmState.hh>
#include <G4HepEmData.hh>
#include <G4HepEmState.hh>
#include <G4HepEmStateInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmMatCutData.hh>

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>

#include "electrons.cuh"
#include "gammas.cuh"

#include <AdePT/benchmarking/NVTX.h>

using namespace AsyncAdePT;

namespace AsyncAdePT {

__constant__ __device__ struct G4HepEmParameters g4HepEmPars;
__constant__ __device__ struct G4HepEmData g4HepEmData;

__constant__ __device__ adeptint::VolAuxData *gVolAuxData = nullptr;
__constant__ __device__ double BzFieldValue               = 0;

/// Transfer volume auxiliary data to GPU
void InitVolAuxArray(adeptint::VolAuxArray &array)
{
  using adeptint::VolAuxData;
  COPCORE_CUDA_CHECK(cudaMalloc(&array.fAuxData_dev, sizeof(VolAuxData) * array.fNumVolumes));
  COPCORE_CUDA_CHECK(
      cudaMemcpy(array.fAuxData_dev, array.fAuxData, sizeof(VolAuxData) * array.fNumVolumes, cudaMemcpyHostToDevice));
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(gVolAuxData, &array.fAuxData_dev, sizeof(VolAuxData *)));
}

/// Communication with the hit processing thread.
struct HitProcessingContext {
  cudaStream_t hitTransferStream;
  std::condition_variable cv{};
  std::mutex mutex{};
  std::atomic_bool keepRunning = true;
};

} // namespace AsyncAdePT

AdeptIntegration::AdeptIntegration(unsigned short nThread, unsigned int trackCapacity, unsigned int hitBufferCapacity,
                                   int debugLevel, std::vector<std::string> const *GPURegionNames,
                                   bool trackInAllRegions)
    : fNThread{nThread}, fTrackCapacity{trackCapacity}, fScoringCapacity{hitBufferCapacity}, fDebugLevel{debugLevel},
      fG4Integrations(nThread), fEventStates(nThread), fGPUNetEnergy(nThread, 0.),
      fTrackInAllRegions{trackInAllRegions}, fGPURegionNames{GPURegionNames}
{
  if (nThread > kMaxThreads)
    throw std::invalid_argument("AdeptIntegration limited to " + std::to_string(kMaxThreads) + " threads");

  for (auto &eventState : fEventStates) {
    std::atomic_init(&eventState, EventState::ScoringRetrieved);
  }

  AdeptIntegration::FullInit();
}

AdeptIntegration::~AdeptIntegration()
{
  FreeGPU();
}

static G4HepEmState *InitG4HepEm()
{
  auto state = new G4HepEmState;
  InitG4HepEmState(state);

  G4HepEmMatCutData *cutData = state->fData->fTheMatCutData;
  G4cout << "fNumG4MatCuts = " << cutData->fNumG4MatCuts << ", fNumMatCutData = " << cutData->fNumMatCutData << G4endl;

  // Copy to GPU.
  CopyG4HepEmDataToGPU(state->fData);
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmPars, state->fParameters, sizeof(G4HepEmParameters)));

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

// Kernel to initialize the set of queues per particle type.
__global__ void InitParticleQueues(ParticleQueues queues, size_t Capacity)
{
  adept::MParray::MakeInstanceAt(Capacity, queues.currentlyActive);
  adept::MParray::MakeInstanceAt(Capacity, queues.nextActive);
  adept::MParray::MakeInstanceAt(Capacity, queues.leakedTracksCurrent);
  adept::MParray::MakeInstanceAt(Capacity, queues.leakedTracksNext);
}

// Init a queue at the designated location
template <typename T>
__global__ void InitQueue(adept::MParrayT<T> *queue, size_t Capacity)
{
  adept::MParrayT<T>::MakeInstanceAt(Capacity, queue);
}

// Kernel function to initialize tracks comming from a Geant4 buffer
__global__ void InjectTracks(AsyncAdePT::TrackDataWithIDs *trackinfo, int ntracks, Secondaries secondaries,
                             const vecgeom::VPlacedVolume *world, adept::MParrayT<QueueIndexPair> *toBeEnqueued,
                             uint64_t initialSeed)
{
  constexpr double tolerance = 10. * vecgeom::kTolerance;
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
    const auto slot = generator->fSlotManager->NextSlot();
#if false
    printf("\t%d,%d: Obtained slot %d for track %d/%d (%d, %d, %d, %d). Slots: (%d %d %d)\n", blockIdx.x, threadIdx.x,
           slot, i, ntracks, trackInfo.eventId, trackInfo.threadId, trackInfo.trackId, trackInfo.pdg,
           secondaries.electrons.fSlotManager->OccupiedSlots(), secondaries.positrons.fSlotManager->OccupiedSlots(),
           secondaries.gammas.fSlotManager->OccupiedSlots());
#endif
    toBeEnqueued->push_back(QueueIndexPair{slot, queueIndex});
    Track &track = generator->fTracks[slot];
    track.rngState.SetSeed(initialSeed * trackInfo.eventId + trackInfo.trackId);
    track.energy       = trackInfo.eKin;
    track.numIALeft[0] = -1.0;
    track.numIALeft[1] = -1.0;
    track.numIALeft[2] = -1.0;

    track.initialRange       = -1.0;
    track.dynamicRangeFactor = -1.0;
    track.tlimitMin          = -1.0;

    track.pos = {trackInfo.position[0], trackInfo.position[1], trackInfo.position[2]};
    track.dir = {trackInfo.direction[0], trackInfo.direction[1], trackInfo.direction[2]};

    track.eventId  = trackInfo.eventId;
    track.threadId = trackInfo.threadId;

    // We locate the pushed point because we run the risk that the
    // point is not located in the GPU region
#ifdef NDEBUG
    constexpr int maxAttempt = 2;
#else
    constexpr int maxAttempt = 10;
#endif
    for (int attempt = 1; attempt < maxAttempt; ++attempt) {
      const auto amount = attempt < 5 ? attempt : (attempt - 5) * -1;
      track.navState.Clear();
      const auto pushedPosition = track.pos + amount * tolerance * track.dir;
      BVHNavigator::LocatePointIn(world, pushedPosition, track.navState, true);
      // The track must be on boundary at this point
      track.navState.SetBoundaryState(true);
      // nextState is initialized as needed.
#ifndef NDEBUG
      const vecgeom::VPlacedVolume *volume = track.navState.Top();
      int lvolID                           = volume->GetLogicalVolume()->id();
      adeptint::VolAuxData const &auxData  = AsyncAdePT::gVolAuxData[lvolID];
      if (auxData.fGPUregion && attempt == 1) {
        break;
      } else {
        printf("Error [%d, %d]: ev=%d track=%d thread=%d: GPUregion=%d volID=%d "
               "x=(%18.15f, %18.15f, %18.15f) dir=(%f, %f, %f) "
               "Safety=%17.15f DistanceToOut=%f shiftAmount=%d\n",
               blockIdx.x, threadIdx.x, trackInfo.eventId, trackInfo.trackId, trackInfo.threadId, auxData.fGPUregion,
               volume->id(), pushedPosition[0], pushedPosition[1], pushedPosition[2], track.dir[0], track.dir[1],
               track.dir[2], BVHNavigator::ComputeSafety(pushedPosition, track.navState),
               volume->DistanceToOut(track.pos, track.dir), amount);
        track.navState.Print();
        if (auxData.fGPUregion) {
          printf("Success in attempt %d shiftAmount %d\n", attempt, amount);
          break;
        }
      }
#endif
    }
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

// Copy particles leaked from the GPU region into a compact buffer
__global__ void FillFromDeviceBuffer(AllLeaked all, AsyncAdePT::TrackDataWithIDs *fromDevice,
                                     unsigned int maxFromDeviceBuffer)
{
  const auto numElectrons = all.leakedElectrons.fLeakedQueue->size();
  const auto numPositrons = all.leakedPositrons.fLeakedQueue->size();
  const auto numGammas    = all.leakedGammas.fLeakedQueue->size();
  const auto total        = numElectrons + numPositrons + numGammas;
  if (blockIdx.x == 0 && threadIdx.x == 0) nFromDevice_dev = total < maxFromDeviceBuffer ? total : maxFromDeviceBuffer;

  for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < total; i += blockDim.x * gridDim.x) {
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

    const auto trackSlot = (*leakedTracks->fLeakedQueue)[queueSlot];
    Track const *const track = leakedTracks->fTracks + trackSlot;

    if (i >= maxFromDeviceBuffer) {
      // No space to transfer it out
      leakedTracks->fLeakedQueueNext->push_back(trackSlot);
    } else {
      fromDevice[i].position[0]  = track->pos[0];
      fromDevice[i].position[1]  = track->pos[1];
      fromDevice[i].position[2]  = track->pos[2];
      fromDevice[i].direction[0] = track->dir[0];
      fromDevice[i].direction[1] = track->dir[1];
      fromDevice[i].direction[2] = track->dir[2];
      fromDevice[i].eKin         = track->energy;
      fromDevice[i].globalTime   = track->globalTime;
      fromDevice[i].localTime    = track->localTime;
      fromDevice[i].properTime   = track->properTime;
      fromDevice[i].pdg          = pdg;
      fromDevice[i].eventId      = track->eventId;
      fromDevice[i].threadId     = track->threadId;

      leakedTracks->fSlotManager->MarkSlotForFreeing(trackSlot);
    }
  }
}

__global__ void FreeSlots(TracksAndSlots tracksAndSlots)
{
  constexpr auto nSlotMgrs = sizeof(tracksAndSlots.slotManagers) / sizeof(tracksAndSlots.slotManagers[0]);
  for (unsigned int i = blockIdx.x; i < nSlotMgrs; i += gridDim.x) {
    tracksAndSlots.slotManagers[i]->FreeMarkedSlots();
  }
}

// Finish iteration: clear queues and fill statistics.
__global__ void FinishIteration(AllParticleQueues all, Stats *stats, TracksAndSlots tracksAndSlots,
                                GammaInteractions gammaInteractions)
{
  if (blockIdx.x == 0) {
    // Clear queues and write statistics
    for (int i = threadIdx.x; i < ParticleType::NumParticleTypes; i += blockDim.x) {
      all.queues[i].currentlyActive->clear();
      stats->inFlight[i]     = all.queues[i].nextActive->size();
      stats->leakedTracks[i] = all.queues[i].leakedTracksCurrent->size() + all.queues[i].leakedTracksNext->size();
      stats->queueFillLevel[i] = tracksAndSlots.slotManagers[i]->FillLevel();
    }
  } else if (blockIdx.x == 1) {
    // Assert that there is enough slots allocated:
    for (int i = threadIdx.x; i < ParticleType::NumParticleTypes; i += blockDim.x) {
      if (all.queues[i].nextActive->size() > tracksAndSlots.slotManagers[i]->OccupiedSlots()) {
        printf("Error particle type %d: %ld in flight while %d slots allocated\n", i, all.queues[i].nextActive->size(),
               tracksAndSlots.slotManagers[i]->OccupiedSlots());
        asm("trap;");
      }
    }
  } else if (blockIdx.x == 2) {
    if (threadIdx.x < gammaInteractions.NInt) {
      gammaInteractions.queues[threadIdx.x]->clear();
    }
    if (threadIdx.x == 0) {
      stats->hitBufferOccupancy = AsyncAdePT::gHitScoringBuffer_dev.fSlotCounter;
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
}

/**
 * Count how many tracks are currently in flight for each event.
 */
__global__ void CountCurrentPopulation(AllParticleQueues all, Stats *stats, TracksAndSlots tracksAndSlots)
{
  constexpr unsigned int N = AdeptIntegration::kMaxThreads;
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
  for (unsigned int queueIndex = blockIdx.x; queueIndex < nQueue; queueIndex += gridDim.x) {
    const auto particleType   = queueIndex / 2;
    Track const *const tracks = tracksAndSlots.tracks[particleType];
    auto const queue = queueIndex < ParticleType::NumParticleTypes ? all.queues[particleType].leakedTracksCurrent
                                                                   : all.queues[particleType].leakedTracksNext;
    const auto end   = queue->size();
    for (unsigned int i = threadIdx.x; i < end; i += blockDim.x) {
      const auto slot     = (*queue)[i];
      const auto threadId = tracks[slot].threadId;
      atomicAdd(stats->perEventLeaked + threadId, 1u);
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

bool AdeptIntegration::InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world)
{
#ifndef NDEBUG
  COPCORE_CUDA_CHECK(vecgeom::cxx::CudaDeviceSetStackLimit(16384 * 2));
#else
  COPCORE_CUDA_CHECK(vecgeom::cxx::CudaDeviceSetStackLimit(16384));
#endif

  // Upload geometry to GPU.
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  cudaManager.LoadGeometry(world);
  auto world_dev = cudaManager.Synchronize();
  // Initialize BVH
  InitBVH();

  return (world_dev != nullptr);
}

bool AdeptIntegration::InitializePhysics(double bz)
{
  // Initialize shared physics data
  fg4hepem_state.reset(InitG4HepEm());

  // Initialize field
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(BzFieldValue, &bz, sizeof(double)));

  return true;
}

namespace {
void allocToDeviceTrackData(GPUstate &gpuState, unsigned int numToDevice)
{
  gpuState.fNumToDevice = numToDevice;
  TrackDataWithIDs *devPtr, *hostPtr;
  COPCORE_CUDA_CHECK(cudaMallocHost(
      &hostPtr, 2 * numToDevice * sizeof(TrackDataWithIDs))); // Double the size to switch between buffers
  COPCORE_CUDA_CHECK(cudaMalloc(&devPtr, numToDevice * sizeof(TrackDataWithIDs)));
  gpuState.toDevice_host = {hostPtr, cudaHostDeleter};
  gpuState.toDevice_dev  = {devPtr, cudaDeleter};
}

void allocFromDeviceTrackData(GPUstate &gpuState, unsigned int numFromDevice)
{
  gpuState.fNumFromDevice = numFromDevice;
  TrackDataWithIDs *devPtr, *hostPtr;
  COPCORE_CUDA_CHECK(cudaMallocHost(&hostPtr, numFromDevice * sizeof(TrackDataWithIDs)));
  COPCORE_CUDA_CHECK(cudaMalloc(&devPtr, numFromDevice * sizeof(TrackDataWithIDs)));
  gpuState.fromDevice_host = {hostPtr, cudaHostDeleter};
  gpuState.fromDevice_dev  = {devPtr, cudaDeleter};
}
} // namespace

void AdeptIntegration::InitializeGPU()
{
  fGPUstate          = std::make_unique<GPUstate>();
  GPUstate &gpuState = *fGPUstate;

  // Allocate structures to manage tracks of an implicit type:
  //  * memory to hold the actual Track elements,
  //  * objects to manage slots inside the memory,
  //  * queues of slots to remember active particle and those needing relocation,
  //  * a stream and an event for synchronization of kernels.

  auto gpuMalloc = [&gpuState](auto &devPtr, std::size_t N, bool emplaceForAutoDelete = true) {
    COPCORE_CUDA_CHECK(cudaMalloc(&devPtr, sizeof(*devPtr) * N));
    if (emplaceForAutoDelete) gpuState.allCudaPointers.emplace_back(devPtr, cudaDeleter);
  };

  SlotManager *slotManagers_dev = nullptr;
  gpuMalloc(slotManagers_dev, ParticleType::NumParticleTypes);

  // Create a stream to synchronize kernels of all particle types.
  COPCORE_CUDA_CHECK(cudaStreamCreate(&gpuState.stream));
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    ParticleType &particleType = gpuState.particles[i];
    const size_t nSlot         = fTrackCapacity * ParticleType::relativeQueueSize[i];
    std::cout << "Queue size for " << i << ": " << nSlot << "\n";
    const size_t sizeOfTrackStorage = sizeof(Track) * nSlot;
    const size_t sizeOfQueueStorage = adept::MParray::SizeOfInstance(nSlot);

    COPCORE_CUDA_CHECK(cudaMalloc(&particleType.tracks, sizeOfTrackStorage));

    particleType.slotManager_host =
        SlotManager{static_cast<SlotManager::value_type>(nSlot), static_cast<SlotManager::value_type>(nSlot)};
    particleType.slotManager      = slotManagers_dev + i;
    COPCORE_CUDA_CHECK(
        cudaMemcpy(particleType.slotManager, &particleType.slotManager_host, sizeof(SlotManager), cudaMemcpyDefault));

    COPCORE_CUDA_CHECK(cudaMalloc(&particleType.queues.currentlyActive, sizeOfQueueStorage));
    COPCORE_CUDA_CHECK(cudaMalloc(&particleType.queues.nextActive, sizeOfQueueStorage));
    COPCORE_CUDA_CHECK(cudaMalloc(&particleType.queues.leakedTracksCurrent, sizeOfQueueStorage));
    COPCORE_CUDA_CHECK(cudaMalloc(&particleType.queues.leakedTracksNext, sizeOfQueueStorage));
    InitParticleQueues<<<1, 1>>>(particleType.queues, nSlot);

    COPCORE_CUDA_CHECK(cudaStreamCreate(&particleType.stream));
    COPCORE_CUDA_CHECK(cudaEventCreate(&particleType.event));
  }

  // init gamma interaction queues
  for (unsigned int i = 0; i < GammaInteractions::NInt; ++i) {
    const auto capacity     = fTrackCapacity / 6;
    const auto instanceSize = adept::MParrayT<GammaInteractions::Data>::SizeOfInstance(capacity);
    COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.gammaInteractions.queues[i], instanceSize));
    InitQueue<GammaInteractions::Data><<<1, 1>>>(gpuState.gammaInteractions.queues[i], capacity);
  }

  // initialize statistics
  COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.stats_dev, sizeof(Stats)));
  COPCORE_CUDA_CHECK(cudaMallocHost(&gpuState.stats, sizeof(Stats)));

  // init scoring structures
  PerEventScoring *deviceScoring;
  gpuMalloc(deviceScoring, fNThread, false);
  gpuState.fScoring_dev.reset(deviceScoring);

  fScoring.clear();
  fScoring.reserve(fNThread);
  for (unsigned int i = 0; i < fNThread; ++i) {
    fScoring.emplace_back(deviceScoring + i);
  }
  gpuState.fHitScoring.reset(new HitScoring(fScoringCapacity, fNThread));

  // initialize buffers for track transfer on host and device
  allocToDeviceTrackData(gpuState, gpuState.fNumToDevice);
  allocFromDeviceTrackData(gpuState, gpuState.fNumFromDevice);
  unsigned int *nFromDevice_host = nullptr;
  COPCORE_CUDA_CHECK(cudaMallocHost(&nFromDevice_host, sizeof(unsigned int)));
  gpuState.nFromDevice.reset(nFromDevice_host);

  fBuffer = std::make_unique<TrackBuffer>(gpuState.toDevice_host.get(), gpuState.fNumToDevice,
                                          gpuState.toDevice_host.get() + gpuState.fNumToDevice, gpuState.fNumToDevice,
                                          fNThread);

  const auto injectQueueSize = adept::MParrayT<QueueIndexPair>::SizeOfInstance(gpuState.fNumToDevice);
  adept::MParrayT<QueueIndexPair> *injectQueue;
  COPCORE_CUDA_CHECK(cudaMalloc(&injectQueue, injectQueueSize));
  gpuState.injectionQueue = {injectQueue, cudaDeleter};
  InitQueue<QueueIndexPair><<<1, 1>>>(gpuState.injectionQueue.get(), gpuState.fNumToDevice);
}

void AdeptIntegration::FreeGPU()
{
  fGPUstate->runTransport = false;
  fGPUWorker.join();

  adeptint::VolAuxData *volAux = nullptr;
  COPCORE_CUDA_CHECK(cudaMemcpyFromSymbol(&volAux, AsyncAdePT::gVolAuxData, sizeof(adeptint::VolAuxData *)));
  COPCORE_CUDA_CHECK(cudaFree(volAux));

  // Free resources.
  GPUstate &gpuState = const_cast<GPUstate &>(*fGPUstate);
  COPCORE_CUDA_CHECK(cudaFree(gpuState.stats_dev));
  COPCORE_CUDA_CHECK(cudaFreeHost(gpuState.stats));

  COPCORE_CUDA_CHECK(cudaStreamDestroy(gpuState.stream));

  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    COPCORE_CUDA_CHECK(cudaFree(gpuState.particles[i].tracks));

    COPCORE_CUDA_CHECK(cudaFree(gpuState.particles[i].queues.currentlyActive));
    COPCORE_CUDA_CHECK(cudaFree(gpuState.particles[i].queues.nextActive));
    COPCORE_CUDA_CHECK(cudaFree(gpuState.particles[i].queues.leakedTracksCurrent));
    COPCORE_CUDA_CHECK(cudaFree(gpuState.particles[i].queues.leakedTracksNext));

    COPCORE_CUDA_CHECK(cudaStreamDestroy(gpuState.particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventDestroy(gpuState.particles[i].event));
  }

  for (auto queue : gpuState.gammaInteractions.queues) {
    COPCORE_CUDA_CHECK(cudaFree(queue));
  }

  gpuState.allCudaPointers.clear();

  // Free G4HepEm data
  FreeG4HepEmData(AdeptIntegration::fg4hepem_state->fData);
}

void AdeptIntegration::ReturnTracksToG4()
{
  const auto &fromDevice               = fGPUstate->fromDevice_host.get();
  TrackDataWithIDs const *const fromDeviceEnd = fromDevice + *fGPUstate->nFromDevice;
  std::scoped_lock lock{fBuffer->fromDeviceMutex};

  for (TrackDataWithIDs *trackIt = fromDevice; trackIt < fromDeviceEnd; ++trackIt) {
    assert(0 <= trackIt->threadId && trackIt->threadId <= fNThread);
    fBuffer->fromDeviceBuffers[trackIt->threadId].push_back(*trackIt);
  }

  AdvanceEventStates(EventState::FlushingTracks, EventState::DeviceFlushed);
  fGPUstate->extractState = GPUstate::ExtractState::Idle;

#ifndef NDEBUG
  for (const auto &trackBuffer : fBuffer->fromDeviceBuffers) {
    if (trackBuffer.empty()) continue;
    const auto eventId = trackBuffer.front().eventId;
    assert(std::all_of(trackBuffer.begin(), trackBuffer.end(),
                       [eventId](const TrackDataWithIDs &track) { return eventId == track.eventId; }));
  }
#endif
}

void AdeptIntegration::AdvanceEventStates(EventState oldState, EventState newState)
{
  for (auto &eventState : fEventStates) {
    EventState expected = oldState;
    eventState.compare_exchange_strong(expected, newState, std::memory_order_release, std::memory_order_relaxed);
  }
}

void AdeptIntegration::TransportLoop()
{
  NVTXTracer tracer{"TransportLoop"};

  using InjectState  = GPUstate::InjectState;
  using ExtractState                            = GPUstate::ExtractState;
  auto &cudaManager                             = vecgeom::cxx::CudaManager::Instance();
  const vecgeom::cuda::VPlacedVolume *world_dev = cudaManager.world_gpu();
  GPUstate &gpuState                            = *fGPUstate;

  ParticleType &electrons = gpuState.particles[ParticleType::Electron];
  ParticleType &positrons = gpuState.particles[ParticleType::Positron];
  ParticleType &gammas    = gpuState.particles[ParticleType::Gamma];

  cudaEvent_t cudaEvent, cudaStatsEvent;
  cudaStream_t transferStream, statsStream, interactionStream;
  COPCORE_CUDA_CHECK(cudaEventCreateWithFlags(&cudaEvent, cudaEventDisableTiming));
  COPCORE_CUDA_CHECK(cudaEventCreateWithFlags(&cudaStatsEvent, cudaEventDisableTiming));
  unique_ptr_cudaEvent cudaEventCleanup{&cudaEvent, cudaEventDeleter};
  unique_ptr_cudaEvent cudaStatsEventCleanup{&cudaStatsEvent, cudaEventDeleter};
  COPCORE_CUDA_CHECK(cudaStreamCreate(&transferStream));
  COPCORE_CUDA_CHECK(cudaStreamCreate(&statsStream));
  COPCORE_CUDA_CHECK(cudaStreamCreate(&interactionStream));
  std::unique_ptr<cudaStream_t, decltype(&cudaStreamDeleter)> cudaStreamCleanup{&transferStream, cudaStreamDeleter};
  std::unique_ptr<cudaStream_t, decltype(&cudaStreamDeleter)> cudaStatsStreamCleanup{&statsStream, cudaStreamDeleter};
  std::unique_ptr<cudaStream_t, decltype(&cudaStreamDeleter)> cudaInteractionStreamCleanup{&interactionStream,
                                                                                           cudaStreamDeleter};
  auto waitForOtherStream = [&cudaEvent](cudaStream_t waitingStream, cudaStream_t streamToWaitFor) {
    COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, streamToWaitFor));
    COPCORE_CUDA_CHECK(cudaStreamWaitEvent(waitingStream, cudaEvent));
  };

  std::unique_ptr<HitProcessingContext> hitProcessing{new HitProcessingContext{transferStream}};
  std::thread hitProcessingThread{&AdeptIntegration::HitProcessingLoop, this, hitProcessing.get()};

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

  SlotManager *const slotMgrArray = gpuState.particles[0].slotManager;
  while (gpuState.runTransport) {
    NVTXTracer nvtx1{"Setup"}, nvtx2{"Setup2"};
    InitSlotManagers<<<80, 256, 0, gpuState.stream>>>(slotMgrArray, ParticleType::NumParticleTypes);
    COPCORE_CUDA_CHECK(cudaMemsetAsync(gpuState.stats_dev, 0, sizeof(Stats), gpuState.stream));

    int inFlight                                                   = 0;
    unsigned int numLeaked                                         = 0;
    unsigned int particlesInFlight[ParticleType::NumParticleTypes] = {1, 1, 1};

    auto needTransport = [](std::atomic<EventState> const &state) {
      return state.load(std::memory_order_acquire) < EventState::LeakedTracksRetrieved;
    };
    // Wait for work from G4 workers:
    while (gpuState.runTransport && std::none_of(fEventStates.begin(), fEventStates.end(), needTransport)) {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(10ms);
    }

    if (fDebugLevel > 2) {
      G4cout << "GPU transport starting" << std::endl;
    }

    COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

#ifdef USE_NVTX
    std::map<AdeptIntegration::EventState, std::string> stateMap{
        {EventState::NewTracksFromG4, "NewTracksFromG4"},
        {EventState::G4RequestsFlush, "G4RequestsFlush"},
        {EventState::Inject, "Inject"},
        {EventState::InjectionCompleted, "InjectionCompleted"},
        {EventState::Transporting, "Transporting"},
        {EventState::WaitingForTransportToFinish, "WaitingForTransportToFinish"},
        {EventState::RequestHitFlush, "RequestHitFlush"},
        {EventState::HitsFlushed, "HitsFlushed"},
        {EventState::FlushingTracks, "FlushingTracks"},
        {EventState::DeviceFlushed, "DeviceFlushed"},
        {EventState::LeakedTracksRetrieved, "LeakedTracksRetrieved"},
        {EventState::ScoringRetrieved, "ScoringRetrieved"}};
#endif

    for (unsigned int iteration = 0;
         inFlight > 0 || gpuState.injectState != InjectState::Idle || gpuState.extractState != ExtractState::Idle ||
         std::any_of(fEventStates.begin(), fEventStates.end(), needTransport);
         ++iteration) {
#ifdef USE_NVTX
      nvtx1.setTag(stateMap[fEventStates[0].load()].data());
      nvtx2.setTag(stateMap[fEventStates[1].load()].data());
#endif

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
      const TracksAndSlots tracksAndSlots       = {{electrons.tracks, positrons.tracks, gammas.tracks},
                                                   {electrons.slotManager, positrons.slotManager, gammas.slotManager}};

      // --------------------------
      // *** Particle injection ***
      // --------------------------
      if (gpuState.injectState == InjectState::Idle) {
        for (auto &eventState : fEventStates) {
          if (const auto state = eventState.load(std::memory_order_acquire); state == EventState::G4RequestsFlush) {
            eventState = EventState::Inject;
          } else if (state == EventState::Inject) {
            eventState = EventState::InjectionCompleted;
          }
        }

        if (auto &toDevice = fBuffer->getActiveBuffer(); toDevice.nTrack > 0) {
          gpuState.injectState = InjectState::CreatingSlots;

          fBuffer->swapToDeviceBuffers();
          std::scoped_lock lock{toDevice.mutex};
          const auto nInject = std::min(toDevice.nTrack.load(), toDevice.maxTracks);
          toDevice.nTrack    = 0;

          if (fDebugLevel > 3) std::cout << "Injecting " << nInject << " to GPU\n";

          // copy buffer of tracks to device
          COPCORE_CUDA_CHECK(cudaMemcpyAsync(gpuState.toDevice_dev.get(), toDevice.tracks,
                                             nInject * sizeof(TrackDataWithIDs), cudaMemcpyHostToDevice,
                                             transferStream));
          // Mark end of copy operation:
          COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, transferStream));

          // Inject AdePT tracks using the track buffer
          constexpr auto injectThreads = 128u;
          const auto injectBlocks      = (nInject + injectThreads - 1) / injectThreads;
          InjectTracks<<<injectBlocks, injectThreads, 0, transferStream>>>(
              gpuState.toDevice_dev.get(), nInject, secondaries, world_dev, gpuState.injectionQueue.get(), fAdePTSeed);
          COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
              transferStream,
              [](void *arg) { (*static_cast<decltype(GPUstate::injectState) *>(arg)) = InjectState::ReadyToEnqueue; },
              &gpuState.injectState));

          // Ensure that copy operation completed before releasing lock on to-device buffer
          COPCORE_CUDA_CHECK(cudaEventSynchronize(cudaEvent));
        } else {
          gpuState.injectState = InjectState::Idle;
        }
      }

      // *** Enqueue particles that are ready on the device ***
      if (gpuState.injectState == InjectState::ReadyToEnqueue) {
        gpuState.injectState = InjectState::Enqueueing;
        EnqueueTracks<<<1, 256, 0, gpuState.stream>>>(allParticleQueues, gpuState.injectionQueue.get());
        // New injection has to wait until particles are enqueued:
        waitForOtherStream(transferStream, gpuState.stream);
      } else if (gpuState.injectState == InjectState::Enqueueing) {
        gpuState.injectState = InjectState::Idle;
      }

      // ------------------
      // *** Transport ***
      // ------------------

      // *** ELECTRONS ***
      {
        const auto [threads, blocks] = computeThreadsAndBlocks(particlesInFlight[ParticleType::Electron]);
        TransportElectrons<PerEventScoring><<<blocks, threads, 0, electrons.stream>>>(
            electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive,
            electrons.queues.leakedTracksCurrent, gpuState.fScoring_dev.get());

        COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, electrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, electrons.event, 0));
      }

      // *** POSITRONS ***
      {
        const auto [threads, blocks] = computeThreadsAndBlocks(particlesInFlight[ParticleType::Positron]);
        TransportPositrons<PerEventScoring><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive,
            positrons.queues.leakedTracksCurrent, gpuState.fScoring_dev.get());

        COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, positrons.event, 0));
      }

      // *** GAMMAS ***
      {
        const auto [threads, blocks] = computeThreadsAndBlocks(particlesInFlight[ParticleType::Gamma]);
        TransportGammas<PerEventScoring><<<blocks, threads, 0, gammas.stream>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive,
            gammas.queues.leakedTracksCurrent, gpuState.fScoring_dev.get(), gpuState.gammaInteractions);

        constexpr unsigned int intThreads = 128;
        ApplyGammaInteractions<PerEventScoring>
            <<<dim3(20, 3, 1), intThreads, 0, gammas.stream>>>(gammas.tracks, secondaries, gammas.queues.nextActive,
                                                               gpuState.fScoring_dev.get(), gpuState.gammaInteractions);

        COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, gammas.event, 0));
      }

      // ---------------------------------------
      // *** Count detailed event statistics ***
      // ---------------------------------------
      {
        AdvanceEventStates(EventState::Transporting, EventState::WaitingForTransportToFinish);
        AdvanceEventStates(EventState::InjectionCompleted, EventState::Transporting);

        // Reset all counters count the currently flying population
        ZeroEventCounters<<<1, 256, 0, statsStream>>>(gpuState.stats_dev);
        CountCurrentPopulation<<<ParticleType::NumParticleTypes, 128, 0, statsStream>>>(
            allParticleQueues, gpuState.stats_dev, tracksAndSlots);
        // Count leaked tracks. Note that new tracks might be added while/after we count:
        CountLeakedTracks<<<2 * ParticleType::NumParticleTypes, 128, 0, statsStream>>>(
            allParticleQueues, gpuState.stats_dev, tracksAndSlots);

        waitForOtherStream(gpuState.stream, statsStream);

        // Get results to host:
        COPCORE_CUDA_CHECK(
            cudaMemcpyAsync(gpuState.stats, gpuState.stats_dev, sizeof(Stats), cudaMemcpyDeviceToHost, statsStream));
        COPCORE_CUDA_CHECK(cudaEventRecord(cudaStatsEvent, statsStream));
      }

      // -------------------------
      // *** Collect particles ***
      // -------------------------

      if (gpuState.extractState == ExtractState::Idle &&
          std::any_of(fEventStates.begin(), fEventStates.end(), [](const auto &eventState) {
            return eventState.load(std::memory_order_acquire) == EventState::HitsFlushed;
          })) {
        gpuState.extractState = ExtractState::FreeingSlots;
        AdvanceEventStates(EventState::HitsFlushed, EventState::FlushingTracks);

        const AllLeaked allLeaked = {.leakedElectrons = {electrons.tracks, electrons.queues.leakedTracksCurrent,
                                                         electrons.queues.leakedTracksNext, electrons.slotManager},
                                     .leakedPositrons = {positrons.tracks, positrons.queues.leakedTracksCurrent,
                                                         positrons.queues.leakedTracksNext, positrons.slotManager},
                                     .leakedGammas    = {gammas.tracks, gammas.queues.leakedTracksCurrent,
                                                         gammas.queues.leakedTracksNext, gammas.slotManager}};

        // Ensure that transport that's writing to the old queues finishes before collecting leaked tracks
        for (auto const &event : {electrons.event, positrons.event, gammas.event}) {
          COPCORE_CUDA_CHECK(cudaStreamWaitEvent(transferStream, event));
        }

        // Populate the staging buffer and copy to host
        constexpr unsigned int block_size = 128;
        const unsigned int grid_size      = (gpuState.fNumFromDevice + block_size - 1) / block_size;
        FillFromDeviceBuffer<<<grid_size, block_size, 0, transferStream>>>(allLeaked, gpuState.fromDevice_dev.get(),
                                                                           gpuState.fNumFromDevice);
        COPCORE_CUDA_CHECK(cudaMemcpyFromSymbolAsync(gpuState.nFromDevice.get(), nFromDevice_dev, sizeof(unsigned int),
                                                     0, cudaMemcpyDeviceToHost, transferStream));
        COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
            transferStream,
            [](void *arg) { (*static_cast<decltype(GPUstate::extractState) *>(arg)) = ExtractState::ReadyToCopy; },
            &gpuState.extractState));
        ClearQueues<<<1, 1, 0, transferStream>>>(electrons.queues.leakedTracksCurrent,
                                                 positrons.queues.leakedTracksCurrent,
                                                 gammas.queues.leakedTracksCurrent);

        electrons.queues.SwapLeakedQueue();
        positrons.queues.SwapLeakedQueue();
        gammas.queues.SwapLeakedQueue();
      }

      if (gpuState.extractState == ExtractState::ReadyToCopy) {
        gpuState.extractState = ExtractState::CopyToHost;
        COPCORE_CUDA_CHECK(cudaMemcpyAsync(gpuState.fromDevice_host.get(), gpuState.fromDevice_dev.get(),
                                           (*gpuState.nFromDevice) * sizeof(TrackDataWithIDs), cudaMemcpyDeviceToHost,
                                           transferStream));
        COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
            transferStream, [](void *thisPtr) { static_cast<AdeptIntegration *>(thisPtr)->ReturnTracksToG4(); }, this));
      }

      // -------------------------
      // *** Finish iteration ***
      // -------------------------

      // This kernel needs to wait that all of the above work (except for asynchronous particle transfer) is done.
      // Don't forget to synchronise any of the transport or event counting with it.
      FinishIteration<<<4, 32, 0, gpuState.stream>>>(allParticleQueues, gpuState.stats_dev, tracksAndSlots,
                                                     gpuState.gammaInteractions);

      // Try to free slots if one of the queues is half full
      if (gpuState.injectState != InjectState::CreatingSlots) {
        bool needSlotFree = false;
        for (unsigned int i = 0; i < ParticleType::NumParticleTypes; ++i) {
          needSlotFree |= gpuState.stats->queueFillLevel[i] >= 0.5;
        }
        if (needSlotFree) {
          // Freeing of slots has to run exclusively
          waitForOtherStream(gpuState.stream, transferStream);
          FreeSlots<<<ParticleType::NumParticleTypes, 256, 0, gpuState.stream>>>(tracksAndSlots);
          waitForOtherStream(transferStream, gpuState.stream);
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
        cudaError_t result;
        while ((result = cudaEventQuery(cudaStatsEvent)) == cudaErrorNotReady) {
          // Cuda uses a busy wait. This reduces CPU consumption by 50%:
          using namespace std::chrono_literals;
          std::this_thread::sleep_for(50us);
        }
        COPCORE_CUDA_CHECK(result);

        for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
          inFlight += gpuState.stats->inFlight[i];
          numLeaked += gpuState.stats->leakedTracks[i];
          particlesInFlight[i] = gpuState.stats->inFlight[i];
        }

        for (unsigned short threadId = 0; threadId < fNThread; ++threadId) {
          const auto state = fEventStates[threadId].load(std::memory_order_acquire);
          if (state == EventState::WaitingForTransportToFinish && gpuState.stats->perEventInFlight[threadId] == 0) {
            fEventStates[threadId] = EventState::RequestHitFlush;
          }
          if (EventState::RequestHitFlush <= state && state < EventState::LeakedTracksRetrieved &&
              gpuState.stats->perEventInFlight[threadId] != 0) {
            std::cerr << "ERROR thread " << threadId << " is in state " << static_cast<unsigned int>(state)
                      << " and occupancy is " << gpuState.stats->perEventInFlight[threadId] << "\n";
          }
        }

        // *** Hit management ***
        if (!gpuState.fHitScoring->ReadyToSwapBuffers()) {
          hitProcessing->cv.notify_one();
        } else {
          if (gpuState.stats->hitBufferOccupancy >= gpuState.fHitScoring->HitCapacity() / 2 ||
              std::any_of(fEventStates.begin(), fEventStates.end(),
                          [](const auto &state) { return state == EventState::RequestHitFlush; })) {
            AdvanceEventStates(EventState::RequestHitFlush, EventState::FlushingHits);
            gpuState.fHitScoring->SwapDeviceBuffers(gpuState.stream);
            hitProcessing->cv.notify_one();
          }
        }
      }

      // *** Notify G4 workers if their events completed ***
      if (std::any_of(fEventStates.begin(), fEventStates.end(),
                      [](const EventState &state) { return state == EventState::DeviceFlushed; })) {
        fCV_G4Workers.notify_all();
      }

      if (fDebugLevel >= 3 && inFlight > 0) {
        std::cerr << inFlight << " in flight ";
        std::cerr << "(" << gpuState.stats->inFlight[ParticleType::Electron] << " "
                  << gpuState.stats->inFlight[ParticleType::Positron] << " "
                  << gpuState.stats->inFlight[ParticleType::Gamma] << "),\tSlots:(" << std::setprecision(3)
                  << gpuState.stats->queueFillLevel[ParticleType::Electron] << " "
                  << gpuState.stats->queueFillLevel[ParticleType::Positron] << " "
                  << gpuState.stats->queueFillLevel[ParticleType::Gamma] << ")";
        std::cerr << ", " << numLeaked << " leaked."
                  << "\tInjectState: " << static_cast<unsigned int>(gpuState.injectState.load())
                  << "\tExtractState: " << static_cast<unsigned int>(gpuState.extractState.load())
                  << "\tHitBuffer: " << gpuState.stats->hitBufferOccupancy;
        if (fDebugLevel >= 4) {
          std::cerr << "\n\tper event: ";
          for (unsigned int i = 0; i < fNThread; ++i) {
            std::cerr << i << ": " << gpuState.stats->perEventInFlight[i]
                      << " (s=" << static_cast<unsigned short>(fEventStates[i].load(std::memory_order_acquire))
                      << ")\t";
          }
        }
        std::cerr << std::endl;
      }

#ifndef NDEBUG
      // *** Check slots ***
      if (false && iteration % 100 == 0 && gpuState.injectState != InjectState::CreatingSlots &&
          gpuState.extractState != ExtractState::FreeingSlots) {
        AssertConsistencyOfSlotManagers<<<120, 256, 0, gpuState.stream>>>(slotMgrArray, ParticleType::NumParticleTypes);
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

    if (fDebugLevel > 2) std::cout << "End transport loop.\n";
  }

  hitProcessing->keepRunning = false;
  hitProcessing->cv.notify_one();
  hitProcessingThread.join();
}

std::shared_ptr<const std::vector<GPUHit>> AdeptIntegration::GetGPUHits(unsigned int threadId) const
{
  return fGPUstate->fHitScoring->GetNextHitsVector(threadId);
}

void AdeptIntegration::HitProcessingLoop(HitProcessingContext *const context)
{
  while (context->keepRunning) {
    std::unique_lock lock(context->mutex);
    context->cv.wait(lock);

    fGPUstate->fHitScoring->TransferHitsToHost(context->hitTransferStream);
    const bool haveNewHits = fGPUstate->fHitScoring->ProcessHits();

    if (haveNewHits) {
      AdvanceEventStates(EventState::FlushingHits, EventState::HitsFlushed);
      fCV_G4Workers.notify_all();
    }
  }
}
