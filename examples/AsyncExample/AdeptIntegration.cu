// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "AdeptIntegration.h"
#include "AdeptIntegration.cuh"

#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

#include <AdePT/Atomic.h>
#include <AdePT/BVHNavigator.h>
#include <AdePT/MParray.h>

#include <CopCore/Global.h>
#include <CopCore/PhysicalConstants.h>
#include <CopCore/Ranluxpp.h>

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

__constant__ __device__ struct G4HepEmParameters g4HepEmPars;
__constant__ __device__ struct G4HepEmData g4HepEmData;

__constant__ __device__ adeptint::VolAuxData *gVolAuxData = nullptr;
__constant__ __device__ double BzFieldValue               = 0;

G4HepEmState *AdeptIntegration::fg4hepem_state{nullptr};

AdeptIntegration::AdeptIntegration(unsigned short nThread, unsigned int trackCapacity, unsigned int bufferThreshold,
                                   int debugLevel, G4Region *region, std::unordered_map<std::string, int> &sensVolIndex,
                                   std::unordered_map<const G4VPhysicalVolume *, int> &scoringMap)
    : fNThread{nThread}, fTrackCapacity{trackCapacity}, fBufferThreshold{bufferThreshold}, fDebugLevel{debugLevel},
      fRegion{region}, sensitive_volume_index{sensVolIndex}, fScoringMap{scoringMap}, fEventStates(nThread),
      fGPUNetEnergy(nThread, 0.)
{
  if (nThread > kMaxThreads)
    throw std::invalid_argument("AdeptIntegration limited to " + std::to_string(kMaxThreads) + " threads");

  for (auto &eventState : fEventStates) {
    std::atomic_init(&eventState, EventState::ScoringRetrieved);
  }

  AdeptIntegration::Initialize();
}

AdeptIntegration::~AdeptIntegration()
{
  FreeGPU();
}

void AdeptIntegration::VolAuxArray::InitializeOnGPU()
{
  // Transfer volume auxiliary data
  COPCORE_CUDA_CHECK(cudaMalloc(&fAuxData_dev, sizeof(VolAuxData) * fNumVolumes));
  COPCORE_CUDA_CHECK(cudaMemcpy(fAuxData_dev, fAuxData, sizeof(VolAuxData) * fNumVolumes, cudaMemcpyHostToDevice));
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(gVolAuxData, &fAuxData_dev, sizeof(VolAuxData *)));
}

void AdeptIntegration::VolAuxArray::FreeGPU()
{
  COPCORE_CUDA_CHECK(cudaFree(fAuxData_dev));
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
__global__ void InjectTracks(adeptint::TrackData *trackinfo, int ntracks, Secondaries secondaries,
                             const vecgeom::VPlacedVolume *world, AdeptScoring *userScoring,
                             adept::MParrayT<QueueIndexPair> *toBeEnqueued)
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
    track.rngState.SetSeed(1234567 * trackInfo.eventId + trackInfo.trackId);
    track.energy       = trackInfo.energy;
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
      const vecgeom::VPlacedVolume *volume = track.navState.Top();
      int lvolID                           = volume->GetLogicalVolume()->id();
      adeptint::VolAuxData const &auxData  = userScoring[trackInfo.threadId].GetAuxData_dev(lvolID);
#ifndef NDEBUG
      if (auxData.fGPUregion && attempt == 1) {
        break;
      } else {
        printf("Error [%d, %d]: ev=%d track=%d: scoring[tid=%d].GetAux_dev[lvolID=%d].fGPUregion=%d volID=%d "
               "x=(%18.15f, %18.15f, %18.15f) dir=(%f, %f, %f) "
               "Safety=%17.15f DistanceToOut=%f shiftAmount=%d\n",
               blockIdx.x, threadIdx.x, trackInfo.eventId, trackInfo.trackId, trackInfo.threadId, lvolID,
               auxData.fGPUregion, volume->id(), pushedPosition[0], pushedPosition[1], pushedPosition[2], track.dir[0],
               track.dir[1], track.dir[2], BVHNavigator::ComputeSafety(pushedPosition, track.navState),
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
__global__ void FillFromDeviceBuffer(AllLeaked all, adeptint::TrackData *fromDevice, unsigned int maxFromDeviceBuffer)
{
  const auto numElectrons = all.leakedElectrons.fLeakedQueue->size();
  const auto numPositrons = all.leakedPositrons.fLeakedQueue->size();
  const auto numGammas    = all.leakedGammas.fLeakedQueue->size();
  const auto total        = numElectrons + numPositrons + numGammas;

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

    if (i >= maxFromDeviceBuffer) {
      // No space to transfer it out
      leakedTracks->fLeakedQueueNext->push_back(trackSlot);
    } else {
      Track const *const track   = leakedTracks->fTracks + trackSlot;
      fromDevice[i].position[0]  = track->pos[0];
      fromDevice[i].position[1]  = track->pos[1];
      fromDevice[i].position[2]  = track->pos[2];
      fromDevice[i].direction[0] = track->dir[0];
      fromDevice[i].direction[1] = track->dir[1];
      fromDevice[i].direction[2] = track->dir[2];
      fromDevice[i].energy       = track->energy;
      fromDevice[i].pdg          = pdg;
      fromDevice[i].threadId     = track->threadId;
      fromDevice[i].eventId      = track->eventId;

      leakedTracks->fSlotManager->MarkSlotForFreeing(trackSlot);
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) nFromDevice_dev = total;
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
      stats->usedSlots[i]    = tracksAndSlots.slotManagers[i]->OccupiedSlots();
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
  }

#if false
  if (blockIdx.x == 3 && threadIdx.x == 0) {
    printf("In flight (kernel): %ld %ld %ld  %ld\tslots: %d %d %d\n", all.queues[0].nextActive->size(),
            all.queues[1].nextActive->size(), all.queues[2].nextActive->size(),
            all.queues[0].nextActive->size() + all.queues[1].nextActive->size() + all.queues[2].nextActive->size(),
            tracksAndSlots.slotManagers[0]->OccupiedSlots(), tracksAndSlots.slotManagers[1]->OccupiedSlots(),
            tracksAndSlots.slotManagers[2]->OccupiedSlots());
  }
#endif
}

__global__ void ZeroEventCounters(Stats *stats)
{
  constexpr unsigned int size = sizeof(stats->occupancy) / sizeof(stats->occupancy[0]);
  for (unsigned int i = threadIdx.x; i < size; i += blockDim.x) {
    stats->occupancy[i] = 0;
  }
}

// TODO: Use shared mem to reduce atomic operations?
__global__ void FillEventCounters(AllParticleQueues all, Stats *stats, TracksAndSlots tracksAndSlots)
{
  // Count occupancy for each event
  constexpr unsigned int NQueue = 3u;
  for (unsigned int queueCounter = blockIdx.x; queueCounter < NQueue * ParticleType::NumParticleTypes;
       queueCounter += gridDim.x) {
    const auto particleType     = queueCounter / NQueue;
    const auto queueType        = queueCounter % NQueue;
    Track const *const tracks   = tracksAndSlots.tracks[particleType];
    adept::MParray const *queue = queueType == 0 ? all.queues[particleType].currentlyActive
                                                 : (queueType == 1 ? all.queues[particleType].leakedTracksCurrent
                                                                   : all.queues[particleType].leakedTracksNext);

    const auto end = queue->size();
    for (unsigned int i = threadIdx.x; i < end; i += blockDim.x) {
      const auto slot     = (*queue)[i];
      const auto threadId = tracks[slot].threadId;
      atomicAdd(stats->occupancy + threadId, 1u);
    }
  }
}

__global__ void ClearQueue(adept::MParray *queue)
{
  queue->clear();
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

#ifndef NDEBUG
__global__ void AssertConsistencyOfSlotManagers(SlotManager *mgrs, std::size_t N)
{
  for (int i = 0; i < N; ++i) {
    SlotManager &mgr = mgrs[i];
    const auto slotCounter = mgr.fSlotCounter;
    const auto freeCounter = mgr.fFreeCounter;

    if (blockIdx.x == 0 && threadIdx.x == 0 && slotCounter < freeCounter) {
      printf("Error %s:%d: Trying to free %d slots in manager %d whereas only %d allocated\n", __FILE__, __LINE__,
             freeCounter, i, slotCounter);
      for (unsigned int i = 0; i < freeCounter; ++i) {
        printf("%d ", mgr.fToFreeList[i]);
      }
      printf("\n");
      assert(false);
    }

    bool doubleFree = false;
    for (unsigned int j = blockIdx.x; j < mgr.fFreeCounter; j += gridDim.x) {
      const auto slotToSearch = mgr.fToFreeList[j];
      for (unsigned int k = j + 1 + threadIdx.x; k < freeCounter; k += blockDim.x) {
        if (slotToSearch == mgr.fToFreeList[k]) {
          printf("Error: Manager %d: Slot %d freed both at %d and at %d\n", i, slotToSearch, k, j);
          doubleFree = true;
          break;
        }
      }
    }

    assert(slotCounter == mgr.fSlotCounter && "Race condition while checking slots");
    assert(freeCounter == mgr.fFreeCounter && "Race condition while checking slots");
    assert(!doubleFree);
  }
}
#endif

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

bool AdeptIntegration::InitializePhysics()
{
  // Initialize shared physics data
  AdeptIntegration::fg4hepem_state = InitG4HepEm();
  // Initialize field
  double bz = 0;
  auto field =
      (G4UniformMagField *)G4TransportationManager::GetTransportationManager()->GetFieldManager()->GetDetectorField();
  if (field) {
    auto field_vect = field->GetConstantFieldValue();
    bz              = field_vect[2];
  }
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(BzFieldValue, &bz, sizeof(double)));

  return true;
}

namespace {
void allocToDeviceTrackData(GPUstate &gpuState, unsigned int numToDevice)
{
  using TrackData       = adeptint::TrackData;
  gpuState.fNumToDevice = numToDevice;
  TrackData *devPtr, *hostPtr;
  COPCORE_CUDA_CHECK(
      cudaMallocHost(&hostPtr, 2 * numToDevice * sizeof(TrackData))); // Double the size to switch between buffers
  COPCORE_CUDA_CHECK(cudaMalloc(&devPtr, numToDevice * sizeof(TrackData)));
  gpuState.toDevice_host = {hostPtr, adeptint::cudaHostDeleter};
  gpuState.toDevice_dev  = {devPtr, adeptint::cudaDeleter};
}

void allocFromDeviceTrackData(GPUstate &gpuState, unsigned int numFromDevice)
{
  using TrackData         = adeptint::TrackData;
  gpuState.fNumFromDevice = numFromDevice;
  adeptint::TrackData *devPtr, *hostPtr;
  COPCORE_CUDA_CHECK(cudaMallocHost(&hostPtr, numFromDevice * sizeof(TrackData)));
  COPCORE_CUDA_CHECK(cudaMalloc(&devPtr, numFromDevice * sizeof(TrackData)));
  gpuState.fromDevice_host = {hostPtr, adeptint::cudaHostDeleter};
  gpuState.fromDevice_dev  = {devPtr, adeptint::cudaDeleter};
}
} // namespace

void AdeptIntegration::InitializeGPU()
{
  using TrackData    = adeptint::TrackData;
  fGPUstate          = std::make_unique<GPUstate>();
  GPUstate &gpuState = *fGPUstate;

  // Allocate structures to manage tracks of an implicit type:
  //  * memory to hold the actual Track elements,
  //  * objects to manage slots inside the memory,
  //  * queues of slots to remember active particle and those needing relocation,
  //  * a stream and an event for synchronization of kernels.
  size_t TracksSize      = sizeof(Track) * fTrackCapacity;
  const size_t QueueSize = adept::MParray::SizeOfInstance(fTrackCapacity);

  auto gpuMalloc = [&gpuState](auto &devPtr, std::size_t N) {
    COPCORE_CUDA_CHECK(cudaMalloc(&devPtr, sizeof(*devPtr) * N));
    gpuState.allCudaPointers.emplace_back(devPtr, adeptint::cudaDeleter);
  };

  SlotManager *slotManagers_dev = nullptr;
  gpuMalloc(slotManagers_dev, ParticleType::NumParticleTypes);

  // Create a stream to synchronize kernels of all particle types.
  COPCORE_CUDA_CHECK(cudaStreamCreate(&gpuState.stream));
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    ParticleType &particleType = gpuState.particles[i];

    // Share hepem state between threads
    COPCORE_CUDA_CHECK(cudaMalloc(&particleType.tracks, TracksSize));

    particleType.slotManager_host = SlotManager{static_cast<SlotManager::value_type>(fTrackCapacity),
                                                static_cast<SlotManager::value_type>(fTrackCapacity)};
    particleType.slotManager      = slotManagers_dev + i;
    COPCORE_CUDA_CHECK(
        cudaMemcpy(particleType.slotManager, &particleType.slotManager_host, sizeof(SlotManager), cudaMemcpyDefault));

    COPCORE_CUDA_CHECK(cudaMalloc(&particleType.queues.currentlyActive, QueueSize));
    COPCORE_CUDA_CHECK(cudaMalloc(&particleType.queues.nextActive, QueueSize));
    COPCORE_CUDA_CHECK(cudaMalloc(&particleType.queues.leakedTracksCurrent, QueueSize));
    COPCORE_CUDA_CHECK(cudaMalloc(&particleType.queues.leakedTracksNext, QueueSize));
    InitParticleQueues<<<1, 1>>>(particleType.queues, fTrackCapacity);

    COPCORE_CUDA_CHECK(cudaStreamCreate(&particleType.stream));
    COPCORE_CUDA_CHECK(cudaEventCreate(&particleType.event));
  }

  // init gamma interaction queues
  for (unsigned int i = 0; i < GammaInteractions::NInt; ++i) {
    const auto capacity     = fTrackCapacity / 3;
    const auto instanceSize = adept::MParrayT<GammaInteractions::Data>::SizeOfInstance(capacity);
    COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.gammaInteractions.queues[i], instanceSize));
    InitQueue<GammaInteractions::Data><<<1, 1>>>(gpuState.gammaInteractions.queues[i], capacity);
  }

  // initialize statistics
  COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.stats_dev, sizeof(Stats)));
  COPCORE_CUDA_CHECK(cudaMallocHost(&gpuState.stats, sizeof(Stats)));

  // init scoring on device
  gpuMalloc(fScoring_dev, fScoring.size());
  for (unsigned int i = 0; i < fNThread; ++i) {
    fScoring[i].InitializeOnGPU(fScoring_dev + i);
  }

  // initialize buffers for track transfer on host and device
  allocToDeviceTrackData(gpuState, gpuState.fNumToDevice);
  allocFromDeviceTrackData(gpuState, gpuState.fNumFromDevice);
  fBuffer = std::make_unique<adeptint::TrackBuffer>(gpuState.toDevice_host.get(), gpuState.fNumToDevice,
                                                    gpuState.toDevice_host.get() + gpuState.fNumToDevice,
                                                    gpuState.fNumToDevice, fNThread);

  const auto injectQueueSize = adept::MParrayT<QueueIndexPair>::SizeOfInstance(gpuState.fNumToDevice);
  adept::MParrayT<QueueIndexPair> *injectQueue;
  COPCORE_CUDA_CHECK(cudaMalloc(&injectQueue, injectQueueSize));
  gpuState.injectionQueue = {injectQueue, adeptint::cudaDeleter};
  InitQueue<QueueIndexPair><<<1, 1>>>(gpuState.injectionQueue.get(), gpuState.fNumToDevice);
}

void AdeptIntegration::FreeGPU()
{
  fGPUstate->runTransport = false;
  fGPUWorker.join();

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
  delete AdeptIntegration::fg4hepem_state;
  AdeptIntegration::fg4hepem_state = nullptr;
}

void AdeptIntegration::TransportLoop()
{
  using TrackData = adeptint::TrackData;
  // Capacity of the different containers aka the maximum number of particles.
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
  adeptint::unique_ptr_cudaEvent cudaEventCleanup{&cudaEvent, adeptint::cudaEventDeleter};
  adeptint::unique_ptr_cudaEvent cudaStatsEventCleanup{&cudaStatsEvent, adeptint::cudaEventDeleter};
  COPCORE_CUDA_CHECK(cudaStreamCreate(&transferStream));
  COPCORE_CUDA_CHECK(cudaStreamCreate(&statsStream));
  COPCORE_CUDA_CHECK(cudaStreamCreate(&interactionStream));
  std::unique_ptr<cudaStream_t, decltype(&adeptint::cudaStreamDeleter)> cudaStreamCleanup{&transferStream,
                                                                                          adeptint::cudaStreamDeleter};
  std::unique_ptr<cudaStream_t, decltype(&adeptint::cudaStreamDeleter)> cudaStatsStreamCleanup{
      &statsStream, adeptint::cudaStreamDeleter};
  std::unique_ptr<cudaStream_t, decltype(&adeptint::cudaStreamDeleter)> cudaInteractionStreamCleanup{
      &interactionStream, adeptint::cudaStreamDeleter};
  auto waitForOtherStream = [&cudaEvent](cudaStream_t waitingStream, cudaStream_t streamToWaitFor) {
    COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, streamToWaitFor));
    COPCORE_CUDA_CHECK(cudaStreamWaitEvent(waitingStream, cudaEvent));
  };

  enum class TransferState { Idle, ToDevice, Enqueued, CollectOnDevice, CopyToHost, BackOnHost };

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

  unsigned int *nFromDevice_host = nullptr;
  COPCORE_CUDA_CHECK(cudaMallocHost(&nFromDevice_host, sizeof(unsigned int)));
  adeptint::unique_ptr_cuda<unsigned int> nFromDevCleanup{nFromDevice_host, adeptint::cudaHostDeleter};

  SlotManager *const slotMgrArray = gpuState.particles[0].slotManager;
  while (gpuState.runTransport) {
    InitSlotManagers<<<80, 256, 0, gpuState.stream>>>(slotMgrArray, ParticleType::NumParticleTypes);
    COPCORE_CUDA_CHECK(cudaMemsetAsync(gpuState.stats_dev, 0, sizeof(Stats), gpuState.stream));

    int inFlight          = 0;
    int loopingNo         = 0;
    int previousElectrons = -1, previousPositrons = -1;
    TransferState transferState = TransferState::Idle;

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

    for (unsigned int iteration = 0; inFlight > 0 || transferState != TransferState::Idle ||
                                     std::any_of(fEventStates.begin(), fEventStates.end(), needTransport);
         ++iteration) {

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

      const auto prevNumElectrons = gpuState.stats->inFlight[ParticleType::Electron];
      const auto prevNumPositrons = gpuState.stats->inFlight[ParticleType::Positron];
      const auto prevNumGammas    = gpuState.stats->inFlight[ParticleType::Gamma];

      // *** ELECTRONS ***
      {
        const auto [threads, blocks] = computeThreadsAndBlocks(prevNumElectrons);
        TransportElectrons<AdeptScoring><<<blocks, threads, 0, electrons.stream>>>(
            electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive,
            electrons.queues.leakedTracksCurrent, fScoring_dev);

        COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, electrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, electrons.event, 0));
      }

      // *** POSITRONS ***
      {
        const auto [threads, blocks] = computeThreadsAndBlocks(prevNumPositrons);
        TransportPositrons<AdeptScoring><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive,
            positrons.queues.leakedTracksCurrent, fScoring_dev);

        COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, positrons.event, 0));
      }

      // *** GAMMAS ***
      {
        const auto [threads, blocks] = computeThreadsAndBlocks(prevNumGammas);
        TransportGammas<AdeptScoring><<<blocks, threads, 0, gammas.stream>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive,
            gammas.queues.leakedTracksCurrent, fScoring_dev, gpuState.gammaInteractions);

        COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(interactionStream, gammas.event, 0));

        constexpr unsigned int intThreads = 256;
        ApplyGammaInteractions<AdeptScoring, 0><<<20, intThreads, 0, gammas.stream>>>(
            gammas.tracks, secondaries, gammas.queues.nextActive, fScoring_dev, gpuState.gammaInteractions);
        ApplyGammaInteractions<AdeptScoring, 1><<<20, intThreads, 0, gammas.stream>>>(
            gammas.tracks, secondaries, gammas.queues.nextActive, fScoring_dev, gpuState.gammaInteractions);
        ApplyGammaInteractions<AdeptScoring, 2><<<40, intThreads, 0, interactionStream>>>(
            gammas.tracks, secondaries, gammas.queues.nextActive, fScoring_dev, gpuState.gammaInteractions);

        COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, gammas.event, 0));
        COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, interactionStream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, gammas.event, 0));
      }

      // *** Count detailed event statistics ***
      {
        ZeroEventCounters<<<1, 256, 0, statsStream>>>(gpuState.stats_dev);
        FillEventCounters<<<3 * ParticleType::NumParticleTypes, 128, 0, statsStream>>>(
            allParticleQueues, gpuState.stats_dev, tracksAndSlots);
        waitForOtherStream(gpuState.stream, statsStream);

        COPCORE_CUDA_CHECK(
            cudaMemcpyAsync(gpuState.stats, gpuState.stats_dev, sizeof(Stats), cudaMemcpyDeviceToHost, statsStream));
        COPCORE_CUDA_CHECK(cudaEventRecord(cudaStatsEvent, statsStream));
      }

      // *** Inject particles ***
      if (transferState == TransferState::Idle) {
        if (auto &toDevice = fBuffer->getActiveBuffer(); toDevice.nTrack > 0) {
          transferState = TransferState::ToDevice;

          std::vector<unsigned int> threadsInFlushState;
          for (unsigned int tid = 0; tid < fEventStates.size(); ++tid) {
            if (fEventStates[tid].load(std::memory_order_acquire) == EventState::G4Flush) {
              threadsInFlushState.push_back(tid);
            }
          }

          fBuffer->swapToDeviceBuffers();
          std::scoped_lock lock{toDevice.mutex};
          const auto nInject = std::min(toDevice.nTrack.load(), toDevice.maxTracks);

          if (fDebugLevel > 3) std::cout << "Injecting " << nInject << " to GPU\n";

          // copy buffer of tracks to device
          COPCORE_CUDA_CHECK(cudaMemcpyAsync(gpuState.toDevice_dev.get(), toDevice.tracks,
                                             nInject * sizeof(adeptint::TrackData), cudaMemcpyHostToDevice,
                                             transferStream));
          // Mark end of copy operation:
          COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, transferStream));

          // Inject AdePT tracks using the track buffer
          constexpr auto injectThreads = 128u;
          const auto injectBlocks      = (nInject + injectThreads - 1) / injectThreads;
          InjectTracks<<<injectBlocks, injectThreads, 0, transferStream>>>(gpuState.toDevice_dev.get(), nInject,
                                                                           secondaries, world_dev, fScoring_dev,
                                                                           gpuState.injectionQueue.get());
          std::set<unsigned int> threadIdsInjected;
          for (unsigned int i = 0; i < nInject; ++i) {
            threadIdsInjected.insert(toDevice.tracks[i].threadId);
          }
          for (auto threadInFlush : threadsInFlushState) {
            if (threadIdsInjected.count(threadInFlush) == 0) {
              fEventStates[threadInFlush] = EventState::InFlight;
            }
          }

          // Ensure that copy operation completed before releasing lock on to-device buffer
          toDevice.nTrack = 0;
          COPCORE_CUDA_CHECK(cudaEventSynchronize(cudaEvent));
        } else {
          transferState = TransferState::Enqueued;
          // Advance event state since nothing to inject
          for (auto &eventState : fEventStates) {
            EventState expected = EventState::G4Flush;
            eventState.compare_exchange_strong(expected, EventState::InFlight, std::memory_order_relaxed,
                                               std::memory_order_relaxed);
          }
        }
      } else if (transferState == TransferState::ToDevice && cudaStreamQuery(transferStream) == cudaSuccess) {
        transferState = TransferState::Enqueued;
        EnqueueTracks<<<1, 256, 0, gpuState.stream>>>(allParticleQueues, gpuState.injectionQueue.get());
      }

      // *** Collect particles ***
      if (transferState == TransferState::Enqueued) {
        transferState             = TransferState::CollectOnDevice;
        const AllLeaked allLeaked = {.leakedElectrons = {electrons.tracks, electrons.queues.leakedTracksCurrent,
                                                         electrons.queues.leakedTracksNext, electrons.slotManager},
                                     .leakedPositrons = {positrons.tracks, positrons.queues.leakedTracksCurrent,
                                                         positrons.queues.leakedTracksNext, positrons.slotManager},
                                     .leakedGammas    = {gammas.tracks, gammas.queues.leakedTracksCurrent,
                                                         gammas.queues.leakedTracksNext, gammas.slotManager}};
        electrons.queues.SwapLeakedQueue();
        positrons.queues.SwapLeakedQueue();
        gammas.queues.SwapLeakedQueue();

        // Ensure that transport that's writing to the old queues finishes before collecting leaked tracks
        for (auto event : {electrons.event, positrons.event, gammas.event}) {
          COPCORE_CUDA_CHECK(cudaStreamWaitEvent(transferStream, event));
        }

        // Populate the staging buffer and copy to host
        constexpr unsigned int block_size = 128;
        const unsigned int grid_size      = (gpuState.fNumFromDevice + block_size - 1) / block_size;
        FillFromDeviceBuffer<<<grid_size, block_size, 0, transferStream>>>(allLeaked, gpuState.fromDevice_dev.get(),
                                                                           gpuState.fNumFromDevice);
        COPCORE_CUDA_CHECK(cudaMemcpyFromSymbolAsync(nFromDevice_host, nFromDevice_dev, sizeof(unsigned int), 0,
                                                     cudaMemcpyDeviceToHost, transferStream));
        ClearQueue<<<1, 1, 0, transferStream>>>(electrons.queues.leakedTracksNext);
        ClearQueue<<<1, 1, 0, transferStream>>>(positrons.queues.leakedTracksNext);
        ClearQueue<<<1, 1, 0, transferStream>>>(gammas.queues.leakedTracksNext);
        // waitForOtherStream(gpuState.stream, transferStream);
      } else if (transferState == TransferState::CollectOnDevice && cudaStreamQuery(transferStream) == cudaSuccess) {
        transferState = TransferState::CopyToHost;
        if (*nFromDevice_host > 0) {
          COPCORE_CUDA_CHECK(cudaMemcpyAsync(gpuState.fromDevice_host.get(), gpuState.fromDevice_dev.get(),
                                             (*nFromDevice_host) * sizeof(TrackData), cudaMemcpyDeviceToHost,
                                             transferStream));
        } else {
          transferState = TransferState::BackOnHost;
        }

        // Freeing of slots has to run exclusively
        FreeSlots<<<ParticleType::NumParticleTypes, 256, 0, gpuState.stream>>>(tracksAndSlots);
        waitForOtherStream(transferStream, gpuState.stream);
      } else if (transferState == TransferState::CopyToHost && cudaStreamQuery(transferStream) == cudaSuccess) {
        transferState = TransferState::BackOnHost;
        std::scoped_lock lock{fBuffer->fromDeviceMutex};

        for (TrackData *trackIt = gpuState.fromDevice_host.get();
             trackIt < gpuState.fromDevice_host.get() + (*nFromDevice_host); ++trackIt) {
          assert(0 <= trackIt->threadId && trackIt->threadId <= fNThread);
          fBuffer->fromDeviceBuffers[trackIt->threadId].push_back(*trackIt);
        }
#ifndef NDEBUG
        for (const auto &trackBuffer : fBuffer->fromDeviceBuffers) {
          if (trackBuffer.empty()) continue;
          const auto eventId = trackBuffer.front().eventId;
          assert(std::all_of(trackBuffer.begin(), trackBuffer.end(),
                             [eventId](const TrackData &track) { return eventId == track.eventId; }));
        }
#endif
      }

      // *** Finish iteration ***
      FinishIteration<<<4, 32, 0, gpuState.stream>>>(allParticleQueues, gpuState.stats_dev, tracksAndSlots,
                                                     gpuState.gammaInteractions);
      COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, gpuState.stream));
      for (auto stream : {electrons.stream, positrons.stream, gammas.stream, statsStream}) {
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, cudaEvent));
      }

      // *** Count particles in flight ***
      inFlight               = 0;
      unsigned int numLeaked = 0;
      // Wait for arrival of device statistics from previous iteration
      COPCORE_CUDA_CHECK(cudaEventSynchronize(cudaStatsEvent));
      for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
        inFlight += gpuState.stats->inFlight[i];
        numLeaked += gpuState.stats->leakedTracks[i];
      }

      if (transferState == TransferState::BackOnHost) {
        transferState = TransferState::Idle;

        // Notify G4 workers if their events completed
        bool eventCompleted = false;
        for (unsigned short threadId = 0; threadId < fNThread; ++threadId) {
          if (gpuState.stats->occupancy[threadId] == 0) {
            EventState expected = EventState::InFlight;
            eventCompleted |= fEventStates[threadId].compare_exchange_strong(
                expected, EventState::DeviceFlushed, std::memory_order_relaxed, std::memory_order_relaxed);
          }
        }
        if (eventCompleted) {
          fBuffer->cv_fromDevice.notify_all();
        }
      }

      if (fDebugLevel >= 3) {
        std::cerr << inFlight << " in flight ";
        if (fDebugLevel >= 4) {
          std::cerr << "(" << gpuState.stats->inFlight[ParticleType::Electron] << " "
                    << gpuState.stats->inFlight[ParticleType::Positron] << " "
                    << gpuState.stats->inFlight[ParticleType::Gamma] << "),\tSlots:("
                    << gpuState.stats->usedSlots[ParticleType::Electron] << " "
                    << gpuState.stats->usedSlots[ParticleType::Positron] << " "
                    << gpuState.stats->usedSlots[ParticleType::Gamma] << ")";
        }
        std::cerr << ", " << numLeaked << " leaked.\n\tper event: ";
        for (unsigned int i = 0; i < fNThread; ++i) {
          std::cerr << i << ": " << gpuState.stats->occupancy[i]
                    << " (s=" << static_cast<unsigned short>(fEventStates[i].load(std::memory_order_acquire)) << ")\t";
        }
        std::cerr << std::endl;
      }

      // TODO: Write this per thread
      // Check if only charged particles are left that are looping.
      if (gpuState.stats->inFlight[ParticleType::Electron] == previousElectrons &&
          gpuState.stats->inFlight[ParticleType::Positron] == previousPositrons &&
          gpuState.stats->inFlight[ParticleType::Gamma] == 0) {
        loopingNo++;
      } else {
        previousElectrons = gpuState.stats->inFlight[ParticleType::Electron];
        previousPositrons = gpuState.stats->inFlight[ParticleType::Positron];
        loopingNo         = 0;
      }

#ifndef NDEBUG
      // *** Check slots ***
      if (transferState == TransferState::Idle || transferState >= TransferState::CopyToHost) {
        AssertConsistencyOfSlotManagers<<<120, 256, 0, gpuState.stream>>>(slotMgrArray, ParticleType::NumParticleTypes);
        COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
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

    // TODO: Add special treatment of looping tracks

    AllParticleQueues queues = {{electrons.queues, positrons.queues, gammas.queues}};
    ClearAllQueues<<<1, 1, 0, gpuState.stream>>>(queues);
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

    // TODO: This should be per event
    fScoring[0].fGlobalScoring.numKilled += inFlight;

    if (fDebugLevel > 2) std::cout << "End transport loop.\n";
  }
}
