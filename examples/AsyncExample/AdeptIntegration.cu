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
  adept::MParray::MakeInstanceAt(Capacity, queues.leakedTracks);
}

// Kernel to initialize the set of queues per particle type.
__global__ void InitInjectionQueue(adept::MParrayT<QueueIndexPair> *queue, size_t Capacity)
{
  adept::MParrayT<QueueIndexPair>::MakeInstanceAt(Capacity, queue);
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
#if false and not defined NDEBUG
    printf("\t%d: Obtained slot %d for %d. (%d %d %d)\n", threadIdx.x, slot, trackInfo.pdg,
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

// Copy particles leaked from the GPU region into a compact buffer
__global__ void FillFromDeviceBuffer(int numLeaked, AllLeaked all, adeptint::TrackData *fromDevice)
{
  const auto numElectrons = all.leakedElectrons.fLeakedQueue->size();
  const auto numPositrons = all.leakedPositrons.fLeakedQueue->size();
  const auto numGammas    = all.leakedGammas.fLeakedQueue->size();
  const auto total        = numElectrons + numPositrons + numGammas;
  assert(numLeaked < 0 || numLeaked == numElectrons + numPositrons + numGammas);

  for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < total; i += blockDim.x * gridDim.x) {
    Track const *trackCollection = nullptr;
    unsigned int trackSlot       = 0;
    SlotManager *slotManager     = nullptr;
    int pdg                      = 0;

    if (i < numGammas) {
      trackCollection = all.leakedGammas.fTracks;
      trackSlot       = (*all.leakedGammas.fLeakedQueue)[i];
      slotManager     = all.leakedGammas.fSlotManager;
      pdg             = 22;
    } else if (i < numGammas + numElectrons) {
      trackCollection = all.leakedElectrons.fTracks;
      trackSlot       = (*all.leakedElectrons.fLeakedQueue)[i - numGammas];
      slotManager     = all.leakedElectrons.fSlotManager;
      pdg             = 11;
    } else {
      trackCollection = all.leakedPositrons.fTracks;
      trackSlot       = (*all.leakedPositrons.fLeakedQueue)[i - numGammas - numElectrons];
      slotManager     = all.leakedPositrons.fSlotManager;
      pdg             = -11;
    }

    Track const *const track   = trackCollection + trackSlot;
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
    slotManager->MarkSlotForFreeing(trackSlot);
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
// TODO: Use shared mem counters ?
// TODO: Distribute work better among blocks
__global__ void FinishIteration(AllParticleQueues all, Stats *stats, TracksAndSlots tracksAndSlots)
{
  for (unsigned int particleType = 0; particleType < ParticleType::NumParticleTypes; ++particleType) {
    const auto &queue   = *all.queues[particleType].nextActive;
    const auto end      = queue.size();
    Track const *tracks = tracksAndSlots.tracks[particleType];
    for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < end; i += blockDim.x * gridDim.x) {
      const auto slot = queue[i];
      atomicAdd(stats->occupancy + tracks[slot].threadId, 1u);
    }
  }

  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < ParticleType::NumParticleTypes; i += blockDim.x) {
      all.queues[i].currentlyActive->clear();
      stats->inFlight[i]     = all.queues[i].nextActive->size();
      stats->leakedTracks[i] = all.queues[i].leakedTracks->size();
      stats->usedSlots[i]    = tracksAndSlots.slotManagers[i]->OccupiedSlots();
    }
  }
}

__global__ void ClearQueue(adept::MParray *queue)
{
  queue->clear();
}

__global__ void ClearLeakedQueues(AllLeaked all)
{
  if (threadIdx.x == 0)
    all.leakedElectrons.fLeakedQueue->clear();
  else if (threadIdx.x == 1)
    all.leakedPositrons.fLeakedQueue->clear();
  else if (threadIdx.x == 2)
    all.leakedGammas.fLeakedQueue->clear();
}

__global__ void ClearAllQueues(AllParticleQueues all)
{
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    all.queues[i].currentlyActive->clear();
    all.queues[i].nextActive->clear();
    all.queues[i].leakedTracks->clear();
  }
}

__global__ void InitSlotManagers(SlotManager *mgr, std::size_t N)
{
  for (int i = 0; i < N; ++i) {
    mgr[i].Clear();
  }
}

__global__ void AssertConsistencyOfSlotManagers(SlotManager *mgrs, std::size_t N)
{
  for (int i = 0; i < N; ++i) {
    SlotManager &mgr = mgrs[i];

    for (unsigned int j = blockIdx.x; j < mgr.fFreeCounter; j += gridDim.x) {
      const auto slotToSearch = mgr.fToFreeList[j];
      for (unsigned int k = threadIdx.x; k < j; k += blockDim.x) {
        if (slotToSearch == mgr.fToFreeList[k]) {
          printf("Error: Manager %d: Slot %d freed both at %d and at %d\n", i, slotToSearch, k, j);
          assert(false);
        }
      }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0 && mgr.fSlotCounter < mgr.fFreeCounter) {
      printf("Error %s:%d: Trying to free %d slots in manager %d whereas only %d allocated\n", __FILE__, __LINE__,
             mgr.fFreeCounter, i, mgr.fSlotCounter);
      for (unsigned int i = 0; i < mgr.fFreeCounter; ++i) {
        printf("%d ", mgr.fToFreeList[i]);
      }
      printf("\n");
      assert(false);
    }
  }
}

bool AdeptIntegration::InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world)
{
  COPCORE_CUDA_CHECK(vecgeom::cxx::CudaDeviceSetStackLimit(8192 * 2));
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
    COPCORE_CUDA_CHECK(cudaMalloc(&particleType.queues.leakedTracks, QueueSize));
    InitParticleQueues<<<1, 1>>>(particleType.queues, fTrackCapacity);

    COPCORE_CUDA_CHECK(cudaStreamCreate(&particleType.stream));
    COPCORE_CUDA_CHECK(cudaEventCreate(&particleType.event));
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
  InitInjectionQueue<<<1, 1>>>(gpuState.injectionQueue.get(), gpuState.fNumToDevice);
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
    COPCORE_CUDA_CHECK(cudaFree(gpuState.particles[i].queues.leakedTracks));

    COPCORE_CUDA_CHECK(cudaStreamDestroy(gpuState.particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventDestroy(gpuState.particles[i].event));
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

  cudaEvent_t cudaEvent;
  cudaStream_t injectParticlesStream;
  COPCORE_CUDA_CHECK(cudaEventCreateWithFlags(&cudaEvent, cudaEventDisableTiming));
  adeptint::unique_ptr_cudaEvent cudaEventCleanup{&cudaEvent, adeptint::cudaEventDeleter};
  COPCORE_CUDA_CHECK(cudaStreamCreate(&injectParticlesStream));
  std::unique_ptr<cudaStream_t, decltype(&adeptint::cudaStreamDeleter)> cudaStreamCleanup{&injectParticlesStream,
                                                                                          adeptint::cudaStreamDeleter};
  bool particleInjectionRunning = false;

  auto computeThreadsAndBlocks = [](unsigned int nParticles) -> std::pair<unsigned int, unsigned int> {
    constexpr int TransportThreads             = 256;
    constexpr int LowOccupancyTransportThreads = 32;

    int transportBlocks = (nParticles + TransportThreads - 1) / TransportThreads;
    if (transportBlocks < 10) {
      transportBlocks = (nParticles + LowOccupancyTransportThreads - 1) / LowOccupancyTransportThreads;
      return {LowOccupancyTransportThreads, transportBlocks};
    }
    return {TransportThreads, transportBlocks};
  };

  SlotManager *const slotMgrArray = gpuState.particles[0].slotManager;
  while (gpuState.runTransport) {
    InitSlotManagers<<<80, 256, 0, gpuState.stream>>>(slotMgrArray, ParticleType::NumParticleTypes);
    COPCORE_CUDA_CHECK(cudaMemsetAsync(gpuState.stats_dev, 0, sizeof(Stats), gpuState.stream));

    int inFlight          = 0;
    int numLeaked         = 0;
    int loopingNo         = 0;
    int previousElectrons = -1, previousPositrons = -1;
    unsigned int lastSlotFree = 0;
    AllLeaked leakedTracks = {
        .leakedElectrons = {electrons.tracks, electrons.queues.leakedTracks, electrons.slotManager},
        .leakedPositrons = {positrons.tracks, positrons.queues.leakedTracks, positrons.slotManager},
        .leakedGammas    = {gammas.tracks, gammas.queues.leakedTracks, gammas.slotManager}};

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

    for (unsigned int iteration = 0;
         inFlight > 0 || std::any_of(fEventStates.begin(), fEventStates.end(), needTransport); ++iteration) {

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

      // *** ELECTRONS ***
      const auto numElectrons = gpuState.stats->inFlight[ParticleType::Electron];
      if (numElectrons > 0) {
        const auto [threads, blocks] = computeThreadsAndBlocks(numElectrons);
        TransportElectrons<AdeptScoring><<<blocks, threads, 0, electrons.stream>>>(
            electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive,
            electrons.queues.leakedTracks, fScoring_dev);

        COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, electrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, electrons.event, 0));
      }

      // *** POSITRONS ***
      const auto numPositrons = gpuState.stats->inFlight[ParticleType::Positron];
      if (numPositrons > 0) {
        const auto [threads, blocks] = computeThreadsAndBlocks(numPositrons);
        TransportPositrons<AdeptScoring><<<blocks, threads, 0, positrons.stream>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive,
            positrons.queues.leakedTracks, fScoring_dev);

        COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, positrons.event, 0));
      }

      // *** GAMMAS ***
      const auto numGammas = gpuState.stats->inFlight[ParticleType::Gamma];
      if (numGammas > 0) {
        const auto [threads, blocks] = computeThreadsAndBlocks(numGammas);
        TransportGammas<AdeptScoring>
            <<<blocks, threads, 0, gammas.stream>>>(gammas.tracks, gammas.queues.currentlyActive, secondaries,
                                                    gammas.queues.nextActive, gammas.queues.leakedTracks, fScoring_dev);

        COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, gammas.event, 0));
      }

      // *** Inject new particles ***
      if (auto &toDevice = fBuffer->getActiveBuffer(); !particleInjectionRunning && toDevice.nTrack > 0) {
        particleInjectionRunning = true;
        fBuffer->swapToDeviceBuffers();
        std::scoped_lock lock{toDevice.mutex};
        const auto nInject = std::min(toDevice.nTrack.load(), toDevice.maxTracks);
        toDevice.nTrack    = 0;
        if (fDebugLevel > 3) std::cout << "Injecting " << nInject << " to GPU\n";

        // copy buffer of tracks to device
        COPCORE_CUDA_CHECK(cudaMemcpyAsync(gpuState.toDevice_dev.get(), toDevice.tracks,
                                           nInject * sizeof(adeptint::TrackData), cudaMemcpyHostToDevice,
                                           injectParticlesStream));
        // Mark end of copy operation:
        COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, injectParticlesStream));

        // Inject AdePT tracks using the track buffer
        constexpr auto injectThreads = 128u;
        const auto injectBlocks      = (nInject + injectThreads - 1) / injectThreads;
        InjectTracks<<<injectBlocks, injectThreads, 0, injectParticlesStream>>>(
            gpuState.toDevice_dev.get(), nInject, secondaries, world_dev, fScoring_dev, gpuState.injectionQueue.get());

        // Update event state for each thread that's injecting
        {
          std::vector<unsigned short> threadIds;
          threadIds.reserve(nInject);
          for (unsigned int i = 0; i < nInject; ++i) {
            threadIds.push_back(toDevice.tracks[i].threadId);
          }
          std::sort(threadIds.begin(), threadIds.end());
          const auto newEnd = std::unique(threadIds.begin(), threadIds.end());
          for (auto it = threadIds.begin(); it < newEnd; ++it) {
            fEventStates[*it].store(EventState::InjectionRunning, std::memory_order_release);
          }
        }

        // Ensure that copy operation completed before releasing lock on to-device buffer
        COPCORE_CUDA_CHECK(cudaEventSynchronize(cudaEvent));
      } else if (particleInjectionRunning && cudaStreamQuery(injectParticlesStream) == cudaSuccess) {
        EnqueueTracks<<<1, 256, 0, injectParticlesStream>>>(allParticleQueues, gpuState.injectionQueue.get());
        COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, injectParticlesStream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, cudaEvent));

        for (auto &state : fEventStates) {
          EventState expected = EventState::InjectionRunning;
          state.compare_exchange_strong(expected, EventState::TracksInjected, std::memory_order_release,
                                        std::memory_order_relaxed);
        }
        particleInjectionRunning = false;
      }

      // *** Finish iteration ***
      // The events ensure synchronization before finishing this iteration and
      // copying the Stats back to the host.
      // TODO: Better grid size?
      FinishIteration<<<80, 128, 0, gpuState.stream>>>(allParticleQueues, gpuState.stats_dev, tracksAndSlots);
      COPCORE_CUDA_CHECK(
          cudaMemcpyAsync(gpuState.stats, gpuState.stats_dev, sizeof(Stats), cudaMemcpyDeviceToHost, gpuState.stream));
      // Mark readiness of GPU stats
      COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, gpuState.stream));
      COPCORE_CUDA_CHECK(cudaMemsetAsync(gpuState.stats_dev, 0, sizeof(Stats), gpuState.stream));

#ifndef NDEBUG
      if (!particleInjectionRunning) {
        AssertConsistencyOfSlotManagers<<<120, 256, 0, gpuState.stream>>>(slotMgrArray, ParticleType::NumParticleTypes);
      }
#endif

      if (!particleInjectionRunning && iteration - lastSlotFree > 10) {
        FreeSlots<<<ParticleType::NumParticleTypes, 256, 0, gpuState.stream>>>(tracksAndSlots);
        COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, gpuState.stream));
        // Freeing of slots cannot overlap with allocation
        // Use electrons event since default event waits for end of copy
        for (auto stream : {injectParticlesStream, electrons.stream, positrons.stream, gammas.stream}) {
          COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, electrons.event));
        }
        lastSlotFree = iteration;
      }

#if false and not defined NDEBUG
      for (int i = 0; i < ParticleType::NumParticleTypes; ++i) {
        ParticleType &part = gpuState.particles[i];
        COPCORE_CUDA_CHECK(cudaMemcpyAsync(&part.slotManager_host, part.slotManager, sizeof(SlotManager),
                                           cudaMemcpyDefault, gpuState.stream));
      }
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));
      {
        unsigned int slotsUsed[3];
        unsigned int slotsMax[3];
        for (int i = 0; i < ParticleType::NumParticleTypes; ++i) {
          ParticleType &part = gpuState.particles[i];
          slotsUsed[i]       = part.slotManager_host.fSlotCounter - part.slotManager_host.fFreeCounter;
          slotsMax[i]        = part.slotManager_host.fSlotCounterMax;
        }
        std::cout << "SlotManager: (" << slotsUsed[0] << ", " << slotsUsed[1] << ", " << slotsUsed[2]
                  << ") slots used.\t max=(" << slotsMax[0] << ", " << slotsMax[1] << ", " << slotsMax[2] << ")\n";
      }
#endif

      // Wait for GPU stats:
      COPCORE_CUDA_CHECK(cudaEventSynchronize(cudaEvent));

      // Count the number of particles in flight.
      inFlight  = 0;
      numLeaked = 0;
      for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
        inFlight += gpuState.stats->inFlight[i];
        numLeaked += gpuState.stats->leakedTracks[i];
      }

      if (fDebugLevel >= 3 && iteration % 10000 == 0) {
        std::cerr << inFlight << " in flight ";
        if (fDebugLevel > 4) {
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

      // Transfer leaked tracks back to per-worker queues
      if (numLeaked > 0 && iteration % 10 == 0) {
        // Ensure proper size of the buffer
        if (static_cast<unsigned int>(numLeaked) > gpuState.fNumFromDevice) {
          unsigned int newSize = gpuState.fNumFromDevice;
          while (newSize < static_cast<unsigned int>(numLeaked))
            newSize *= 2;

          std::cerr << __FILE__ << ':' << __LINE__ << ": Increasing fromDevice buffer to " << newSize
                    << " to accommodate " << numLeaked << "\n";
          allocFromDeviceTrackData(gpuState, newSize);
        }

        // Populate the staging buffer, copy to host, and clear the queues of leaked tracks
        constexpr unsigned int block_size = 256;
        unsigned int grid_size            = (numLeaked + block_size - 1) / block_size;

        // TODO: Try to make this async to not impede transport
        FillFromDeviceBuffer<<<grid_size, block_size, 0, gpuState.stream>>>(numLeaked, leakedTracks,
                                                                            gpuState.fromDevice_dev.get());
        COPCORE_CUDA_CHECK(cudaMemcpyAsync(gpuState.fromDevice_host.get(), gpuState.fromDevice_dev.get(),
                                           numLeaked * sizeof(TrackData), cudaMemcpyDeviceToHost, gpuState.stream));
        COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, gpuState.stream));
        ClearLeakedQueues<<<1, 3, 0, gpuState.stream>>>(leakedTracks);

        {
          std::scoped_lock lock{fBuffer->fromDeviceMutex};
          COPCORE_CUDA_CHECK(cudaEventSynchronize(cudaEvent));
          for (TrackData *trackIt = gpuState.fromDevice_host.get();
               trackIt < gpuState.fromDevice_host.get() + numLeaked; ++trackIt) {
            fBuffer->fromDeviceBuffers[trackIt->threadId].push_back(*trackIt);
          }
        }
        numLeaked = 0;
      }

      if (numLeaked == 0 && !particleInjectionRunning) {
        bool eventCompleted = false;
        for (unsigned short threadId = 0; threadId < fNThread; ++threadId) {
          if (gpuState.stats->occupancy[threadId] == 0) {
            EventState expected = EventState::TracksInjected;
            eventCompleted |= fEventStates[threadId].compare_exchange_strong(
                expected, EventState::DeviceFlushed, std::memory_order_relaxed, std::memory_order_relaxed);
          }
        }
        if (eventCompleted) {
          fBuffer->cv_fromDevice.notify_all();
        }
      }
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
