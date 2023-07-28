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

// Kernel function to initialize tracks comming from a Geant4 buffer
__global__ void InjectTracks(adeptint::TrackData *trackinfo, int ntracks, Secondaries secondaries,
                             const vecgeom::VPlacedVolume *world, AdeptScoring *userScoring)
{
  constexpr double tolerance = 10. * vecgeom::kTolerance;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ntracks; i += blockDim.x * gridDim.x) {
    ParticleGenerator *generator = nullptr;
    const auto &trackInfo        = trackinfo[i];
    // These tracks come from Geant4, do not count them here
    switch (trackInfo.pdg) {
    case 11:
      generator = &secondaries.electrons;
      break;
    case -11:
      generator = &secondaries.positrons;
      break;
    case 22:
      generator = &secondaries.gammas;
    };
    assert(generator != nullptr && "Unsupported pdg type");

    Track &track = generator->NextTrack();
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

    track.navState.Clear();
    // We locate the pushed point because we run the risk that the
    // point is not located in the GPU region
    BVHNavigator::LocatePointIn(world, track.pos + tolerance * track.dir, track.navState, true);
    // The track must be on boundary at this point
    track.navState.SetBoundaryState(true);
    // nextState is initialized as needed.
    auto volume                         = track.navState.Top();
    int lvolID                          = volume->GetLogicalVolume()->id();
    adeptint::VolAuxData const &auxData = userScoring[trackInfo.threadId].GetAuxData_dev(lvolID);
    assert(auxData.fGPUregion);
  }
}

// Copy particles leaked from the GPU region into a compact buffer
__global__ void FillFromDeviceBuffer(int numLeaked, AllLeaked all, adeptint::TrackData *fromDevice)
{
  const auto numElectrons = all.leakedElectrons.fLeakedQueue->size();
  const auto numPositrons = all.leakedPositrons.fLeakedQueue->size();
  const auto numGammas    = all.leakedGammas.fLeakedQueue->size();
  assert(numLeaked == numElectrons + numPositrons + numGammas);

  for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < numLeaked; i += blockDim.x * gridDim.x) {
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
      atomicAdd(stats->occupancy + tracks[i].threadId, 1u);
    }
  }

  constexpr auto nSlotMgrs = sizeof(tracksAndSlots.slotManagers) / sizeof(tracksAndSlots.slotManagers[0]);
  for (unsigned int i = blockIdx.x; i < nSlotMgrs; i += gridDim.x) {
    tracksAndSlots.slotManagers[i]->FreeMarkedSlots();
  }

  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < ParticleType::NumParticleTypes; i += blockDim.x) {
      all.queues[i].currentlyActive->clear();
      stats->inFlight[i]     = all.queues[i].nextActive->size();
      stats->leakedTracks[i] = all.queues[i].leakedTracks->size();
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

bool AdeptIntegration::InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world)
{
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
  COPCORE_CUDA_CHECK(vecgeom::cxx::CudaDeviceSetStackLimit(8192));

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
                                                static_cast<SlotManager::value_type>(fTrackCapacity / 10)};
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
  COPCORE_CUDA_CHECK(cudaEventCreateWithFlags(&cudaEvent, cudaEventDisableTiming));
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

    int inFlight          = 0;
    int numLeaked         = 0;
    int loopingNo         = 0;
    int previousElectrons = -1, previousPositrons = -1;
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

    for (unsigned int iteration = 0; inFlight > 0 || numLeaked > 0 || (fBuffer->getActiveBuffer().nTrack > 0) ||
                                     std::any_of(fEventStates.begin(), fEventStates.end(), needTransport);
         ++iteration) {
      COPCORE_CUDA_CHECK(cudaMemsetAsync(gpuState.stats_dev, 0, sizeof(Stats), gpuState.stream));

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

      // *** Inject new particles ***
      bool injectionRequested = false;
      for (unsigned short slot = 0; slot < fNThread && !injectionRequested; ++slot) {
        auto const &injectBuffer = fBuffer->getActiveBuffer();
        if (fEventStates[slot].load(std::memory_order_acquire) == EventState::FlushRequested &&
            fBuffer->tracksLeftForSlot(slot)) {
          injectionRequested = true;
          break;
        }
      }
      if (auto &toDevice = fBuffer->getActiveBuffer();
          injectionRequested || toDevice.nTrack > static_cast<unsigned int>(fBufferThreshold)) {
        fBuffer->swapToDeviceBuffers();
        std::scoped_lock lock{toDevice.mutex};
        const auto nInject = std::min(toDevice.nTrack.load(), toDevice.maxTracks);
        if (fDebugLevel > 2) std::cout << "Injecting " << nInject << " to GPU\n";

        // copy buffer of tracks to device
        COPCORE_CUDA_CHECK(cudaMemcpyAsync(gpuState.toDevice_dev.get(), toDevice.tracks,
                                           nInject * sizeof(adeptint::TrackData), cudaMemcpyHostToDevice,
                                           gpuState.stream));
        COPCORE_CUDA_CHECK(cudaEventRecord(cudaEvent, gpuState.stream));

        // Inject AdePT tracks using the track buffer
        const auto injectThreads = std::min(nInject, 32u);
        const auto injectBlocks = (nInject + injectThreads - 1) / injectThreads;
        InjectTracks<<<injectBlocks, injectThreads, 0, gpuState.stream>>>(gpuState.toDevice_dev.get(), nInject,
                                                                          secondaries, world_dev, fScoring_dev);

        std::vector<unsigned short> eventsInjected;
        eventsInjected.reserve(nInject);
        for (unsigned int i = 0; i < nInject; ++i) {
          eventsInjected.push_back(toDevice.tracks[i].threadId);
        }
        std::sort(eventsInjected.begin(), eventsInjected.end());
        const auto newEnd = std::unique(eventsInjected.begin(), eventsInjected.end());
        for (auto it = eventsInjected.begin(); it < newEnd; ++it) {
          EventState expected = EventState::NewTracksForDevice;
          fEventStates[*it].compare_exchange_strong(expected, EventState::Transporting, std::memory_order_relaxed,
                                                    std::memory_order_relaxed);
        }

        toDevice.nTrack = 0;
        // Can only release the buffer once the copy operation completed:
        COPCORE_CUDA_CHECK(cudaEventSynchronize(cudaEvent));
      }

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

      // The events ensure synchronization before finishing this iteration and
      // copying the Stats back to the host.
      AllParticleQueues queues = {{electrons.queues, positrons.queues, gammas.queues}};
      TracksAndSlots tracks    = {{electrons.tracks, positrons.tracks, gammas.tracks},
                                  {electrons.slotManager, positrons.slotManager, gammas.slotManager}};
      // TODO: Better grid size?
      FinishIteration<<<80, 128, 0, gpuState.stream>>>(queues, gpuState.stats_dev, tracks);
      COPCORE_CUDA_CHECK(
          cudaMemcpyAsync(gpuState.stats, gpuState.stats_dev, sizeof(Stats), cudaMemcpyDeviceToHost, gpuState.stream));

#if false and not defined NDEBUG
    for (int i = 0; i < ParticleType::NumParticleTypes; ++i) {
      ParticleType &part = gpuState.particles[i];
      COPCORE_CUDA_CHECK(cudaMemcpyAsync(&part.slotManager_host, part.slotManager, sizeof(SlotManager),
                                         cudaMemcpyDefault, gpuState.stream));
    }
    std::cout << "SlotManager: ";
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));
    for (int i = 0; i < ParticleType::NumParticleTypes; ++i) {
      ParticleType &part = gpuState.particles[i];
      std::cout << part.slotManager_host.fSlotCounter << "\t";
    }
    std::cout << " slots used.\n";
#endif

      // Finally synchronize all kernels.
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

      // Count the number of particles in flight.
      inFlight  = 0;
      numLeaked = 0;
      for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
        inFlight += gpuState.stats->inFlight[i];
        numLeaked += gpuState.stats->leakedTracks[i];
      }

      if (fDebugLevel > 3) {
        std::cerr << inFlight << " in flight ";
        if (fDebugLevel > 4) {
          std::cerr << "(" << gpuState.stats->inFlight[ParticleType::Electron] << " "
                    << gpuState.stats->inFlight[ParticleType::Positron] << " "
                    << gpuState.stats->inFlight[ParticleType::Gamma] << ") ";
        }
        std::cerr << ", " << numLeaked << " leaked.\nEvent occupancies:\n\t";
        for (unsigned int i = 0; i < fNThread; ++i) {
          std::cerr << i << ": " << gpuState.stats->occupancy[i] << "\t";
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
      if (iteration % 5 == 4 && numLeaked > 0) {
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

      if (numLeaked == 0) {
        bool canNotify = false;
        for (unsigned short threadId = 0; threadId < fNThread; ++threadId) {
          if (gpuState.stats->occupancy[threadId] == 0 && !fBuffer->tracksLeftForSlot(threadId)) {
            EventState expected = EventState::FlushRequested;
            canNotify |= fEventStates[threadId].compare_exchange_strong(
                expected, EventState::DeviceFlushed, std::memory_order_release, std::memory_order_relaxed);
          }
        }
        if (canNotify) {
          // Notify G4 workers of event completion:
          fBuffer->cv_fromDevice.notify_all();
        }
      }

#warning Figure out how to treat looping tracks
    }

    AllParticleQueues queues = {{electrons.queues, positrons.queues, gammas.queues}};
    ClearAllQueues<<<1, 1, 0, gpuState.stream>>>(queues);
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

#warning FIXME
    fScoring[0].fGlobalScoring.numKilled += inFlight;

    if (fDebugLevel > 2) std::cout << "End transport loop.\n";
  }
}
