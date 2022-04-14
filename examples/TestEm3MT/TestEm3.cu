// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "TestEm3.h"
#include "TestEm3.cuh"

#include <AdePT/Atomic.h>
#include <AdePT/BVHNavigator.h>
#include <AdePT/MParray.h>

#include <CopCore/Global.h>
#include <CopCore/PhysicalConstants.h>
#include <CopCore/Ranluxpp.h>

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Stopwatch.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

#include <G4HepEmData.hh>
#include <G4HepEmElectronInit.hh>
#include <G4HepEmGammaInit.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmMaterialInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmParametersInit.hh>

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <thread>
#include <vector>

__constant__ __device__ struct G4HepEmParameters g4HepEmPars;
__constant__ __device__ struct G4HepEmData g4HepEmData;

__constant__ __device__ int *MCIndex = nullptr;

__constant__ __device__ int Zero = 0;

struct G4HepEmState {
  G4HepEmData data;
  G4HepEmParameters parameters;
};

static G4HepEmState *InitG4HepEm()
{
  G4HepEmState *state = new G4HepEmState;
  InitG4HepEmData(&state->data);
  InitHepEmParameters(&state->parameters);

  InitMaterialAndCoupleData(&state->data, &state->parameters);

  InitElectronData(&state->data, &state->parameters, true);
  InitElectronData(&state->data, &state->parameters, false);
  InitGammaData(&state->data, &state->parameters);

  G4HepEmMatCutData *cutData = state->data.fTheMatCutData;
  std::cout << "fNumG4MatCuts = " << cutData->fNumG4MatCuts << ", fNumMatCutData = " << cutData->fNumMatCutData
            << std::endl;

  // Copy to GPU.
  CopyG4HepEmDataToGPU(&state->data);
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmPars, &state->parameters, sizeof(G4HepEmParameters)));

  // Create G4HepEmData with the device pointers.
  G4HepEmData dataOnDevice;
  dataOnDevice.fTheMatCutData   = state->data.fTheMatCutData_gpu;
  dataOnDevice.fTheMaterialData = state->data.fTheMaterialData_gpu;
  dataOnDevice.fTheElementData  = state->data.fTheElementData_gpu;
  dataOnDevice.fTheElectronData = state->data.fTheElectronData_gpu;
  dataOnDevice.fThePositronData = state->data.fThePositronData_gpu;
  dataOnDevice.fTheSBTableData  = state->data.fTheSBTableData_gpu;
  dataOnDevice.fTheGammaData    = state->data.fTheGammaData_gpu;
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

static void FreeG4HepEm(G4HepEmState *state)
{
  FreeG4HepEmData(&state->data);
  delete state;
}

// A bundle of queues per particle type:
//  * Two for active particles, one for the current iteration and the second for the next.
struct ParticleQueues {
  adept::MParray *currentlyActive;
  adept::MParray *nextActive;

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

// Kernel to initialize the set of queues per particle type.
__global__ void InitParticleQueues(ParticleQueues queues, size_t Capacity)
{
  adept::MParray::MakeInstanceAt(Capacity, queues.currentlyActive);
  adept::MParray::MakeInstanceAt(Capacity, queues.nextActive);
}

// Kernel function to initialize a set of primary particles.
__global__ void InitPrimaries(ParticleGenerator generator, int startEvent, int numEvents, double energy, double startX,
                              const vecgeom::VPlacedVolume *world, GlobalScoring *globalScoring)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEvents; i += blockDim.x * gridDim.x) {
    Track &track = generator.NextTrack();

    track.rngState.SetSeed(startEvent + i);
    track.energy       = energy;
    track.numIALeft[0] = -1.0;
    track.numIALeft[1] = -1.0;
    track.numIALeft[2] = -1.0;

    track.initialRange       = -1.0;
    track.dynamicRangeFactor = -1.0;
    track.tlimitMin          = -1.0;

    track.pos = {vecgeom::Precision(startX), 0, 0};
    track.dir = {1.0, 0, 0};
    track.navState.Clear();
    BVHNavigator::LocatePointIn(world, track.pos, track.navState, true);

    atomicAdd(&globalScoring->numElectrons, 1);
  }
}

// A data structure to transfer statistics after each iteration.
struct Stats {
  int inFlight[ParticleType::NumParticleTypes];
};

// Finish iteration: clear queues and fill statistics.
__global__ void FinishIteration(AllParticleQueues all, Stats *stats)
{
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    all.queues[i].currentlyActive->clear();
    stats->inFlight[i] = all.queues[i].nextActive->size();
  }
}

__global__ void ClearQueue(adept::MParray *queue)
{
  queue->clear();
}

struct ThreadData {
  ParticleType particles[ParticleType::NumParticleTypes];
  cudaStream_t stream;
  Stats *stats;
  Stats *stats_dev;

  const vecgeom::cuda::VPlacedVolume *world;
  const int *MCIndex;
  const SlotManager *slotManagerInit;

  ScoringPerVolume *scoringPerVolume;
  GlobalScoring *globalScoring;

  int id;
  int threads;
  int numParticles;
  int batch;
  double energy;
  double startX;

  void Allocate(size_t capacity)
  {
    // Allocate structures to manage tracks of an implicit type:
    //  * memory to hold the actual Track elements,
    //  * objects to manage slots inside the memory,
    //  * queues of slots to remember active particle and those needing relocation,
    //  * a stream and an event for synchronization of kernels.
    const size_t TracksSize = sizeof(Track) * capacity;
    const size_t QueueSize  = adept::MParray::SizeOfInstance(capacity);

    for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
      COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].tracks, TracksSize));

      COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].slotManager, sizeof(SlotManager)));

      COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].queues.currentlyActive, QueueSize));
      COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].queues.nextActive, QueueSize));
      InitParticleQueues<<<1, 1>>>(particles[i].queues, capacity);

      COPCORE_CUDA_CHECK(cudaStreamCreate(&particles[i].stream));
      COPCORE_CUDA_CHECK(cudaEventCreate(&particles[i].event));
    }
    COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

    // Create a stream to synchronize kernels of all particle types.
    COPCORE_CUDA_CHECK(cudaStreamCreate(&stream));

    COPCORE_CUDA_CHECK(cudaMalloc(&stats_dev, sizeof(Stats)));
    COPCORE_CUDA_CHECK(cudaMallocHost(&stats, sizeof(Stats)));
  }

  void Free()
  {
    COPCORE_CUDA_CHECK(cudaFree(stats_dev));
    COPCORE_CUDA_CHECK(cudaFreeHost(stats));

    COPCORE_CUDA_CHECK(cudaStreamDestroy(stream));

    for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
      COPCORE_CUDA_CHECK(cudaFree(particles[i].tracks));
      COPCORE_CUDA_CHECK(cudaFree(particles[i].slotManager));

      COPCORE_CUDA_CHECK(cudaFree(particles[i].queues.currentlyActive));
      COPCORE_CUDA_CHECK(cudaFree(particles[i].queues.nextActive));

      COPCORE_CUDA_CHECK(cudaStreamDestroy(particles[i].stream));
      COPCORE_CUDA_CHECK(cudaEventDestroy(particles[i].event));
    }
  }
};

static void Worker(ThreadData *data)
{
  ParticleType *particles = data->particles;
  ParticleType &electrons = particles[ParticleType::Electron];
  ParticleType &positrons = particles[ParticleType::Positron];
  ParticleType &gammas    = particles[ParticleType::Gamma];

  Stats *stats         = data->stats;
  Stats *stats_dev     = data->stats_dev;
  cudaStream_t &stream = data->stream;

  ScoringPerVolume *scoringPerVolume = data->scoringPerVolume;
  GlobalScoring *globalScoring       = data->globalScoring;

  // Calculate this thread's chunk.
  int perThread  = data->numParticles / data->threads;
  int remainder  = data->numParticles % data->threads;
  int startEvent = 1 + data->id * perThread;
  if (data->id < remainder) {
    perThread++;
    startEvent += data->id;
  } else {
    startEvent += remainder;
  }
  int endEvent = startEvent + perThread;

  for (; startEvent < endEvent; startEvent += data->batch) {
    int left  = endEvent - startEvent;
    int chunk = std::min(left, data->batch);

    for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
      COPCORE_CUDA_CHECK(cudaMemcpyAsync(particles[i].slotManager, data->slotManagerInit, sizeof(SlotManager),
                                         cudaMemcpyDeviceToDevice, stream));
    }

    // Initialize primary particles.
    constexpr int InitThreads = 32;
    int initBlocks            = (chunk + InitThreads - 1) / InitThreads;
    ParticleGenerator electronGenerator(electrons.tracks, electrons.slotManager, electrons.queues.currentlyActive);
    InitPrimaries<<<initBlocks, InitThreads, 0, stream>>>(electronGenerator, startEvent, chunk, data->energy,
                                                          data->startX, data->world, globalScoring);
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));

    stats->inFlight[ParticleType::Electron] = chunk;
    stats->inFlight[ParticleType::Positron] = 0;
    stats->inFlight[ParticleType::Gamma]    = 0;

    constexpr int MaxBlocks        = 1024;
    constexpr int TransportThreads = 32;
    int transportBlocks;

    int inFlight;
    int loopingNo         = 0;
    int previousElectrons = -1, previousPositrons = -1;

    do {
      Secondaries secondaries = {
          .electrons = {electrons.tracks, electrons.slotManager, electrons.queues.nextActive},
          .positrons = {positrons.tracks, positrons.slotManager, positrons.queues.nextActive},
          .gammas    = {gammas.tracks, gammas.slotManager, gammas.queues.nextActive},
      };

      // *** ELECTRONS ***
      int numElectrons = stats->inFlight[ParticleType::Electron];
      if (numElectrons > 0) {
        transportBlocks = (numElectrons + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        TransportElectrons<<<transportBlocks, TransportThreads, 0, electrons.stream>>>(
            electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive, globalScoring,
            scoringPerVolume);

        COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, electrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, electrons.event, 0));
      }

      // *** POSITRONS ***
      int numPositrons = stats->inFlight[ParticleType::Positron];
      if (numPositrons > 0) {
        transportBlocks = (numPositrons + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        TransportPositrons<<<transportBlocks, TransportThreads, 0, positrons.stream>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive, globalScoring,
            scoringPerVolume);

        COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, positrons.event, 0));
      }

      // *** GAMMAS ***
      int numGammas = stats->inFlight[ParticleType::Gamma];
      if (numGammas > 0) {
        transportBlocks = (numGammas + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        TransportGammas<<<transportBlocks, TransportThreads, 0, gammas.stream>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive, globalScoring,
            scoringPerVolume);

        COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, gammas.event, 0));
      }

      // *** END OF TRANSPORT ***

      // The events ensure synchronization before finishing this iteration and
      // copying the Stats back to the host.
      AllParticleQueues queues = {{electrons.queues, positrons.queues, gammas.queues}};
      FinishIteration<<<1, 1, 0, stream>>>(queues, stats_dev);
      COPCORE_CUDA_CHECK(cudaMemcpyAsync(stats, stats_dev, sizeof(Stats), cudaMemcpyDeviceToHost, stream));

      // Finally synchronize all kernels.
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));

      // Count the number of particles in flight.
      inFlight = 0;
      for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
        inFlight += stats->inFlight[i];
      }

      // Swap the queues for the next iteration.
      electrons.queues.SwapActive();
      positrons.queues.SwapActive();
      gammas.queues.SwapActive();

      // Check if only charged particles are left that are looping.
      numElectrons = stats->inFlight[ParticleType::Electron];
      numPositrons = stats->inFlight[ParticleType::Positron];
      numGammas    = stats->inFlight[ParticleType::Gamma];
      if (numElectrons == previousElectrons && numPositrons == previousPositrons && numGammas == 0) {
        loopingNo++;
      } else {
        previousElectrons = numElectrons;
        previousPositrons = numPositrons;
        loopingNo         = 0;
      }

    } while (inFlight > 0 && loopingNo < 200);

    if (inFlight > 0) {
      for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
        ParticleType &pType   = particles[i];
        int inFlightParticles = stats->inFlight[i];
        if (inFlightParticles == 0) {
          continue;
        }

        ClearQueue<<<1, 1, 0, stream>>>(pType.queues.currentlyActive);
      }
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
}

void TestEm3(const vecgeom::cxx::VPlacedVolume *world, int numParticles, double energy, int numThreads, int batch,
             double startX, const int *MCIndex_host, ScoringPerVolume *scoringPerVolume_host, int numVolumes,
             GlobalScoring *globalScoring_host)
{
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  cudaManager.LoadGeometry(world);
  cudaManager.Synchronize();

  const vecgeom::cuda::VPlacedVolume *world_dev = cudaManager.world_gpu();

  InitBVH();

  G4HepEmState *state = InitG4HepEm();

  // Transfer MC indices.
  int *MCIndex_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&MCIndex_dev, sizeof(int) * numVolumes));
  COPCORE_CUDA_CHECK(cudaMemcpy(MCIndex_dev, MCIndex_host, sizeof(int) * numVolumes, cudaMemcpyHostToDevice));
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(MCIndex, &MCIndex_dev, sizeof(int *)));

  // Capacity of the different containers aka the maximum number of particles.
  constexpr int Capacity = 1024 * 1024;

  std::cout << "INFO: capacity of containers set to " << Capacity << std::endl;
  if (batch == -1) {
    // Rule of thumb: at most 1000 particles of one type per GeV primary.
    batch = Capacity / ((int)energy / copcore::units::GeV) / 1000;
  } else if (batch < 1) {
    batch = 1;
  }
  std::cout << "INFO: batching " << batch << " particles for transport on the GPU" << std::endl;
  if (BzFieldValue != 0) {
    std::cout << "INFO: running with field Bz = " << BzFieldValue / copcore::units::tesla << " T" << std::endl;
  } else {
    std::cout << "INFO: running with magnetic field OFF" << std::endl;
  }

  // Allocate memory to score charged track length and energy deposit per volume.
  double *chargedTrackLength = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&chargedTrackLength, sizeof(double) * numVolumes));
  COPCORE_CUDA_CHECK(cudaMemset(chargedTrackLength, 0, sizeof(double) * numVolumes));
  double *energyDeposit = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&energyDeposit, sizeof(double) * numVolumes));
  COPCORE_CUDA_CHECK(cudaMemset(energyDeposit, 0, sizeof(double) * numVolumes));

  // Allocate and initialize scoring data structures.
  GlobalScoring *globalScoring = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&globalScoring, sizeof(GlobalScoring)));
  COPCORE_CUDA_CHECK(cudaMemset(globalScoring, 0, sizeof(GlobalScoring)));

  ScoringPerVolume *scoringPerVolume = nullptr;
  ScoringPerVolume scoringPerVolume_devPtrs;
  scoringPerVolume_devPtrs.chargedTrackLength = chargedTrackLength;
  scoringPerVolume_devPtrs.energyDeposit      = energyDeposit;
  COPCORE_CUDA_CHECK(cudaMalloc(&scoringPerVolume, sizeof(ScoringPerVolume)));
  COPCORE_CUDA_CHECK(
      cudaMemcpy(scoringPerVolume, &scoringPerVolume_devPtrs, sizeof(ScoringPerVolume), cudaMemcpyHostToDevice));

  // Allocate memory to hold a "vanilla" SlotManager to initialize for each batch.
  SlotManager slotManagerInit(Capacity);
  SlotManager *slotManagerInit_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&slotManagerInit_dev, sizeof(SlotManager)));
  COPCORE_CUDA_CHECK(cudaMemcpy(slotManagerInit_dev, &slotManagerInit, sizeof(SlotManager), cudaMemcpyHostToDevice));

  // Set up the threads, including their allocation.
  std::vector<std::thread> threads(numThreads);
  std::vector<ThreadData> threadData(numThreads);

  for (int t = 0; t < numThreads; t++) {
    ThreadData &data      = threadData[t];
    data.world            = world_dev;
    data.MCIndex          = MCIndex_dev;
    data.slotManagerInit  = slotManagerInit_dev;
    data.scoringPerVolume = scoringPerVolume;
    data.globalScoring    = globalScoring;

    data.id           = t;
    data.threads      = numThreads;
    data.numParticles = numParticles;
    data.batch        = batch;
    data.energy       = energy;
    data.startX       = startX;

    data.Allocate(Capacity);
  }

  // Start the clock and launch the threads.
  vecgeom::Stopwatch timer;
  timer.Start();

  std::cout << std::endl << "Simulating particles ...";

  for (int t = 0; t < numThreads; t++) {
    threads[t] = std::thread(Worker, &threadData[t]);
  }

  // Join the threads.
  for (auto &&t : threads) {
    t.join();
  }

  std::cout << " done!" << std::endl;

  auto time = timer.Stop();
  std::cout << "Run time: " << time << "\n";

  // Transfer back scoring.
  COPCORE_CUDA_CHECK(cudaMemcpy(globalScoring_host, globalScoring, sizeof(GlobalScoring), cudaMemcpyDeviceToHost));

  // Transfer back the scoring per volume (charged track length and energy deposit).
  COPCORE_CUDA_CHECK(cudaMemcpy(scoringPerVolume_host->chargedTrackLength, scoringPerVolume_devPtrs.chargedTrackLength,
                                sizeof(double) * numVolumes, cudaMemcpyDeviceToHost));
  COPCORE_CUDA_CHECK(cudaMemcpy(scoringPerVolume_host->energyDeposit, scoringPerVolume_devPtrs.energyDeposit,
                                sizeof(double) * numVolumes, cudaMemcpyDeviceToHost));

  // Free resources.
  for (auto &&d : threadData) {
    d.Free();
  }

  COPCORE_CUDA_CHECK(cudaFree(MCIndex_dev));
  COPCORE_CUDA_CHECK(cudaFree(chargedTrackLength));
  COPCORE_CUDA_CHECK(cudaFree(energyDeposit));

  COPCORE_CUDA_CHECK(cudaFree(globalScoring));
  COPCORE_CUDA_CHECK(cudaFree(scoringPerVolume));

  COPCORE_CUDA_CHECK(cudaFree(slotManagerInit_dev));

  FreeG4HepEm(state);
}
