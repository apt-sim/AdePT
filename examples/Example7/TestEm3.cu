// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "TestEm3.h"
#include "TestEm3.cuh"

#include <AdePT/base/Atomic.h>
#include <AdePT/base/LoopNavigator.h>
#include <AdePT/base/MParray.h>

#include <CopCore/Global.h>
#include <CopCore/PhysicalConstants.h>
#include <CopCore/Ranluxpp.h>

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Stopwatch.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

#include <G4HepEmData.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmState.hh>

#include <iostream>
#include <iomanip>
#include <stdio.h>

__constant__ __device__ struct G4HepEmParameters g4HepEmPars;
__constant__ __device__ struct G4HepEmData g4HepEmData;

__constant__ __device__ int *MCIndex = nullptr;

G4HepEmData dataOnDevice;

static void CopyG4HepEmToDevice(G4HepEmState *hostState)
{
  // params don't need a free
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmPars, hostState->fParameters, sizeof(G4HepEmParameters)));

  // data is awkward as G4HepEmData stores both host and device pointers
  // We have to get the constant/device g4HepEmData struct's main pointers as device pointers
  // 1. Populate the device side pointers... the state/data will own this
  CopyG4HepEmDataToGPU(hostState->fData);

  // 2. Create G4HepEmData with the device pointers. This is just temporary, and the pointers aren't owned...
  //G4HepEmData dataOnDevice;
  dataOnDevice.fTheMatCutData   = hostState->fData->fTheMatCutData_gpu;
  dataOnDevice.fTheMaterialData = hostState->fData->fTheMaterialData_gpu;
  dataOnDevice.fTheElementData  = hostState->fData->fTheElementData_gpu;
  dataOnDevice.fTheElectronData = hostState->fData->fTheElectronData_gpu;
  dataOnDevice.fThePositronData = hostState->fData->fThePositronData_gpu;
  dataOnDevice.fTheSBTableData  = hostState->fData->fTheSBTableData_gpu;
  dataOnDevice.fTheGammaData    = hostState->fData->fTheGammaData_gpu;

  // 3. Copy this to g4HepEmData
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmData, &dataOnDevice, sizeof(G4HepEmData)));
}


// A bundle of queues per particle type:
//  * Two for active particles, one for the current iteration and the second for the next.
//  * One for all particles that need to be relocated to the next volume.
struct ParticleQueues {
  adept::MParray *currentlyActive;
  adept::MParray *nextActive;
  adept::MParray *relocate;

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
  adept::MParray::MakeInstanceAt(Capacity, queues.relocate);
}

// Kernel function to initialize a set of primary particles.
__global__ void InitPrimaries(ParticleGenerator generator, int startEvent, int numEvents, double energy, double startX,
                              const vecgeom::VPlacedVolume *world, GlobalScoring *globalScoring)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEvents; i += blockDim.x * gridDim.x) {
    Track &track = generator.NextTrack();

    track.rngState.SetSeed(314159265 * (startEvent + i));
    track.energy       = energy;
    track.numIALeft[0] = -1.0;
    track.numIALeft[1] = -1.0;
    track.numIALeft[2] = -1.0;

    track.pos = {vecgeom::Precision(startX), 0, 0};
    track.dir = {1.0, 0, 0};
    track.currentState.Clear();
    LoopNavigator::LocatePointIn(world, track.pos, track.currentState, true);
    // nextState is initialized as needed.

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
    all.queues[i].relocate->clear();
  }
}

// Deposit energy of particles still in flight.
__global__ void DepositEnergy(Track *allTracks, const adept::MParray *queue, GlobalScoring *globalScoring,
                              ScoringPerVolume *scoringPerVolume)
{
  int queueSize = queue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < queueSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*queue)[i];
    Track &currentTrack = allTracks[slot];
    auto volume         = currentTrack.currentState.Top();
    if (volume == nullptr) {
      // The particle left the world, why wasn't it killed before?!
      continue;
    }
    int volumeID = volume->id();

    double energy = currentTrack.energy;
    atomicAdd(&globalScoring->energyDeposit, energy);
    atomicAdd(&scoringPerVolume->energyDeposit[volumeID], energy);
  }
}

__global__ void ClearQueue(adept::MParray *queue)
{
  queue->clear();
}

void TestEm3(const vecgeom::cxx::VPlacedVolume *world, int numParticles, double energy, int batch, double startX,
             const int *MCIndex_host, ScoringPerVolume *scoringPerVolume_host, int numVolumes,
             GlobalScoring *globalScoring_host,
             G4HepEmState* hepEMState_host)
{
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  cudaManager.LoadGeometry(world);
  cudaManager.Synchronize();

  const vecgeom::cuda::VPlacedVolume *world_dev = cudaManager.world_gpu();

  CopyG4HepEmToDevice(hepEMState_host);

  // Transfer MC indices.
  int *MCIndex_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&MCIndex_dev, sizeof(int) * numVolumes));
  COPCORE_CUDA_CHECK(cudaMemcpy(MCIndex_dev, MCIndex_host, sizeof(int) * numVolumes, cudaMemcpyHostToDevice));
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(MCIndex, &MCIndex_dev, sizeof(int *)));

  // Capacity of the different containers aka the maximum number of particles.
  constexpr int Capacity = 256 * 1024;

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

  // Allocate structures to manage tracks of an implicit type:
  //  * memory to hold the actual Track elements,
  //  * objects to manage slots inside the memory,
  //  * queues of slots to remember active particle and those needing relocation,
  //  * a stream and an event for synchronization of kernels.
  constexpr size_t TracksSize  = sizeof(Track) * Capacity;
  constexpr size_t ManagerSize = sizeof(SlotManager);
  const size_t QueueSize       = adept::MParray::SizeOfInstance(Capacity);

  ParticleType particles[ParticleType::NumParticleTypes];
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].tracks, TracksSize));

    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].slotManager, ManagerSize));

    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].queues.currentlyActive, QueueSize));
    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].queues.nextActive, QueueSize));
    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].queues.relocate, QueueSize));
    InitParticleQueues<<<1, 1>>>(particles[i].queues, Capacity);

    COPCORE_CUDA_CHECK(cudaStreamCreate(&particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventCreate(&particles[i].event));
  }
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  ParticleType &electrons = particles[ParticleType::Electron];
  ParticleType &positrons = particles[ParticleType::Positron];
  ParticleType &gammas    = particles[ParticleType::Gamma];

  // Create a stream to synchronize kernels of all particle types.
  cudaStream_t stream;
  COPCORE_CUDA_CHECK(cudaStreamCreate(&stream));

  // Allocate memory to score charged track length and energy deposit per volume.
  double *chargedTrackLength = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&chargedTrackLength, sizeof(double) * numVolumes));
  COPCORE_CUDA_CHECK(cudaMemset(chargedTrackLength, 0, sizeof(double) * numVolumes));
  double *energyDeposit = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&energyDeposit, sizeof(double) * numVolumes));
  COPCORE_CUDA_CHECK(cudaMemset(energyDeposit, 0, sizeof(double) * numVolumes));

  // Allocate and initialize scoring and statistics.
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

  Stats *stats_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&stats_dev, sizeof(Stats)));
  Stats *stats = nullptr;
  COPCORE_CUDA_CHECK(cudaMallocHost(&stats, sizeof(Stats)));

  vecgeom::Stopwatch timer;
  timer.Start();

  std::cout << std::endl << "Simulating particle ";
  for (int startEvent = 1; startEvent <= numParticles; startEvent += batch) {
    std::cout << startEvent << " ... " << std::flush;
    int left  = numParticles - startEvent + 1;
    int chunk = std::min(left, batch);

    SlotManager slotManagerInit(Capacity);
    for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
      COPCORE_CUDA_CHECK(cudaMemcpy(particles[i].slotManager, &slotManagerInit, ManagerSize, cudaMemcpyHostToDevice));
    }

    // Initialize primary particles.
    constexpr int InitThreads = 32;
    int initBlocks            = (chunk + InitThreads - 1) / InitThreads;
    ParticleGenerator electronGenerator(electrons.tracks, electrons.slotManager, electrons.queues.currentlyActive);
    InitPrimaries<<<initBlocks, InitThreads>>>(electronGenerator, startEvent, chunk, energy, startX, world_dev,
                                               globalScoring);
    COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

    stats->inFlight[ParticleType::Electron] = numParticles;
    stats->inFlight[ParticleType::Positron] = 0;
    stats->inFlight[ParticleType::Gamma]    = 0;

    constexpr int MaxBlocks        = 1024;
    constexpr int TransportThreads = 32;
    constexpr int RelocateThreads  = 32;
    int transportBlocks, relocateBlocks;

    int inFlight;
    int iterNo = 0;

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

        relocateBlocks = std::min(numElectrons, MaxBlocks);

        TransportElectrons<<<transportBlocks, TransportThreads, 0, electrons.stream>>>(
            electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive,
            electrons.queues.relocate, globalScoring, scoringPerVolume);

        RelocateToNextVolume<<<relocateBlocks, RelocateThreads, 0, electrons.stream>>>(electrons.tracks,
                                                                                       electrons.queues.relocate);

        COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, electrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, electrons.event, 0));
      }

      // *** POSITRONS ***
      int numPositrons = stats->inFlight[ParticleType::Positron];
      if (numPositrons > 0) {
        transportBlocks = (numPositrons + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        relocateBlocks = std::min(numPositrons, MaxBlocks);

        TransportPositrons<<<transportBlocks, TransportThreads, 0, positrons.stream>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive,
            positrons.queues.relocate, globalScoring, scoringPerVolume);

        RelocateToNextVolume<<<relocateBlocks, RelocateThreads, 0, positrons.stream>>>(positrons.tracks,
                                                                                       positrons.queues.relocate);

        COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, positrons.event, 0));
      }

      // *** GAMMAS ***
      int numGammas = stats->inFlight[ParticleType::Gamma];
      if (numGammas > 0) {
        transportBlocks = (numGammas + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        relocateBlocks = std::min(numGammas, MaxBlocks);

        TransportGammas<<<transportBlocks, TransportThreads, 0, gammas.stream>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive, gammas.queues.relocate,
            globalScoring, scoringPerVolume);

        RelocateToNextVolume<<<relocateBlocks, RelocateThreads, 0, gammas.stream>>>(gammas.tracks,
                                                                                    gammas.queues.relocate);

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

      iterNo++;
    } while (inFlight > 0 && iterNo < 1000);

    if (inFlight > 0) {
      std::cout << std::endl;
      std::cout << "WARN: Depositing energy of " << inFlight << " particles still in flight!" << std::endl;
      constexpr int DepositThreads = 32;

      for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
        ParticleType &pType   = particles[i];
        int inFlightParticles = stats->inFlight[i];
        if (inFlightParticles == 0) {
          continue;
        }

        int depositBlocks = (inFlightParticles + DepositThreads - 1) / DepositThreads;
        depositBlocks     = std::min(depositBlocks, MaxBlocks);
        DepositEnergy<<<depositBlocks, DepositThreads, 0, stream>>>(pType.tracks, pType.queues.currentlyActive,
                                                                    globalScoring, scoringPerVolume);
        ClearQueue<<<1, 1, 0, stream>>>(pType.queues.currentlyActive);
      }
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));
      std::cout << " ... ";
    }
  }
  std::cout << "done!" << std::endl;

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
  COPCORE_CUDA_CHECK(cudaFree(MCIndex_dev));
  COPCORE_CUDA_CHECK(cudaFree(chargedTrackLength));
  COPCORE_CUDA_CHECK(cudaFree(energyDeposit));

  COPCORE_CUDA_CHECK(cudaFree(globalScoring));
  COPCORE_CUDA_CHECK(cudaFree(scoringPerVolume));
  COPCORE_CUDA_CHECK(cudaFree(stats_dev));
  COPCORE_CUDA_CHECK(cudaFreeHost(stats));

  COPCORE_CUDA_CHECK(cudaStreamDestroy(stream));

  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    COPCORE_CUDA_CHECK(cudaFree(particles[i].tracks));
    COPCORE_CUDA_CHECK(cudaFree(particles[i].slotManager));

    COPCORE_CUDA_CHECK(cudaFree(particles[i].queues.currentlyActive));
    COPCORE_CUDA_CHECK(cudaFree(particles[i].queues.nextActive));
    COPCORE_CUDA_CHECK(cudaFree(particles[i].queues.relocate));

    COPCORE_CUDA_CHECK(cudaStreamDestroy(particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventDestroy(particles[i].event));
  }

}
