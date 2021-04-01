// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example9.h"
#include "example9.cuh"

#include <AdePT/Atomic.h>
#include <AdePT/LoopNavigator.h>
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

__constant__ __device__ struct G4HepEmParameters g4HepEmPars;
__constant__ __device__ struct G4HepEmData g4HepEmData;

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
__global__ void InitPrimaries(ParticleGenerator generator, int particles, double energy,
                              const vecgeom::VPlacedVolume *world)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particles; i += blockDim.x * gridDim.x) {
    Track &track = generator.NextTrack();

    track.rngState.SetSeed(314159265 * (i + 1));
    track.energy       = energy;
    track.numIALeft[0] = -1.0;
    track.numIALeft[1] = -1.0;
    track.numIALeft[2] = -1.0;

    track.pos = {0, 0, 0};
    track.dir = {1.0, 0, 0};
    LoopNavigator::LocatePointIn(world, track.pos, track.currentState, true);
    // nextState is initialized as needed.
  }
}

// A data structure to transfer statistics after each iteration.
struct Stats {
  GlobalScoring scoring;
  int inFlight[ParticleType::NumParticleTypes];
};

// Finish iteration: clear queues and fill statistics.
__global__ void FinishIteration(AllParticleQueues all, const GlobalScoring *scoring, Stats *stats)
{
  stats->scoring = *scoring;
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    all.queues[i].currentlyActive->clear();
    stats->inFlight[i] = all.queues[i].nextActive->size();
    all.queues[i].relocate->clear();
  }
}

void example9(const vecgeom::cxx::VPlacedVolume *world, int numParticles, double energy)
{
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  cudaManager.LoadGeometry(world);
  cudaManager.Synchronize();

  const vecgeom::cuda::VPlacedVolume *world_dev = cudaManager.world_gpu();

  G4HepEmState *state = InitG4HepEm();

  // Capacity of the different containers aka the maximum number of particles.
  constexpr int Capacity = 256 * 1024;

  std::cout << "INFO: capacity of containers set to " << Capacity << std::endl;

  // Allocate structures to manage tracks of an implicit type:
  //  * memory to hold the actual Track elements,
  //  * objects to manage slots inside the memory,
  //  * queues of slots to remember active particle and those needing relocation,
  //  * a stream and an event for synchronization of kernels.
  constexpr size_t TracksSize  = sizeof(Track) * Capacity;
  constexpr size_t ManagerSize = sizeof(SlotManager);
  const size_t QueueSize       = adept::MParray::SizeOfInstance(Capacity);

  ParticleType particles[ParticleType::NumParticleTypes];
  SlotManager slotManagerInit(Capacity);
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].tracks, TracksSize));

    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].slotManager, ManagerSize));
    COPCORE_CUDA_CHECK(cudaMemcpy(particles[i].slotManager, &slotManagerInit, ManagerSize, cudaMemcpyHostToDevice));

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

  // Allocate and initialize scoring and statistics.
  GlobalScoring *scoring = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&scoring, sizeof(GlobalScoring)));
  COPCORE_CUDA_CHECK(cudaMemset(scoring, 0, sizeof(GlobalScoring)));

  Stats *stats_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&stats_dev, sizeof(Stats)));
  Stats *stats = nullptr;
  COPCORE_CUDA_CHECK(cudaMallocHost(&stats, sizeof(Stats)));

  // Initialize primary particles.
  constexpr int InitThreads = 32;
  int initBlocks            = (numParticles + InitThreads - 1) / InitThreads;
  ParticleGenerator electronGenerator(electrons.tracks, electrons.slotManager, electrons.queues.currentlyActive);
  InitPrimaries<<<initBlocks, InitThreads>>>(electronGenerator, numParticles, energy, world_dev);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  stats->inFlight[ParticleType::Electron] = numParticles;
  stats->inFlight[ParticleType::Positron] = 0;
  stats->inFlight[ParticleType::Gamma]    = 0;

  std::cout << "INFO: running with field Bz = " << BzFieldValue / copcore::units::tesla << " T";
  std::cout << std::endl;

  constexpr int MaxBlocks        = 1024;
  constexpr int TransportThreads = 32;
  constexpr int RelocateThreads  = 32;
  int transportBlocks, relocateBlocks;

  vecgeom::Stopwatch timer;
  timer.Start();

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

      TransportElectrons</*IsElectron*/ true><<<transportBlocks, TransportThreads, 0, electrons.stream>>>(
          electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive,
          electrons.queues.relocate, scoring);

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

      TransportElectrons</*IsElectron*/ false><<<transportBlocks, TransportThreads, 0, positrons.stream>>>(
          positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive,
          positrons.queues.relocate, scoring);

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
          gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive,
          gammas.queues.relocate, scoring);

      RelocateToNextVolume<<<relocateBlocks, RelocateThreads, 0, gammas.stream>>>(gammas.tracks,
                                                                                  gammas.queues.relocate);

      COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
      COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, gammas.event, 0));
    }

    // *** END OF TRANSPORT ***

    // The events ensure synchronization before finishing this iteration and
    // copying the Stats back to the host.
    AllParticleQueues queues = {{electrons.queues, positrons.queues, gammas.queues}};
    FinishIteration<<<1, 1, 0, stream>>>(queues, scoring, stats_dev);
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

    std::cout << std::fixed << std::setprecision(4) << std::setfill(' ');
    std::cout << "iter " << std::setw(4) << iterNo << " -- tracks in flight: " << std::setw(5) << inFlight
              << " energy deposition: " << std::setw(10) << stats->scoring.energyDeposit / copcore::units::GeV
              << " number of secondaries: " << std::setw(5) << stats->scoring.secondaries
              << " number of hits: " << std::setw(4) << stats->scoring.hits;
    std::cout << std::endl;

    iterNo++;
  } while (inFlight > 0 && iterNo < 1000);

  auto time_cpu = timer.Stop();
  std::cout << "Run time: " << time_cpu << "\n";

  // Free resources.
  COPCORE_CUDA_CHECK(cudaFree(scoring));
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

  FreeG4HepEm(state);
}
