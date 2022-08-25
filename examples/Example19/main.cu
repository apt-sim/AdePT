// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example.h"
#include "example.cuh"

#include <AdePT/Atomic.h>
#include <AdePT/BVHNavigator.h>
#include <AdePT/MParray.h>
#include <AdePT/NVTX.h>

#include <CopCore/Global.h>
#include <CopCore/PhysicalConstants.h>
#include <CopCore/Ranluxpp.h>

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Stopwatch.h>
#include <VecGeom/management/GeoManager.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

#include <G4HepEmState.hh>
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

__constant__ __device__ int *MCIndex = nullptr;

void InitG4HepEmGPU(G4HepEmState *state)
{
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
  SOAData soaData;

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
__global__ void InitPrimaries(ParticleGenerator generator, int startEvent, int numEvents, double energy,
                              const vecgeom::VPlacedVolume *world, GlobalScoring *globalScoring,
                              bool rotatingParticleGun)
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

    track.pos = {0, 0, 0};
    if (rotatingParticleGun) {
      // Generate particles flat in phi and in eta between -5 and 5. We'll lose the far forwards ones, so no need to
      // simulate.
      const double phi = 2. * M_PI * track.rngState.Rndm();
      const double eta = -5. + 10. * track.rngState.Rndm();
      track.dir.x()    = static_cast<vecgeom::Precision>(cos(phi) / cosh(eta));
      track.dir.y()    = static_cast<vecgeom::Precision>(sin(phi) / cosh(eta));
      track.dir.z()    = static_cast<vecgeom::Precision>(tanh(eta));
    } else {
      track.dir = {1.0, 0, 0};
    }
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

void runGPU(int numParticles, double energy, int batch, const int *MCIndex_host,
            ScoringPerVolume *scoringPerVolume_host, GlobalScoring *globalScoring_host, int numVolumes, int numPlaced,
            G4HepEmState *state, bool rotatingParticleGun)
{
  NVTXTracer tracer("InitG4HepEM");
  InitG4HepEmGPU(state);

  tracer.setTag("InitParticles/malloc/copy");
  // Transfer MC indices.
  int *MCIndex_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&MCIndex_dev, sizeof(int) * numVolumes));
  COPCORE_CUDA_CHECK(cudaMemcpy(MCIndex_dev, MCIndex_host, sizeof(int) * numVolumes, cudaMemcpyHostToDevice));
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(MCIndex, &MCIndex_dev, sizeof(int *)));

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  // Capacity of the different containers aka the maximum number of particles.
  // Use 1/5 of GPU memory for each of e+/e-/gammas, leaving 2/5 for the rest.
  const size_t Capacity = (deviceProp.totalGlobalMem / sizeof(Track) / 5);

  std::cout << "INFO: capacity of containers set to " << Capacity << std::endl;
  if (batch == -1) {
    // Rule of thumb: at most 2000 particles of one type per GeV primary.
    batch = Capacity / ((int)energy / copcore::units::GeV) / 2000;
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
  const size_t TracksSize  = sizeof(Track) * Capacity;
  const size_t ManagerSize = sizeof(SlotManager);
  const size_t QueueSize   = adept::MParray::SizeOfInstance(Capacity);

  ParticleType particles[ParticleType::NumParticleTypes];
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].tracks, TracksSize));

    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].slotManager, ManagerSize));

    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].queues.currentlyActive, QueueSize));
    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].queues.nextActive, QueueSize));
    InitParticleQueues<<<1, 1>>>(particles[i].queues, Capacity);

    COPCORE_CUDA_CHECK(cudaStreamCreate(&particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventCreate(&particles[i].event));

    COPCORE_CUDA_CHECK(
        cudaMalloc(&particles[i].soaData.nextInteraction, sizeof(SOAData::nextInteraction[0]) * Capacity));
  }
  COPCORE_CUDA_CHECK(
      cudaMalloc(&particles[ParticleType::Gamma].soaData.gamma_PEmxSec, sizeof(SOAData::gamma_PEmxSec[0]) * Capacity));
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  ParticleType &electrons = particles[ParticleType::Electron];
  ParticleType &positrons = particles[ParticleType::Positron];
  ParticleType &gammas    = particles[ParticleType::Gamma];

  // Create a stream to synchronize kernels of all particle types.
  cudaStream_t stream;
  COPCORE_CUDA_CHECK(cudaStreamCreate(&stream));

  cudaStream_t interactionStreams[3];
  for (auto i = 0; i < 3; ++i)
    COPCORE_CUDA_CHECK(cudaStreamCreate(&interactionStreams[i]));

  // Allocate memory to score charged track length and energy deposit per volume.
  double *chargedTrackLength = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&chargedTrackLength, sizeof(double) * numPlaced));
  COPCORE_CUDA_CHECK(cudaMemset(chargedTrackLength, 0, sizeof(double) * numPlaced));
  double *energyDeposit = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&energyDeposit, sizeof(double) * numPlaced));
  COPCORE_CUDA_CHECK(cudaMemset(energyDeposit, 0, sizeof(double) * numPlaced));

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

  // Allocate memory to hold a "vanilla" SlotManager to initialize for each batch.
  SlotManager slotManagerInit(Capacity);
  SlotManager *slotManagerInit_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&slotManagerInit_dev, sizeof(SlotManager)));
  COPCORE_CUDA_CHECK(cudaMemcpy(slotManagerInit_dev, &slotManagerInit, sizeof(SlotManager), cudaMemcpyHostToDevice));

  vecgeom::Stopwatch timer;
  timer.Start();
  tracer.setTag("sim");

  std::cout << std::endl << "Simulating particles ";
  const bool detailed = (numParticles / batch) < 50;
  if (!detailed) {
    std::cout << "... " << std::flush;
  }

  unsigned long long killed = 0;
  tracer.setTag("start event loop");

  for (int startEvent = 1; startEvent <= numParticles; startEvent += batch) {
    if (detailed) {
      std::cout << startEvent << " ... " << std::flush;
    }
    int left  = numParticles - startEvent + 1;
    int chunk = std::min(left, batch);

    for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
      COPCORE_CUDA_CHECK(cudaMemcpyAsync(particles[i].slotManager, slotManagerInit_dev, ManagerSize,
                                         cudaMemcpyDeviceToDevice, stream));
    }

    // Initialize primary particles.
    constexpr int InitThreads = ThreadsPerBlock;
    int initBlocks            = (chunk + ThreadsPerBlock - 1) / ThreadsPerBlock;
    ParticleGenerator electronGenerator(electrons.tracks, electrons.slotManager, electrons.queues.currentlyActive);
    auto world_dev = vecgeom::cxx::CudaManager::Instance().world_gpu();
    InitPrimaries<<<initBlocks, InitThreads, 0, stream>>>(electronGenerator, startEvent, chunk, energy, world_dev,
                                                          globalScoring, rotatingParticleGun);
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));

    stats->inFlight[ParticleType::Electron] = chunk;
    stats->inFlight[ParticleType::Positron] = 0;
    stats->inFlight[ParticleType::Gamma]    = 0;

    constexpr int MaxBlocks = 8192;
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
        transportBlocks = (numElectrons + ThreadsPerBlock - 1) / ThreadsPerBlock;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        TransportElectrons<<<transportBlocks, ThreadsPerBlock, 0, electrons.stream>>>(
            electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive, globalScoring,
            scoringPerVolume, electrons.soaData);

        COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, electrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(interactionStreams[0], electrons.event, 0));

        IonizationEl<<<transportBlocks, ThreadsPerBlock, 0, interactionStreams[0]>>>(
            electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive, globalScoring,
            scoringPerVolume, electrons.soaData);
        BremsstrahlungEl<<<transportBlocks, ThreadsPerBlock, 0, electrons.stream>>>(
            electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive, globalScoring,
            scoringPerVolume, electrons.soaData);

        for (auto streamToWaitFor : {interactionStreams[0], electrons.stream}) {
          COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, streamToWaitFor));
          COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, electrons.event, 0));
        }
      }

      // *** POSITRONS ***
      int numPositrons = stats->inFlight[ParticleType::Positron];
      if (numPositrons > 0) {
        transportBlocks = (numPositrons + ThreadsPerBlock - 1) / ThreadsPerBlock;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        TransportPositrons<<<transportBlocks, ThreadsPerBlock, 0, positrons.stream>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive, globalScoring,
            scoringPerVolume, positrons.soaData);

        COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(interactionStreams[1], positrons.event, 0));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(interactionStreams[2], positrons.event, 0));

        IonizationPos<<<transportBlocks, ThreadsPerBlock, 0, interactionStreams[1]>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive, globalScoring,
            scoringPerVolume, positrons.soaData);
        BremsstrahlungPos<<<transportBlocks, ThreadsPerBlock, 0, positrons.stream>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive, globalScoring,
            scoringPerVolume, positrons.soaData);
        AnnihilationPos<<<transportBlocks, ThreadsPerBlock, 0, interactionStreams[2]>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive, globalScoring,
            scoringPerVolume, positrons.soaData);

        for (auto streamToWaitFor : {interactionStreams[1], positrons.stream, interactionStreams[2]}) {
          COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, streamToWaitFor));
          COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, positrons.event, 0));
        }
      }

      // *** GAMMAS ***
      int numGammas = stats->inFlight[ParticleType::Gamma];
      if (numGammas > 0) {
        transportBlocks = (numGammas + ThreadsPerBlock - 1) / ThreadsPerBlock;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        TransportGammas<<<transportBlocks, ThreadsPerBlock, 0, gammas.stream>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive, globalScoring,
            scoringPerVolume, gammas.soaData);

        COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, gammas.event, 0));

        for (auto i = 0; i < 3; ++i) {
          COPCORE_CUDA_CHECK(cudaStreamWaitEvent(interactionStreams[i], gammas.event, 0));
        }
        // About 2% of all gammas:
        PairCreation<<<transportBlocks, ThreadsPerBlock, 0, interactionStreams[0]>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive, globalScoring,
            scoringPerVolume, gammas.soaData);
        // About 10% of all gammas:
        ComptonScattering<<<transportBlocks, ThreadsPerBlock, 0, interactionStreams[1]>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive, globalScoring,
            scoringPerVolume, gammas.soaData);
        // About 15% of all gammas:
        PhotoelectricEffect<<<transportBlocks, ThreadsPerBlock, 0, interactionStreams[2]>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive, globalScoring,
            scoringPerVolume, gammas.soaData);
        for (auto i = 0; i < 3; ++i) {
          COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, interactionStreams[i]));
          COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, positrons.event, 0));
        }
        COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, positrons.event, 0));
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

      tracer.setOccupancy(inFlight);

      tracer.setOccupancy(inFlight);

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
      killed += inFlight;
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
  std::cout << "done!" << std::endl;

  auto time = timer.Stop();
  std::cout << "Run time: " << time << "\n";

  // Transfer back scoring.
  COPCORE_CUDA_CHECK(cudaMemcpy(globalScoring_host, globalScoring, sizeof(GlobalScoring), cudaMemcpyDeviceToHost));
  globalScoring_host->numKilled = killed;

  // Transfer back the scoring per volume (charged track length and energy deposit).
  COPCORE_CUDA_CHECK(cudaMemcpy(scoringPerVolume_host->chargedTrackLength, scoringPerVolume_devPtrs.chargedTrackLength,
                                sizeof(double) * numPlaced, cudaMemcpyDeviceToHost));
  COPCORE_CUDA_CHECK(cudaMemcpy(scoringPerVolume_host->energyDeposit, scoringPerVolume_devPtrs.energyDeposit,
                                sizeof(double) * numPlaced, cudaMemcpyDeviceToHost));

  // Free resources.
  COPCORE_CUDA_CHECK(cudaFree(MCIndex_dev));
  COPCORE_CUDA_CHECK(cudaFree(chargedTrackLength));
  COPCORE_CUDA_CHECK(cudaFree(energyDeposit));

  COPCORE_CUDA_CHECK(cudaFree(globalScoring));
  COPCORE_CUDA_CHECK(cudaFree(scoringPerVolume));
  COPCORE_CUDA_CHECK(cudaFree(stats_dev));
  COPCORE_CUDA_CHECK(cudaFreeHost(stats));
  COPCORE_CUDA_CHECK(cudaFree(slotManagerInit_dev));

  COPCORE_CUDA_CHECK(cudaStreamDestroy(stream));

  for (auto i = 0; i < 3; ++i)
    COPCORE_CUDA_CHECK(cudaStreamDestroy(interactionStreams[i]));

  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    COPCORE_CUDA_CHECK(cudaFree(particles[i].tracks));
    COPCORE_CUDA_CHECK(cudaFree(particles[i].slotManager));

    COPCORE_CUDA_CHECK(cudaFree(particles[i].queues.currentlyActive));
    COPCORE_CUDA_CHECK(cudaFree(particles[i].queues.nextActive));

    COPCORE_CUDA_CHECK(cudaStreamDestroy(particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventDestroy(particles[i].event));

    COPCORE_CUDA_CHECK(cudaFree(particles[i].soaData.nextInteraction));
    if (particles[i].soaData.gamma_PEmxSec) COPCORE_CUDA_CHECK(cudaFree(particles[i].soaData.gamma_PEmxSec));
  }
}
