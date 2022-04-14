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

struct ParticleType {
  adept::TrackManager<Track> *trackmgr;
  cudaStream_t stream;
  cudaEvent_t event;

  enum {
    Electron = 0,
    Positron = 1,
    Gamma    = 2,

    NumParticleTypes,
  };
};

// Track managers for the three particle types.
struct AllTrackManagers {
  adept::TrackManager<Track> *trackmgr[ParticleType::NumParticleTypes];
};

// Kernel function to initialize a set of primary particles.
__global__ void InitPrimaries(adept::TrackManager<Track> *trackmgr, int startEvent, int numEvents, double energy,
                              double startX, const vecgeom::VPlacedVolume *world, GlobalScoring *globalScoring)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEvents; i += blockDim.x * gridDim.x) {
    Track &track = trackmgr->NextTrack();

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
  adept::TrackManager<Track>::Stats mgr_stats[ParticleType::NumParticleTypes];
};

// Finish iteration: refresh track managers and fill statistics.
__global__ void FinishIteration(AllTrackManagers all, Stats *stats)
{
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    all.trackmgr[i]->refresh_stats();
    stats->mgr_stats[i] = all.trackmgr[i]->fStats;
  }
}

void TestEm3(const vecgeom::cxx::VPlacedVolume *world, int numParticles, double energy, int batch, double startX,
             const int *MCIndex_host, ScoringPerVolume *scoringPerVolume_host, int numVolumes,
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
    // Rule of thumb: at most 200 particles at a time of one type per GeV primary.
    batch = Capacity / ((int)energy / copcore::units::GeV) / 200;
  } else if (batch < 1) {
    batch = 1;
  }
  std::cout << "INFO: batching " << batch << " particles for transport on the GPU" << std::endl;
  if (BzFieldValue != 0) {
    std::cout << "INFO: running with field Bz = " << BzFieldValue / copcore::units::tesla << " T" << std::endl;
  } else {
    std::cout << "INFO: running with magnetic field OFF" << std::endl;
  }

  // Allocate track managers, streams and synchronizaion events.
  AllTrackManagers allmgr_h, allmgr_d;
  ParticleType particles[ParticleType::NumParticleTypes];
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    allmgr_h.trackmgr[i]  = new adept::TrackManager<Track>(Capacity);
    allmgr_d.trackmgr[i]  = allmgr_h.trackmgr[i]->ConstructOnDevice();
    particles[i].trackmgr = allmgr_d.trackmgr[i];

    COPCORE_CUDA_CHECK(cudaStreamCreate(&particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventCreate(&particles[i].event));
  }
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  Secondaries secondaries{allmgr_d.trackmgr[0], allmgr_d.trackmgr[1], allmgr_d.trackmgr[2]};

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

  std::cout << std::endl << "Simulating particles ";
  const bool detailed = (numParticles / batch) < 50;
  if (!detailed) {
    std::cout << "... " << std::flush;
  }

  unsigned long long killed = 0;
  int num_compact           = 0;

  for (int startEvent = 1; startEvent <= numParticles; startEvent += batch) {
    if (detailed) {
      std::cout << startEvent << " ... " << std::flush;
    }
    int left  = numParticles - startEvent + 1;
    int chunk = std::min(left, batch);

    // Initialize primary particles.
    constexpr float compactThreshold = 0.9;
    constexpr int InitThreads        = 32;
    int initBlocks                   = (chunk + InitThreads - 1) / InitThreads;
    InitPrimaries<<<initBlocks, InitThreads, 0, stream>>>(electrons.trackmgr, startEvent, chunk, energy, startX,
                                                          world_dev, globalScoring);
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));

    allmgr_h.trackmgr[ParticleType::Electron]->fStats.fInFlight = chunk;
    allmgr_h.trackmgr[ParticleType::Positron]->fStats.fInFlight = 0;
    allmgr_h.trackmgr[ParticleType::Gamma]->fStats.fInFlight    = 0;

    constexpr int MaxBlocks        = 1024;
    constexpr int TransportThreads = 32;
    int transportBlocks;

    int inFlight;
    int loopingNo         = 0;
    int previousElectrons = -1, previousPositrons = -1;

    do {
      // *** ELECTRONS ***
      int numElectrons = allmgr_h.trackmgr[ParticleType::Electron]->fStats.fInFlight;
      if (numElectrons > 0) {
        transportBlocks = (numElectrons + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        TransportElectrons<<<transportBlocks, TransportThreads, 0, electrons.stream>>>(electrons.trackmgr, secondaries,
                                                                                       globalScoring, scoringPerVolume);

        COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, electrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, electrons.event, 0));
      }

      // *** POSITRONS ***
      int numPositrons = allmgr_h.trackmgr[ParticleType::Positron]->fStats.fInFlight;
      if (numPositrons > 0) {
        transportBlocks = (numPositrons + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        TransportPositrons<<<transportBlocks, TransportThreads, 0, positrons.stream>>>(positrons.trackmgr, secondaries,
                                                                                       globalScoring, scoringPerVolume);

        COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, positrons.event, 0));
      }

      // *** GAMMAS ***
      int numGammas = allmgr_h.trackmgr[ParticleType::Gamma]->fStats.fInFlight;
      if (numGammas > 0) {
        transportBlocks = (numGammas + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        TransportGammas<<<transportBlocks, TransportThreads, 0, gammas.stream>>>(gammas.trackmgr, secondaries,
                                                                                 globalScoring, scoringPerVolume);

        COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, gammas.event, 0));
      }

      // *** END OF TRANSPORT ***

      // The events ensure synchronization before finishing this iteration and
      // copying the Stats back to the host.
      FinishIteration<<<1, 1, 0, stream>>>(allmgr_d, stats_dev);
      COPCORE_CUDA_CHECK(cudaMemcpyAsync(stats, stats_dev, sizeof(Stats), cudaMemcpyDeviceToHost, stream));

      // Finally synchronize all kernels.
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));

      // Count the number of particles in flight.
      inFlight = 0;
      for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
        // Update stats for host track manager objects
        allmgr_h.trackmgr[i]->fStats = stats->mgr_stats[i];
        inFlight += stats->mgr_stats[i].fInFlight;
        auto compacted = allmgr_h.trackmgr[i]->SwapAndCompact(compactThreshold, particles[i].stream);
        if (compacted) num_compact++;
      }

      // Check if only charged particles are left that are looping.
      numElectrons = allmgr_h.trackmgr[ParticleType::Electron]->fStats.fInFlight;
      numPositrons = allmgr_h.trackmgr[ParticleType::Positron]->fStats.fInFlight;
      numGammas    = allmgr_h.trackmgr[ParticleType::Gamma]->fStats.fInFlight;
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
        int inFlightParticles = allmgr_h.trackmgr[i]->fStats.fInFlight;
        if (inFlightParticles == 0) {
          continue;
        }

        allmgr_h.trackmgr[i]->Clear(particles[i].stream);
      }
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
  std::cout << "done!\nCompacted track containers " << num_compact << " times.\n";

  auto time = timer.Stop();
  std::cout << "Run time: " << time << "\n";

  // Transfer back scoring.
  COPCORE_CUDA_CHECK(cudaMemcpy(globalScoring_host, globalScoring, sizeof(GlobalScoring), cudaMemcpyDeviceToHost));
  globalScoring_host->numKilled = killed;

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
    allmgr_h.trackmgr[i]->FreeFromDevice();
    delete allmgr_h.trackmgr[i];

    COPCORE_CUDA_CHECK(cudaStreamDestroy(particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventDestroy(particles[i].event));
  }

  FreeG4HepEm(state);
}
