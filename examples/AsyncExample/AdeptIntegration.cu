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

#include "electrons.cuh"
#include "gammas.cuh"

__constant__ __device__ struct G4HepEmParameters g4HepEmPars;
__constant__ __device__ struct G4HepEmData g4HepEmData;

__constant__ __device__ adeptint::VolAuxData *gVolAuxData = nullptr;
__constant__ __device__ double BzFieldValue               = 0;

G4HepEmState *AdeptIntegration::fg4hepem_state{nullptr};
SlotManager *slotManagerInit_dev = nullptr;
int AdeptIntegration::kCapacity = 1024 * 1024;

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
__global__ void InitTracks(adeptint::TrackData *trackinfo, int ntracks, int startTrack, int event,
                           Secondaries secondaries, const vecgeom::VPlacedVolume *world, AdeptScoring *userScoring)
{
  constexpr double tolerance = 10. * vecgeom::kTolerance;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ntracks; i += blockDim.x * gridDim.x) {
    ParticleGenerator *generator = nullptr;
    // These tracks come from Geant4, do not count them here
    switch (trackinfo[i].pdg) {
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
    track.rngState.SetSeed(1234567 * event + startTrack + i);
    track.energy       = trackinfo[i].energy;
    track.numIALeft[0] = -1.0;
    track.numIALeft[1] = -1.0;
    track.numIALeft[2] = -1.0;

    track.initialRange       = -1.0;
    track.dynamicRangeFactor = -1.0;
    track.tlimitMin          = -1.0;

    track.pos = {trackinfo[i].position[0], trackinfo[i].position[1], trackinfo[i].position[2]};
    track.dir = {trackinfo[i].direction[0], trackinfo[i].direction[1], trackinfo[i].direction[2]};
    track.navState.Clear();
    // We locate the pushed point because we run the risk that the
    // point is not located in the GPU region
    BVHNavigator::LocatePointIn(world, track.pos + tolerance * track.dir, track.navState, true);
    // The track must be on boundary at this point
    track.navState.SetBoundaryState(true);
    // nextState is initialized as needed.
    auto volume                         = track.navState.Top();
    int lvolID                          = volume->GetLogicalVolume()->id();
    adeptint::VolAuxData const &auxData = userScoring->GetAuxData_dev(lvolID);
    assert(auxData.fGPUregion);
  }
}

// Copy particles leaked from the GPU region into a compact buffer
__global__ void FillFromDeviceBuffer(int numLeaked, AllLeaked all, adeptint::TrackData *fromDevice)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numLeaked) return;
  int numElectrons = all.leakedElectrons.fLeakedQueue->size();
  int numPositrons = all.leakedPositrons.fLeakedQueue->size();
  int numGammas    = all.leakedGammas.fLeakedQueue->size();
  assert(numLeaked == numElectrons + numPositrons + numGammas);

  const Track *track{nullptr};
  if (i < numElectrons) {
    int trackIndex    = (*all.leakedElectrons.fLeakedQueue)[i];
    track             = &all.leakedElectrons.fTracks[trackIndex];
    fromDevice[i].pdg = 11;
  } else if (i < numElectrons + numPositrons) {
    int trackIndex    = (*all.leakedPositrons.fLeakedQueue)[i - numElectrons];
    track             = &all.leakedPositrons.fTracks[trackIndex];
    fromDevice[i].pdg = -11;
  } else {
    int trackIndex    = (*all.leakedGammas.fLeakedQueue)[i - numElectrons - numPositrons];
    track             = &all.leakedGammas.fTracks[trackIndex];
    fromDevice[i].pdg = 22;
  }

  fromDevice[i].position[0]  = track->pos[0];
  fromDevice[i].position[1]  = track->pos[1];
  fromDevice[i].position[2]  = track->pos[2];
  fromDevice[i].direction[0] = track->dir[0];
  fromDevice[i].direction[1] = track->dir[1];
  fromDevice[i].direction[2] = track->dir[2];
  fromDevice[i].energy       = track->energy;
}

// Finish iteration: clear queues and fill statistics.
__global__ void FinishIteration(AllParticleQueues all, Stats *stats)
{
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    all.queues[i].currentlyActive->clear();
    stats->inFlight[i]     = all.queues[i].nextActive->size();
    stats->leakedTracks[i] = all.queues[i].leakedTracks->size();
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
    all.queues[i].leakedTracks->clear();
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
  SlotManager slotManagerInit(kCapacity);
  COPCORE_CUDA_CHECK(cudaMalloc(&slotManagerInit_dev, sizeof(SlotManager)));
  COPCORE_CUDA_CHECK(cudaMemcpy(slotManagerInit_dev, &slotManagerInit, sizeof(SlotManager), cudaMemcpyHostToDevice));

  return true;
}

void AdeptIntegration::InitializeGPU()
{
  using TrackData    = adeptint::TrackData;
  fGPUstate          = new GPUstate;
  GPUstate &gpuState = *fGPUstate;

  // Capacity of the different containers aka the maximum number of particles.
  G4cout << "INFO: batching " << fMaxBatch << " particles for transport on the GPU" << G4endl;

  // Allocate structures to manage tracks of an implicit type:
  //  * memory to hold the actual Track elements,
  //  * objects to manage slots inside the memory,
  //  * queues of slots to remember active particle and those needing relocation,
  //  * a stream and an event for synchronization of kernels.
  size_t TracksSize            = sizeof(Track) * kCapacity;
  constexpr size_t ManagerSize = sizeof(SlotManager);
  const size_t QueueSize       = adept::MParray::SizeOfInstance(kCapacity);
  // Create a stream to synchronize kernels of all particle types.
  COPCORE_CUDA_CHECK(cudaStreamCreate(&gpuState.stream));
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    // Share hepem state between threads
    COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.particles[i].tracks, TracksSize));

    COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.particles[i].slotManager, ManagerSize));

    COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.particles[i].queues.currentlyActive, QueueSize));
    COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.particles[i].queues.nextActive, QueueSize));
    COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.particles[i].queues.leakedTracks, QueueSize));
    InitParticleQueues<<<1, 1>>>(gpuState.particles[i].queues, kCapacity);

    COPCORE_CUDA_CHECK(cudaStreamCreate(&gpuState.particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventCreate(&gpuState.particles[i].event));
  }
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  // initialize statistics
  COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.stats_dev, sizeof(Stats)));
  COPCORE_CUDA_CHECK(cudaMallocHost(&gpuState.stats, sizeof(Stats)));

  // initialize buffers of tracks on device
  COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.toDevice_dev, fMaxBatch * sizeof(TrackData)));
  gpuState.fNumFromDevice = 1000;
  COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.fromDevice_dev, gpuState.fNumFromDevice * sizeof(TrackData)));
  fBuffer.fromDevice = new TrackData[1000];
}

void AdeptIntegration::FreeGPU()
{
  // Free resources.
  GPUstate &gpuState = *static_cast<GPUstate *>(fGPUstate);
  COPCORE_CUDA_CHECK(cudaFree(gpuState.stats_dev));
  COPCORE_CUDA_CHECK(cudaFreeHost(gpuState.stats));
  COPCORE_CUDA_CHECK(cudaFree(gpuState.toDevice_dev));

  COPCORE_CUDA_CHECK(cudaStreamDestroy(gpuState.stream));

  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    COPCORE_CUDA_CHECK(cudaFree(gpuState.particles[i].tracks));
    COPCORE_CUDA_CHECK(cudaFree(gpuState.particles[i].slotManager));

    COPCORE_CUDA_CHECK(cudaFree(gpuState.particles[i].queues.currentlyActive));
    COPCORE_CUDA_CHECK(cudaFree(gpuState.particles[i].queues.nextActive));
    COPCORE_CUDA_CHECK(cudaFree(gpuState.particles[i].queues.leakedTracks));

    COPCORE_CUDA_CHECK(cudaStreamDestroy(gpuState.particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventDestroy(gpuState.particles[i].event));
  }

  COPCORE_CUDA_CHECK(cudaFree(slotManagerInit_dev));

  // Free G4HepEm data
  FreeG4HepEmData(AdeptIntegration::fg4hepem_state->fData);
  delete AdeptIntegration::fg4hepem_state;
  AdeptIntegration::fg4hepem_state = nullptr;
}

void AdeptIntegration::ShowerGPU(int event, TrackBuffer &buffer) // const &buffer)
{
  using TrackData = adeptint::TrackData;
  // Capacity of the different containers aka the maximum number of particles.
  auto &cudaManager                             = vecgeom::cxx::CudaManager::Instance();
  const vecgeom::cuda::VPlacedVolume *world_dev = cudaManager.world_gpu();
  GPUstate &gpuState                            = *static_cast<GPUstate *>(fGPUstate);

  ParticleType &electrons = gpuState.particles[ParticleType::Electron];
  ParticleType &positrons = gpuState.particles[ParticleType::Positron];
  ParticleType &gammas    = gpuState.particles[ParticleType::Gamma];

  // copy buffer of tracks to device
  COPCORE_CUDA_CHECK(cudaMemcpyAsync(gpuState.toDevice_dev, buffer.toDevice.data(),
                                     buffer.toDevice.size() * sizeof(adeptint::TrackData), cudaMemcpyHostToDevice,
                                     gpuState.stream));

  // initialize slot manager
  SlotManager slotManagerInit(kCapacity);
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    COPCORE_CUDA_CHECK(cudaMemcpyAsync(gpuState.particles[i].slotManager, slotManagerInit_dev, sizeof(SlotManager),
                                       cudaMemcpyDeviceToDevice, gpuState.stream));
  }

  if (fDebugLevel > 0) {
    G4cout << std::dec << G4endl << "GPU transporting event " << event << " for CPU thread "
           << G4Threading::G4GetThreadId() << ": " << std::flush;
  }

  // Initialize AdePT tracks using the track buffer copied from CPU
  constexpr int initThreads = 32;
  int initBlocks            = (buffer.toDevice.size() + initThreads - 1) / initThreads;
  Secondaries secondaries   = {
      .electrons = {electrons.tracks, electrons.slotManager, electrons.queues.nextActive},
      .positrons = {positrons.tracks, positrons.slotManager, positrons.queues.nextActive},
      .gammas    = {gammas.tracks, gammas.slotManager, gammas.queues.nextActive},
  };

  InitTracks<<<initBlocks, initThreads, 0, gpuState.stream>>>(
      gpuState.toDevice_dev, buffer.toDevice.size(), buffer.startTrack, event, secondaries, world_dev, fScoring_dev);

  COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

  gpuState.stats->inFlight[ParticleType::Electron] = buffer.nelectrons;
  gpuState.stats->inFlight[ParticleType::Positron] = buffer.npositrons;
  gpuState.stats->inFlight[ParticleType::Gamma]    = buffer.ngammas;

  constexpr int MaxBlocks        = 1024;
  constexpr int TransportThreads = 32;
  int transportBlocks;

  int inFlight          = 0;
  int numLeaked         = 0;
  int loopingNo         = 0;
  int previousElectrons = -1, previousPositrons = -1;
  AllLeaked leakedTracks = {.leakedElectrons = {electrons.tracks, electrons.queues.leakedTracks},
                            .leakedPositrons = {positrons.tracks, positrons.queues.leakedTracks},
                            .leakedGammas    = {gammas.tracks, gammas.queues.leakedTracks}};

  do {

    // *** ELECTRONS ***
    int numElectrons = gpuState.stats->inFlight[ParticleType::Electron];
    if (numElectrons > 0) {
      transportBlocks = (numElectrons + TransportThreads - 1) / TransportThreads;
      transportBlocks = std::min(transportBlocks, MaxBlocks);

      TransportElectrons<AdeptScoring><<<transportBlocks, TransportThreads, 0, electrons.stream>>>(
          electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive,
          electrons.queues.leakedTracks, fScoring_dev);

      COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, electrons.stream));
      COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, electrons.event, 0));
    }

    // *** POSITRONS ***
    int numPositrons = gpuState.stats->inFlight[ParticleType::Positron];
    if (numPositrons > 0) {
      transportBlocks = (numPositrons + TransportThreads - 1) / TransportThreads;
      transportBlocks = std::min(transportBlocks, MaxBlocks);

      TransportPositrons<AdeptScoring><<<transportBlocks, TransportThreads, 0, positrons.stream>>>(
          positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive,
          positrons.queues.leakedTracks, fScoring_dev);

      COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
      COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, positrons.event, 0));
    }

    // *** GAMMAS ***
    int numGammas = gpuState.stats->inFlight[ParticleType::Gamma];
    if (numGammas > 0) {
      transportBlocks = (numGammas + TransportThreads - 1) / TransportThreads;
      transportBlocks = std::min(transportBlocks, MaxBlocks);

      TransportGammas<AdeptScoring><<<transportBlocks, TransportThreads, 0, gammas.stream>>>(
          gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive,
          gammas.queues.leakedTracks, fScoring_dev);

      COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
      COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, gammas.event, 0));
    }

    // *** END OF TRANSPORT ***

    // The events ensure synchronization before finishing this iteration and
    // copying the Stats back to the host.
    AllParticleQueues queues = {{electrons.queues, positrons.queues, gammas.queues}};
    FinishIteration<<<1, 1, 0, gpuState.stream>>>(queues, gpuState.stats_dev);
    COPCORE_CUDA_CHECK(
        cudaMemcpyAsync(gpuState.stats, gpuState.stats_dev, sizeof(Stats), cudaMemcpyDeviceToHost, gpuState.stream));

    // Finally synchronize all kernels.
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

    // Count the number of particles in flight.
    inFlight  = 0;
    numLeaked = 0;
    for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
      inFlight += gpuState.stats->inFlight[i];
      numLeaked += gpuState.stats->leakedTracks[i];
    }

    // Swap the queues for the next iteration.
    electrons.queues.SwapActive();
    positrons.queues.SwapActive();
    gammas.queues.SwapActive();

    // Check if only charged particles are left that are looping.
    numElectrons = gpuState.stats->inFlight[ParticleType::Electron];
    numPositrons = gpuState.stats->inFlight[ParticleType::Positron];
    numGammas    = gpuState.stats->inFlight[ParticleType::Gamma];
    if (numElectrons == previousElectrons && numPositrons == previousPositrons && numGammas == 0) {
      loopingNo++;
    } else {
      previousElectrons = numElectrons;
      previousPositrons = numPositrons;
      loopingNo         = 0;
    }

    // Update the active queues that for next iteration
    secondaries.electrons.SetActiveQueue(electrons.queues.nextActive);
    secondaries.positrons.SetActiveQueue(positrons.queues.nextActive);
    secondaries.gammas.SetActiveQueue(gammas.queues.nextActive);
  } while (inFlight > 0 && loopingNo < 200);

  if (fDebugLevel > 0) {
    G4cout << inFlight << " in flight, " << numLeaked << " leaked\n";
  }
  // Transfer back leaked tracks
  if (numLeaked) {
    fBuffer.numFromDevice = numLeaked;
    // Make sure the size of the allocated track array is large enough
    if (gpuState.fNumFromDevice < numLeaked) {
      gpuState.fNumFromDevice = numLeaked;
      delete[] fBuffer.fromDevice;
      fBuffer.fromDevice = new TrackData[numLeaked];
      COPCORE_CUDA_CHECK(cudaFree(gpuState.fromDevice_dev));
      COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.fromDevice_dev, numLeaked * sizeof(TrackData)));
    }
    // Populate the buffer from sparse memory
    constexpr unsigned int block_size = 256;
    unsigned int grid_size            = (numLeaked + block_size - 1) / block_size;
    FillFromDeviceBuffer<<<grid_size, block_size, 0, gpuState.stream>>>(numLeaked, leakedTracks,
                                                                        gpuState.fromDevice_dev);
    // Copy the buffer from device to host
    COPCORE_CUDA_CHECK(
        cudaMemcpyAsync(gpuState.stats, gpuState.stats_dev, sizeof(Stats), cudaMemcpyDeviceToHost, gpuState.stream));
    COPCORE_CUDA_CHECK(cudaMemcpyAsync(fBuffer.fromDevice, gpuState.fromDevice_dev, numLeaked * sizeof(TrackData),
                                       cudaMemcpyDeviceToHost, gpuState.stream));
    // Sort by energy the tracks coming from device to ensure reproducibility
    fSorted.reserve(numLeaked);
    std::iota(fSorted.begin(), fSorted.begin() + numLeaked, 0); // Fill with 0, 1, ...
    std::sort(fSorted.begin(), fSorted.begin() + numLeaked,
              [&](int i, int j) {return fBuffer.fromDevice[i] < fBuffer.fromDevice[j];});
    fBuffer.fromDevice_sorted.clear();
    fBuffer.fromDevice_sorted.reserve(numLeaked);
    for (auto i = 0; i < numLeaked; ++i)
      fBuffer.fromDevice_sorted.push_back(fBuffer.fromDevice[fSorted[i]]);
  }

  AllParticleQueues queues = {{electrons.queues, positrons.queues, gammas.queues}};
  ClearAllQueues<<<1, 1, 0, gpuState.stream>>>(queues);
  COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

  // Transfer back scoring.
  fScoring->CopyHitsToHost();
  fScoring->fGlobalScoring.numKilled = inFlight;
}
