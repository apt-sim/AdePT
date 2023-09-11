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
#include <numeric>
#include <algorithm>

#include "electrons.cuh"
#include "gammas.cuh"

__constant__ __device__ struct G4HepEmParameters g4HepEmPars;
__constant__ __device__ struct G4HepEmData g4HepEmData;

__constant__ __device__ adeptint::VolAuxData *gVolAuxData = nullptr;
__constant__ __device__ double BzFieldValue               = 0;

G4HepEmState *AdeptIntegration::fg4hepem_state{nullptr};
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

// Kernel function to initialize tracks comming from a Geant4 buffer
__global__ void InitTracks(adeptint::TrackData *trackinfo, int ntracks, int startTrack, int event,
                           Secondaries secondaries, const vecgeom::VPlacedVolume *world, AdeptScoring *userScoring)
{
  constexpr double tolerance = 10. * vecgeom::kTolerance;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ntracks; i += blockDim.x * gridDim.x) {
    adept::TrackManager<Track> *trackmgr = nullptr;
    // These tracks come from Geant4, do not count them here
    switch (trackinfo[i].pdg) {
    case 11:
      trackmgr = secondaries.electrons;
      break;
    case -11:
      trackmgr = secondaries.positrons;
      break;
    case 22:
      trackmgr = secondaries.gammas;
    };
    assert(trackmgr != nullptr && "Unsupported pdg type");

    Track &track = trackmgr->NextTrack();
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

// Kernel to initialize the set of leaked queues per particle type.
__global__ void InitLeakedQueues(AllTrackManagers allMgr, size_t Capacity)
{
  for (int i = 0; i < ParticleType::NumParticleTypes; i++)
    MParrayTracks::MakeInstanceAt(Capacity, allMgr.leakedTracks[i]);
}

// Copy particles leaked from the GPU region into a compact buffer
__global__ void FillFromDeviceBuffer(int numLeaked, LeakedTracks all, adeptint::TrackData *fromDevice)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numLeaked) return;
  int numElectrons = all.leakedElectrons->size();
  int numPositrons = all.leakedPositrons->size();
  int numGammas    = all.leakedGammas->size();
  assert(numLeaked == numElectrons + numPositrons + numGammas);

  if (i < numElectrons) {
    fromDevice[i] = (*all.leakedElectrons)[i];
  } else if (i < numElectrons + numPositrons) {
    fromDevice[i] = (*all.leakedPositrons)[i - numElectrons];
  } else {
    fromDevice[i] = (*all.leakedGammas)[i - numElectrons - numPositrons];
  }
}

// Finish iteration: refresh track managers and fill statistics.
__global__ void FinishIteration(AllTrackManagers all, Stats *stats)
{
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    all.trackmgr[i]->refresh_stats();
    stats->mgr_stats[i] = all.trackmgr[i]->fStats;
    stats->leakedTracks[i] = all.leakedTracks[i]->size();
  }
}

// Clear device leaked queues
__global__ void ClearLeakedQueues(LeakedTracks all)
{
  all.leakedElectrons->clear();
  all.leakedPositrons->clear();
  all.leakedGammas->clear();
}

bool AdeptIntegration::InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world)
{
  COPCORE_CUDA_CHECK(vecgeom::cxx::CudaDeviceSetStackLimit(8192));
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

void AdeptIntegration::PrepareLeakedBuffers(int numLeaked)
{
  // Make sure the size of the allocated track array is large enough
  using TrackData    = adeptint::TrackData;
  GPUstate &gpuState = *static_cast<GPUstate *>(fGPUstate);
  if (fBuffer.buffSize < numLeaked) {
    if (fBuffer.buffSize) {
      delete[] fBuffer.fromDeviceBuff;
      COPCORE_CUDA_CHECK(cudaFree(gpuState.fromDevice_dev));
    }
    fBuffer.buffSize = numLeaked;
    fBuffer.fromDevice.reserve(numLeaked);
    fBuffer.fromDeviceBuff = new TrackData[numLeaked];
    COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.fromDevice_dev, numLeaked * sizeof(TrackData)));
  }
}

void AdeptIntegration::InitializeGPU()
{
  using TrackData    = adeptint::TrackData;
  fGPUstate          = new GPUstate;
  GPUstate &gpuState = *fGPUstate;

  // Allocate track managers, streams and synchronizaion events.
  const size_t QueueSize = MParrayTracks::SizeOfInstance(kCapacity);
  // Create a stream to synchronize kernels of all particle types.
  COPCORE_CUDA_CHECK(cudaStreamCreate(&gpuState.stream));

  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    gpuState.allmgr_h.trackmgr[i]  = new adept::TrackManager<Track>(kCapacity);
    gpuState.allmgr_d.trackmgr[i]  = gpuState.allmgr_h.trackmgr[i]->ConstructOnDevice();
    gpuState.particles[i].trackmgr = gpuState.allmgr_d.trackmgr[i];
    COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.allmgr_d.leakedTracks[i], QueueSize));
    gpuState.particles[i].leakedTracks = gpuState.allmgr_d.leakedTracks[i];

    COPCORE_CUDA_CHECK(cudaStreamCreate(&gpuState.particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventCreate(&gpuState.particles[i].event));
  }
  InitLeakedQueues<<<1, 1, 0, gpuState.stream>>>(gpuState.allmgr_d, QueueSize);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  // initialize statistics
  COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.stats_dev, sizeof(Stats)));
  COPCORE_CUDA_CHECK(cudaMallocHost(&gpuState.stats, sizeof(Stats)));

  // initialize buffers of tracks on device
  COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.toDevice_dev, fMaxBatch * sizeof(TrackData)));
  PrepareLeakedBuffers(1000);
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
    gpuState.allmgr_h.trackmgr[i]->FreeFromDevice();
    delete gpuState.allmgr_h.trackmgr[i];
    COPCORE_CUDA_CHECK(cudaFree(gpuState.particles[i].leakedTracks));

    COPCORE_CUDA_CHECK(cudaStreamDestroy(gpuState.particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventDestroy(gpuState.particles[i].event));
  }

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

  Secondaries secondaries{gpuState.allmgr_d.trackmgr[0], gpuState.allmgr_d.trackmgr[1], gpuState.allmgr_d.trackmgr[2]};

  ParticleType &electrons = gpuState.particles[ParticleType::Electron];
  ParticleType &positrons = gpuState.particles[ParticleType::Positron];
  ParticleType &gammas    = gpuState.particles[ParticleType::Gamma];

  // copy buffer of tracks to device
  COPCORE_CUDA_CHECK(cudaMemcpyAsync(gpuState.toDevice_dev, buffer.toDevice.data(),
                                     buffer.toDevice.size() * sizeof(adeptint::TrackData), cudaMemcpyHostToDevice,
                                     gpuState.stream));

  if (fDebugLevel > 0) {
    G4cout << std::dec << G4endl << "GPU transporting event " << event << " for CPU thread "
           << G4Threading::G4GetThreadId() << ": " << std::flush;
  }

  // Initialize AdePT tracks using the track buffer copied from CPU
  constexpr int initThreads = 32;
  int initBlocks            = (buffer.toDevice.size() + initThreads - 1) / initThreads;

  InitTracks<<<initBlocks, initThreads, 0, gpuState.stream>>>(
      gpuState.toDevice_dev, buffer.toDevice.size(), buffer.startTrack, event, secondaries, world_dev, fScoring_dev);

  COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

  gpuState.allmgr_h.trackmgr[ParticleType::Electron]->fStats.fInFlight = buffer.nelectrons;
  gpuState.allmgr_h.trackmgr[ParticleType::Positron]->fStats.fInFlight = buffer.npositrons;
  gpuState.allmgr_h.trackmgr[ParticleType::Gamma]->fStats.fInFlight    = buffer.ngammas;

  constexpr float compactThreshold = 0.9;
  constexpr int MaxBlocks          = 1024;
  constexpr int TransportThreads   = 32;
  int transportBlocks;

  int inFlight          = 0;
  int killed            = 0;
  int numLeaked         = 0;
  int num_compact       = 0;
  int loopingNo         = 0;
  int previousElectrons = -1, previousPositrons = -1;
  LeakedTracks leakedTracks = {.leakedElectrons = electrons.leakedTracks,
                            .leakedPositrons = positrons.leakedTracks,
                            .leakedGammas    = gammas.leakedTracks};

  auto copyLeakedTracksFromGPU = [&](int numLeaked)
  {
    PrepareLeakedBuffers(numLeaked);
    // Populate the buffer from sparse memory
    constexpr unsigned int block_size = 256;
    unsigned int grid_size            = (numLeaked + block_size - 1) / block_size;
    FillFromDeviceBuffer<<<grid_size, block_size, 0, gpuState.stream>>>(numLeaked, leakedTracks,
                                                                        gpuState.fromDevice_dev);
    // Copy the buffer from device to host
    COPCORE_CUDA_CHECK(cudaMemcpyAsync(fBuffer.fromDeviceBuff, gpuState.fromDevice_dev, numLeaked * sizeof(TrackData),
                                        cudaMemcpyDeviceToHost, gpuState.stream));
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));
    fBuffer.fromDevice.insert(fBuffer.fromDevice.end(), &fBuffer.fromDeviceBuff[0], &fBuffer.fromDeviceBuff[numLeaked]);
  };
  
  int niter = 0;
  do {

    // *** ELECTRONS ***
    int numElectrons = gpuState.allmgr_h.trackmgr[ParticleType::Electron]->fStats.fInFlight;
    if (numElectrons > 0) {
      transportBlocks = (numElectrons + TransportThreads - 1) / TransportThreads;
      transportBlocks = std::min(transportBlocks, MaxBlocks);

      TransportElectrons<AdeptScoring><<<transportBlocks, TransportThreads, 0, electrons.stream>>>(
          electrons.trackmgr, secondaries, electrons.leakedTracks, fScoring_dev);

      COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, electrons.stream));
      COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, electrons.event, 0));
    }

    // *** POSITRONS ***
    int numPositrons = gpuState.allmgr_h.trackmgr[ParticleType::Positron]->fStats.fInFlight;
    if (numPositrons > 0) {
      transportBlocks = (numPositrons + TransportThreads - 1) / TransportThreads;
      transportBlocks = std::min(transportBlocks, MaxBlocks);

      TransportPositrons<AdeptScoring><<<transportBlocks, TransportThreads, 0, positrons.stream>>>(
          positrons.trackmgr, secondaries, positrons.leakedTracks, fScoring_dev);

      COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
      COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, positrons.event, 0));
    }

    // *** GAMMAS ***
    int numGammas = gpuState.allmgr_h.trackmgr[ParticleType::Gamma]->fStats.fInFlight;
    if (numGammas > 0) {
      transportBlocks = (numGammas + TransportThreads - 1) / TransportThreads;
      transportBlocks = std::min(transportBlocks, MaxBlocks);

      TransportGammas<AdeptScoring><<<transportBlocks, TransportThreads, 0, gammas.stream>>>(
          gammas.trackmgr, secondaries, gammas.leakedTracks, fScoring_dev);

      COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
      COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, gammas.event, 0));
    }

    // *** END OF TRANSPORT ***

    // The events ensure synchronization before finishing this iteration and
    // copying the Stats back to the host.
    FinishIteration<<<1, 1, 0, gpuState.stream>>>(gpuState.allmgr_d, gpuState.stats_dev);
    COPCORE_CUDA_CHECK(
        cudaMemcpyAsync(gpuState.stats, gpuState.stats_dev, sizeof(Stats), cudaMemcpyDeviceToHost, gpuState.stream));

    // Finally synchronize all kernels.
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

    // Count the number of particles in flight.
    inFlight  = 0;
    numLeaked = 0;
    for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
      // Update stats for host track manager objects
      gpuState.allmgr_h.trackmgr[i]->fStats = gpuState.stats->mgr_stats[i];
      inFlight += gpuState.stats->mgr_stats[i].fInFlight;
      numLeaked += gpuState.stats->leakedTracks[i];
      // Compact the particle track buffer if needed
      auto compacted = gpuState.allmgr_h.trackmgr[i]->SwapAndCompact(compactThreshold, gpuState.particles[i].stream);
      if (compacted) num_compact++;
    }

    // Check if only charged particles are left that are looping.
    numElectrons = gpuState.allmgr_h.trackmgr[ParticleType::Electron]->fStats.fInFlight;
    numPositrons = gpuState.allmgr_h.trackmgr[ParticleType::Positron]->fStats.fInFlight;
    numGammas    = gpuState.allmgr_h.trackmgr[ParticleType::Gamma]->fStats.fInFlight;
    if (fDebugLevel > 1) {
      printf("iter %d: elec %d, pos %d, gam %d, leak %d\n", niter++, numElectrons, numPositrons, numGammas, numLeaked);
    }
    if (numElectrons == previousElectrons && numPositrons == previousPositrons && numGammas == 0) {
      loopingNo++;
    } else {
      previousElectrons = numElectrons;
      previousPositrons = numPositrons;
      loopingNo         = 0;
    }

  } while (inFlight > 0 && loopingNo < 200);

  if (fDebugLevel > 0) {
    G4cout << inFlight << " in flight, " << numLeaked << " leaked, " << num_compact << " compacted\n";
  }

  // Transfer the leaked tracks from GPU
  if (numLeaked) {
    copyLeakedTracksFromGPU(numLeaked);
    // Sort by energy the tracks coming from device to ensure reproducibility
    std::sort(fBuffer.fromDevice.begin(), fBuffer.fromDevice.end());
  }

  if (inFlight > 0) {
    killed += inFlight;
    for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
      int inFlightParticles = gpuState.allmgr_h.trackmgr[i]->fStats.fInFlight;
      if (inFlightParticles == 0) {
        continue;
      }

      gpuState.allmgr_h.trackmgr[i]->Clear(gpuState.particles[i].stream);
    }
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));
  }

  ClearLeakedQueues<<<1, 1, 0, gpuState.stream>>>(leakedTracks);
  COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

  // Transfer back scoring.
  fScoring->CopyHitsToHost();
  fScoring->fGlobalScoring.numKilled = inFlight;
}
