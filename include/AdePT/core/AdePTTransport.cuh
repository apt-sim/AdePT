// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRANSPORT_CUH
#define ADEPT_TRANSPORT_CUH

#include <AdePT/core/AdePTScoringTemplate.cuh>
#include <AdePT/core/HostScoringStruct.cuh>
#include <AdePT/core/HostScoringImpl.cuh>
#include <AdePT/core/AdePTTransportStruct.cuh>
#include <AdePT/magneticfield/GeneralMagneticField.h>

#include <AdePT/base/Atomic.h>
#include <AdePT/navigation/AdePTNavigator.h>
#include <AdePT/base/MParray.h>

#include <AdePT/kernels/electrons.cuh>
#include <AdePT/kernels/gammas.cuh>

#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif
#ifdef ADEPT_USE_SURF
#include <VecGeom/surfaces/cuda/BrepCudaManager.h>
#endif

#include <AdePT/copcore/Global.h>
#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/copcore/Ranluxpp.h>

#include <G4HepEmData.hh>
#include <G4HepEmState.hh>
#include <G4HepEmStateInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmParametersInit.hh>
#include <G4HepEmMaterialInit.hh>
#include <G4HepEmElectronInit.hh>
#include <G4HepEmGammaInit.hh>
#include <G4HepEmConfig.hh>

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <numeric>
#include <algorithm>

namespace adept_impl {
inline __constant__ __device__ struct G4HepEmParameters g4HepEmPars;
inline __constant__ __device__ struct G4HepEmData g4HepEmData;

inline __constant__ __device__ adeptint::VolAuxData *gVolAuxData = nullptr;

bool InitializeVolAuxArray(adeptint::VolAuxArray &array)
{
  // Transfer volume auxiliary data
  COPCORE_CUDA_CHECK(cudaMalloc(&array.fAuxData_dev, sizeof(VolAuxData) * array.fNumVolumes));
  COPCORE_CUDA_CHECK(
      cudaMemcpy(array.fAuxData_dev, array.fAuxData, sizeof(VolAuxData) * array.fNumVolumes, cudaMemcpyHostToDevice));
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(gVolAuxData, &array.fAuxData_dev, sizeof(VolAuxData *)));
  return true;
}

void FreeVolAuxArray(adeptint::VolAuxArray &array)
{
  COPCORE_CUDA_CHECK(cudaFree(array.fAuxData_dev));
}

G4HepEmState *InitG4HepEm(G4HepEmConfig *hepEmConfig)
{
  // here we call everything from InitG4HepEmState, as we need to provide the parameters from the G4HepEmConfig and do
  // not want to initialize to the default values
  auto state = new G4HepEmState;

  // Use the config-provided parameters
  state->fParameters = hepEmConfig->GetG4HepEmParameters();

  // Initialize data and fill each subtable using its initialize function
  state->fData = new G4HepEmData;
  InitG4HepEmData(state->fData);
  InitMaterialAndCoupleData(state->fData, state->fParameters);

  // electrons, positrons, gamma
  InitElectronData(state->fData, state->fParameters, true);
  InitElectronData(state->fData, state->fParameters, false);
  InitGammaData(state->fData, state->fParameters);

  G4HepEmMatCutData *cutData = state->fData->fTheMatCutData;
  G4cout << "fNumG4MatCuts = " << cutData->fNumG4MatCuts << ", fNumMatCutData = " << cutData->fNumMatCutData << G4endl;

  // Copy to GPU.
  CopyG4HepEmDataToGPU(state->fData);
  CopyG4HepEmParametersToGPU(state->fParameters);

  // Create G4HepEmParameters with the device pointer
  G4HepEmParameters parametersOnDevice        = *state->fParameters;
  parametersOnDevice.fParametersPerRegion     = state->fParameters->fParametersPerRegion_gpu;
  parametersOnDevice.fParametersPerRegion_gpu = nullptr;

  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmPars, &parametersOnDevice, sizeof(G4HepEmParameters)));

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
                           Secondaries secondaries, const vecgeom::VPlacedVolume *world, AdeptScoring *userScoring,
                           VolAuxData const *auxDataArray)
{
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

    Track &track   = trackmgr->NextTrack();
    track.parentId = trackinfo[i].parentId;

    track.rngState.SetSeed(1234567 * event + startTrack + i);
    track.eKin         = trackinfo[i].eKin;
    track.vertexEkin   = trackinfo[i].vertexEkin;
    track.numIALeft[0] = -1.0;
    track.numIALeft[1] = -1.0;
    track.numIALeft[2] = -1.0;
    track.numIALeft[3] = -1.0;

    track.initialRange       = 1.0e+21;
    track.dynamicRangeFactor = 0.04;
    track.tlimitMin          = 1.0E-7;

    track.pos                     = {trackinfo[i].position[0], trackinfo[i].position[1], trackinfo[i].position[2]};
    track.dir                     = {trackinfo[i].direction[0], trackinfo[i].direction[1], trackinfo[i].direction[2]};
    track.vertexPosition          = {trackinfo[i].vertexPosition[0], trackinfo[i].vertexPosition[1],
                                     trackinfo[i].vertexPosition[2]};
    track.vertexMomentumDirection = {trackinfo[i].vertexMomentumDirection[0], trackinfo[i].vertexMomentumDirection[1],
                                     trackinfo[i].vertexMomentumDirection[2]};

    track.globalTime = trackinfo[i].globalTime;
    track.localTime  = trackinfo[i].localTime;
    track.properTime = trackinfo[i].properTime;

    track.weight = trackinfo[i].weight;

    track.originNavState.Clear();
    track.originNavState = trackinfo[i].originNavState;

    // setting up the NavState
    track.navState.Clear();
    track.navState = trackinfo[i].navState;
    // nextState is initialized as needed.

#ifndef ADEPT_USE_SURF
    int lvolID = track.navState.Top()->GetLogicalVolume()->id();
#else
    int lvolID = track.navState.GetLogicalId();
#endif
    assert(auxDataArray[lvolID].fGPUregion);
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
__global__ void FinishIteration(AllTrackManagers all, Stats *stats, AdeptScoring *scoring)
{
  // Update track manager stats
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    all.trackmgr[i]->refresh_stats();
    stats->mgr_stats[i]    = all.trackmgr[i]->fStats;
    stats->leakedTracks[i] = all.leakedTracks[i]->size();
  }
  // Update hit buffer stats
  adept_scoring::EndOfIterationGPU(scoring);
  stats->scoring_stats = *scoring->fStats_dev;
}

// Clear device leaked queues
__global__ void ClearLeakedQueues(LeakedTracks all)
{
  all.leakedElectrons->clear();
  all.leakedPositrons->clear();
  all.leakedGammas->clear();
}

template <typename FieldType>
bool InitializeBField(FieldType &magneticField)
{

  // Allocate and copy the FieldType instance (not the field array itself), and set the global device pointer
  FieldType *dMagneticFieldInstance = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&dMagneticFieldInstance, sizeof(FieldType)));
  COPCORE_CUDA_CHECK(cudaMemcpy(dMagneticFieldInstance, &magneticField, sizeof(FieldType), cudaMemcpyHostToDevice));
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(gMagneticField, &dMagneticFieldInstance, sizeof(FieldType *)));

  return true;
}

template <typename FieldType>
void FreeBField()
{
  FieldType *dMagneticFieldInstance = nullptr;

  // Retrieve the global device pointer from the symbol
  COPCORE_CUDA_CHECK(cudaMemcpyFromSymbol(&dMagneticFieldInstance, gMagneticField, sizeof(FieldType *)));

  if (dMagneticFieldInstance) {
    // Free the device memory and reset global device pointer
    COPCORE_CUDA_CHECK(cudaFree(dMagneticFieldInstance));
    FieldType *nullPtr = nullptr;
    COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(gMagneticField, &nullPtr, sizeof(FieldType *)));
  }
}

void PrepareLeakedBuffers(int numLeaked, adeptint::TrackBuffer &buffer, GPUstate &gpuState)
{
  // Make sure the size of the allocated track array is large enough
  using TrackData = adeptint::TrackData;
  if (buffer.buffSize < numLeaked) {
    if (buffer.buffSize) {
      delete[] buffer.fromDeviceBuff;
      COPCORE_CUDA_CHECK(cudaFree(gpuState.fromDevice_dev));
    }
    buffer.buffSize = numLeaked;
    buffer.fromDevice.reserve(numLeaked);
    buffer.fromDeviceBuff = new TrackData[numLeaked];
    COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.fromDevice_dev, numLeaked * sizeof(TrackData)));
  }
}

void CopySurfaceModelToGPU()
{
// Copy surface data to GPU
#ifdef ADEPT_USE_SURF
#ifdef ADEPT_USE_SURF_SINGLE
  using SurfData        = vgbrep::SurfData<float>;
  using BrepCudaManager = vgbrep::BrepCudaManager<float>;
#else
  using SurfData        = vgbrep::SurfData<double>;
  using BrepCudaManager = vgbrep::BrepCudaManager<double>;
#endif
  BrepCudaManager::Instance().TransferSurfData(SurfData::Instance());
  printf("== Surface data transferred to GPU\n");
#endif
}

GPUstate *InitializeGPU(adeptint::TrackBuffer &buffer, int capacity, int maxbatch)
{
  using TrackData   = adeptint::TrackData;
  auto gpuState_ptr = new GPUstate;
  auto &gpuState    = *gpuState_ptr;
  // Allocate track managers, streams and synchronization events.
  const size_t kQueueSize = MParrayTracks::SizeOfInstance(capacity);
  // Create a stream to synchronize kernels of all particle types.
  COPCORE_CUDA_CHECK(cudaStreamCreate(&gpuState.stream));

  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    gpuState.allmgr_h.trackmgr[i]  = new adept::TrackManager<Track>(capacity);
    gpuState.allmgr_d.trackmgr[i]  = gpuState.allmgr_h.trackmgr[i]->ConstructOnDevice();
    gpuState.particles[i].trackmgr = gpuState.allmgr_d.trackmgr[i];
    COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.allmgr_d.leakedTracks[i], kQueueSize));
    gpuState.particles[i].leakedTracks = gpuState.allmgr_d.leakedTracks[i];

    COPCORE_CUDA_CHECK(cudaStreamCreate(&gpuState.particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventCreate(&gpuState.particles[i].event));
  }

  InitLeakedQueues<<<1, 1, 0, gpuState.stream>>>(gpuState.allmgr_d, kQueueSize);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  // initialize statistics
  COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.stats_dev, sizeof(Stats)));
  COPCORE_CUDA_CHECK(cudaMallocHost(&gpuState.stats, sizeof(Stats)));

  // initialize buffers of tracks on device
  COPCORE_CUDA_CHECK(cudaMalloc(&gpuState.toDevice_dev, maxbatch * sizeof(TrackData)));
  PrepareLeakedBuffers(1000, buffer, gpuState);
  return gpuState_ptr;
}

AdeptScoring *InitializeScoringGPU(AdeptScoring *scoring)
{
  // Initialize Scoring
  return adept_scoring::InitializeOnGPU(scoring);
}

void FreeGPU(GPUstate &gpuState, G4HepEmState *g4hepem_state)
{
  // Free resources.
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
  FreeG4HepEmData(g4hepem_state->fData);
  FreeG4HepEmParametersOnGPU(g4hepem_state->fParameters);
  delete g4hepem_state;

// Free magnetic field
#ifdef ADEPT_USE_EXT_BFIELD
  FreeBField<GeneralMagneticField>();
#else
  FreeBField<UniformMagneticField>();
#endif
}

template <typename IntegrationLayer>
void ShowerGPU(IntegrationLayer &integration, int event, adeptint::TrackBuffer &buffer, GPUstate &gpuState,
               AdeptScoring *scoring, AdeptScoring *scoring_dev, int debugLevel)
{
  using TrackData   = adeptint::TrackData;
  using VolAuxArray = adeptint::VolAuxArray;
  // Capacity of the different containers aka the maximum number of particles.
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();

  static std::chrono::steady_clock::time_point startTime;
  if (debugLevel >= 2) {
    // Initialize startTime only on the first call
    static bool isInitialized = false;
    if (!isInitialized) {
      startTime     = std::chrono::steady_clock::now();
      isInitialized = true;
    }
  }
  const vecgeom::cuda::VPlacedVolume *world_dev = cudaManager.world_gpu();
  Secondaries secondaries{gpuState.allmgr_d.trackmgr[0], gpuState.allmgr_d.trackmgr[1], gpuState.allmgr_d.trackmgr[2]};
  ParticleType &electrons = gpuState.particles[ParticleType::Electron];
  ParticleType &positrons = gpuState.particles[ParticleType::Positron];
  ParticleType &gammas    = gpuState.particles[ParticleType::Gamma];

  // copy buffer of tracks to device
  COPCORE_CUDA_CHECK(cudaMemcpyAsync(gpuState.toDevice_dev, buffer.toDevice.data(),
                                     buffer.toDevice.size() * sizeof(adeptint::TrackData), cudaMemcpyHostToDevice,
                                     gpuState.stream));

#ifndef DEBUG_SINGLE_THREAD
  constexpr int initThreads = 32;
#else
  constexpr int initThreads = 1;
#endif
  int initBlocks = (buffer.toDevice.size() + initThreads - 1) / initThreads;

  // Initialize AdePT tracks using the track buffer copied from CPU
  InitTracks<<<initBlocks, initThreads, 0, gpuState.stream>>>(gpuState.toDevice_dev, buffer.toDevice.size(),
                                                              buffer.startTrack, event, secondaries, world_dev,
                                                              scoring_dev, VolAuxArray::GetInstance().fAuxData_dev);

  COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));

  gpuState.allmgr_h.trackmgr[ParticleType::Electron]->fStats.fInFlight = buffer.nelectrons;
  gpuState.allmgr_h.trackmgr[ParticleType::Positron]->fStats.fInFlight = buffer.npositrons;
  gpuState.allmgr_h.trackmgr[ParticleType::Gamma]->fStats.fInFlight    = buffer.ngammas;

  constexpr float compactThreshold = 0.9;
#ifdef DEBUG_SINGLE_THREAD
  constexpr int TransportThreads = 1;
#else
  constexpr int MaxBlocks        = 1024;
  constexpr int TransportThreads = 32;
#endif
  int transportBlocks   = 1;
  int inFlight          = 0;
  int killed            = 0;
  int numLeaked         = 0;
  int num_compact       = 0;
  int loopingNo         = 0;
  int previousElectrons = -1, previousPositrons = -1, previousGammas = -1;
  LeakedTracks leakedTracks = {.leakedElectrons = electrons.leakedTracks,
                               .leakedPositrons = positrons.leakedTracks,
                               .leakedGammas    = gammas.leakedTracks};

  auto copyLeakedTracksFromGPU = [&](int numLeaked) {
    PrepareLeakedBuffers(numLeaked, buffer, gpuState);
    // Populate the buffer from sparse memory
    constexpr unsigned int block_size = 256;
    unsigned int grid_size            = (numLeaked + block_size - 1) / block_size;
    FillFromDeviceBuffer<<<grid_size, block_size, 0, gpuState.stream>>>(numLeaked, leakedTracks,
                                                                        gpuState.fromDevice_dev);
    // Copy the buffer from device to host
    COPCORE_CUDA_CHECK(cudaMemcpyAsync(buffer.fromDeviceBuff, gpuState.fromDevice_dev, numLeaked * sizeof(TrackData),
                                       cudaMemcpyDeviceToHost, gpuState.stream));
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(gpuState.stream));
    buffer.fromDevice.insert(buffer.fromDevice.end(), &buffer.fromDeviceBuff[0], &buffer.fromDeviceBuff[numLeaked]);
  };

  int niter = 0;
  do {

    // *** ELECTRONS ***
    int numElectrons = gpuState.allmgr_h.trackmgr[ParticleType::Electron]->fStats.fInFlight;
    if (numElectrons > 0) {
#ifndef DEBUG_SINGLE_THREAD
      transportBlocks = (numElectrons + TransportThreads - 1) / TransportThreads;
      transportBlocks = std::min(transportBlocks, MaxBlocks);
#endif
      TransportElectrons<AdeptScoring><<<transportBlocks, TransportThreads, 0, electrons.stream>>>(
          electrons.trackmgr, secondaries, electrons.leakedTracks, scoring_dev,
          VolAuxArray::GetInstance().fAuxData_dev);
      COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, electrons.stream));
      COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, electrons.event, 0));
    }

    // *** POSITRONS ***
    int numPositrons = gpuState.allmgr_h.trackmgr[ParticleType::Positron]->fStats.fInFlight;
    if (numPositrons > 0) {
#ifndef DEBUG_SINGLE_THREAD
      transportBlocks = (numPositrons + TransportThreads - 1) / TransportThreads;
      transportBlocks = std::min(transportBlocks, MaxBlocks);
#endif
      TransportPositrons<AdeptScoring><<<transportBlocks, TransportThreads, 0, positrons.stream>>>(
          positrons.trackmgr, secondaries, positrons.leakedTracks, scoring_dev,
          VolAuxArray::GetInstance().fAuxData_dev);
      COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
      COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, positrons.event, 0));
    }

    // *** GAMMAS ***
    int numGammas = gpuState.allmgr_h.trackmgr[ParticleType::Gamma]->fStats.fInFlight;
    if (numGammas > 0) {
#ifndef DEBUG_SINGLE_THREAD
      transportBlocks = (numGammas + TransportThreads - 1) / TransportThreads;
      transportBlocks = std::min(transportBlocks, MaxBlocks);
#endif
      TransportGammas<AdeptScoring><<<transportBlocks, TransportThreads, 0, gammas.stream>>>(
          gammas.trackmgr, secondaries, gammas.leakedTracks, scoring_dev, VolAuxArray::GetInstance().fAuxData_dev);
      COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
      COPCORE_CUDA_CHECK(cudaStreamWaitEvent(gpuState.stream, gammas.event, 0));
    }

    // *** END OF TRANSPORT ***

    // The events ensure synchronization before finishing this iteration and
    // copying the Stats back to the host.
    FinishIteration<<<1, 1, 0, gpuState.stream>>>(gpuState.allmgr_d, gpuState.stats_dev, scoring_dev);
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

    scoring->fStats = gpuState.stats->scoring_stats;
    adept_scoring::EndOfIteration<IntegrationLayer>(*scoring, scoring_dev, gpuState.stream, integration);

    // Check if only charged particles are left that are looping.
    numElectrons = gpuState.allmgr_h.trackmgr[ParticleType::Electron]->fStats.fInFlight;
    numPositrons = gpuState.allmgr_h.trackmgr[ParticleType::Positron]->fStats.fInFlight;
    numGammas    = gpuState.allmgr_h.trackmgr[ParticleType::Gamma]->fStats.fInFlight;
    if (debugLevel > 1) {
      auto elapsedTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
      printf("Time elapsed: %.6fs eventId %d iter %d: elec %d, pos %d, gam %d, leak %d\n", elapsedTime, event, niter++,
             numElectrons, numPositrons, numGammas, numLeaked);
    }
    if (numElectrons == previousElectrons && numPositrons == previousPositrons && numGammas == previousGammas) {
      loopingNo++;
      if (loopingNo == 200)
        printf("Killing %d electrons %d positrons %d gammas due to looping detection %d\n", numElectrons, numPositrons,
               numGammas, loopingNo);
    } else {
      previousElectrons = numElectrons;
      previousPositrons = numPositrons;
      previousGammas    = numGammas;
      loopingNo         = 0;
    }

  } while (inFlight > 0 && loopingNo < 200);

  if (debugLevel > 0) {
    std::cout << inFlight << " in flight, " << numLeaked << " leaked, " << num_compact << " compacted\n";
  }

  // Transfer the leaked tracks from GPU
  if (numLeaked) {
    // for(int i=0; i<numLeaked; i++)
    //   printf("%d\n", electrons.leakedTracks[i].id);
    copyLeakedTracksFromGPU(numLeaked);
    // Sort by energy the tracks coming from device to ensure reproducibility
    std::sort(buffer.fromDevice.begin(), buffer.fromDevice.end());
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

  adept_scoring::EndOfTransport<IntegrationLayer>(*scoring, scoring_dev, gpuState.stream, integration);
}

// explicit instantiation
template bool InitializeBField<GeneralMagneticField>(GeneralMagneticField &);
template bool InitializeBField<UniformMagneticField>(UniformMagneticField &);

} // namespace adept_impl

#endif
