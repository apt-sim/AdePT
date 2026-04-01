// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AsyncAdePTTransport.hh>

#include <VecGeom/management/BVHManager.h>
#include <VecGeom/management/GeoManager.h>
#ifdef ADEPT_USE_SURF
#include <VecGeom/surfaces/BrepHelper.h>
#endif

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>

#include <cassert>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace async_adept_impl {
void setDeviceLimits(int stackLimit = 0, int heapLimit = 0);
void CopySurfaceModelToGPU();
void InitWDTOnDevice(const adeptint::WDTHostPacked &, adeptint::WDTDeviceBuffers &, unsigned short);
void UploadG4HepEmToGPU(G4HepEmData *hepEmData, G4HepEmParameters *hepEmParameters);
std::thread LaunchGPUWorker(int, int, int, int, AsyncAdePT::TrackBuffer &, AsyncAdePT::GPUstate &,
                            std::vector<std::atomic<AsyncAdePT::EventState>> &, std::condition_variable &, int, int,
                            bool, bool, unsigned short, const double, bool);
std::unique_ptr<AsyncAdePT::GPUstate, AsyncAdePT::GPUstateDeleter> InitializeGPU(
    int trackCapacity, int leakCapacity, int scoringCapacity, int numThreads, AsyncAdePT::TrackBuffer &trackBuffer,
    double CPUCapacityFactor, double CPUCopyFraction, std::string &generalBfieldFile,
    const std::vector<float> &uniformBfieldValues);
void FreeGPU(std::unique_ptr<AsyncAdePT::GPUstate, AsyncAdePT::GPUstateDeleter> &, std::thread &,
             adeptint::WDTDeviceBuffers &);
} // namespace async_adept_impl

namespace AsyncAdePT {

AsyncAdePTTransport::AsyncAdePTTransport(AdePTConfiguration &configuration,
                                         std::unique_ptr<AdePTG4HepEmState> adeptG4HepEmState,
                                         adeptint::VolAuxData *auxData, const adeptint::WDTHostPacked &wdtPacked,
                                         const std::vector<float> &uniformFieldValues)
    : fAdePTSeed{configuration.GetAdePTSeed()}, fNThread{(ushort)configuration.GetNumThreads()},
      fTrackCapacity{(uint)(1024 * 1024 * configuration.GetMillionsOfTrackSlots())},
      fLeakCapacity{(uint)(1024 * 1024 * configuration.GetMillionsOfLeakSlots())},
      fScoringCapacity{(uint)(1024 * 1024 * configuration.GetMillionsOfHitSlots())},
      fDebugLevel{configuration.GetVerbosity()}, fCUDAStackLimit{configuration.GetCUDAStackLimit()},
      fCUDAHeapLimit{configuration.GetCUDAHeapLimit()}, fLastNParticlesOnCPU{configuration.GetLastNParticlesOnCPU()},
      fMaxWDTIter{configuration.GetMaxWDTIter()}, fAdePTG4HepEmState(std::move(adeptG4HepEmState)),
      fEventStates(fNThread), fTrackInAllRegions{configuration.GetTrackInAllRegions()},
      fGPURegionNames{configuration.GetGPURegionNames()}, fCPURegionNames{configuration.GetCPURegionNames()},
      fReturnAllSteps{configuration.GetCallUserSteppingAction()},
      fReturnFirstAndLastStep{configuration.GetCallUserTrackingAction() || configuration.GetCallUserSteppingAction()},
      fBfieldFile{configuration.GetCovfieBfieldFile()}, fCPUCapacityFactor{configuration.GetCPUCapacityFactor()},
      fCPUCopyFraction{configuration.GetHitBufferFlushThreshold()},
      fHitBufferSafetyFactor{configuration.GetHitBufferSafetyFactor()}
{
  if (fNThread > kMaxThreads)
    throw std::invalid_argument("AsyncAdePTTransport limited to " + std::to_string(kMaxThreads) + " threads");

  for (auto &eventState : fEventStates) {
    std::atomic_init(&eventState, EventState::LeakedTracksRetrieved);
  }

  Initialize(auxData, wdtPacked, uniformFieldValues);
}

AsyncAdePTTransport::~AsyncAdePTTransport()
{
  async_adept_impl::FreeGPU(std::ref(fGPUstate), fGPUWorker, fWDTDev);
}

void AsyncAdePTTransport::AddTrack(int pdg, uint64_t trackId, uint64_t parentId, double energy, double x, double y,
                                   double z, double dirx, double diry, double dirz, double globalTime, double localTime,
                                   double properTime, float weight, unsigned short stepCounter, int threadId,
                                   unsigned int eventId, vecgeom::NavigationState &&state)
{
  if (pdg != 11 && pdg != -11 && pdg != 22) {
    std::cerr << __FILE__ << ":" << __LINE__ << ": Only supporting EM tracks. Got pdgID=" << pdg << "\n";
    return;
  }

  TrackDataWithIDs track{pdg,         trackId,
                         parentId,    energy,
                         x,           y,
                         z,           dirx,
                         diry,        dirz,
                         globalTime,  localTime,
                         properTime,  weight,
                         stepCounter, std::move(state),
                         eventId,     static_cast<short>(threadId)};

  {
    auto trackHandle  = fBuffer->createToDeviceSlot();
    trackHandle.track = std::move(track);
  }

  fEventStates[threadId].store(EventState::NewTracksFromG4, std::memory_order_release);
}

bool AsyncAdePTTransport::InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world)
{
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  async_adept_impl::setDeviceLimits(fCUDAStackLimit, fCUDAHeapLimit);

  bool success = true;
#ifdef ADEPT_USE_SURF
#ifdef ADEPT_MIXED_PRECISION
  using SurfData   = vgbrep::SurfData<float>;
  using BrepHelper = vgbrep::BrepHelper<float>;
#else
  using SurfData   = vgbrep::SurfData<double>;
  using BrepHelper = vgbrep::BrepHelper<double>;
#endif
  auto start = std::chrono::steady_clock::now();
  if (!BrepHelper::Instance().Convert()) return false;
  BrepHelper::Instance().PrintSurfData();
  auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
  std::cout << "== Conversion to surface model done in " << elapsed.count() << " [s]\n";
  cudaManager.SynchronizeNavigationTable();
  async_adept_impl::CopySurfaceModelToGPU();
#else
  cudaManager.LoadGeometry(world);
  auto world_dev = cudaManager.Synchronize();
  success        = world_dev != nullptr;
  InitBVH();
#endif
  return success;
}

bool AsyncAdePTTransport::InitializePhysics()
{
  if (!fAdePTG4HepEmState) {
    throw std::runtime_error("AsyncAdePTTransport::InitializePhysics: Missing AdePT-owned G4HepEm state.");
  }

  async_adept_impl::UploadG4HepEmToGPU(fAdePTG4HepEmState->GetData(), fAdePTG4HepEmState->GetParameters());
  return true;
}

void AsyncAdePTTransport::Initialize(adeptint::VolAuxData *auxData, const adeptint::WDTHostPacked &wdtPacked,
                                     const std::vector<float> &uniformFieldValues)
{
  if (vecgeom::GeoManager::Instance().GetRegisteredVolumesCount() == 0)
    throw std::runtime_error("AsyncAdePTTransport::Initialize: Number of geometry volumes is zero.");

  std::cout << "=== AsyncAdePTTransport: initializing geometry and physics\n";
  if (!vecgeom::GeoManager::Instance().IsClosed())
    throw std::runtime_error("AsyncAdePTTransport::Initialize: VecGeom geometry not closed.");

  const vecgeom::cxx::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (!InitializeGeometry(world))
    throw std::runtime_error("AsyncAdePTTransport::Initialize: Cannot initialize geometry on GPU");

  if (!InitializePhysics())
    throw std::runtime_error("AsyncAdePTTransport::Initialize cannot initialize physics on GPU");

  const auto numVolumes   = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  auto &volAuxArray       = adeptint::VolAuxArray::GetInstance();
  volAuxArray.fNumVolumes = numVolumes;
  volAuxArray.fAuxData    = auxData;
  AsyncAdePT::InitVolAuxArray(volAuxArray);

  fHasWDTRegions = !wdtPacked.regions.empty();

  adeptint::WDTDeviceBuffers wdtDev;
  async_adept_impl::InitWDTOnDevice(wdtPacked, wdtDev, fMaxWDTIter);
  fWDTDev = wdtDev;

  std::cout << "\nAllocating " << 4 * 8192 * fNThread << " To-device buffer slots\n";
  std::cout << "\nAllocating " << 2048 * fNThread << " From-device buffer slots\n";
  fBuffer = std::make_unique<TrackBuffer>(4 * 8192 * fNThread, 2048 * fNThread, fNThread);

  assert(fBuffer != nullptr);

  fGPUstate  = async_adept_impl::InitializeGPU(fTrackCapacity, fLeakCapacity, fScoringCapacity, fNThread, *fBuffer,
                                               fCPUCapacityFactor, fCPUCopyFraction, fBfieldFile, uniformFieldValues);
  fGPUWorker = async_adept_impl::LaunchGPUWorker(fTrackCapacity, fLeakCapacity, fScoringCapacity, fNThread, *fBuffer,
                                                 *fGPUstate, fEventStates, fCV_G4Workers, fAdePTSeed, fDebugLevel,
                                                 fReturnAllSteps, fReturnFirstAndLastStep, fLastNParticlesOnCPU,
                                                 fHitBufferSafetyFactor, fHasWDTRegions);
}

void AsyncAdePTTransport::InitBVH()
{
  vecgeom::cxx::BVHManager::Init();
  vecgeom::cxx::BVHManager::DeviceInit();
}

void AsyncAdePTTransport::RequestFlush(int threadId)
{
  assert(static_cast<unsigned int>(threadId) < fBuffer->fromDeviceBuffers.size());
  fEventStates[threadId].store(EventState::G4RequestsFlush, std::memory_order_release);
}

void AsyncAdePTTransport::WaitForFlushProgress()
{
  std::unique_lock lock{fMutex_G4Workers};
  fCV_G4Workers.wait(lock);
}

bool AsyncAdePTTransport::IsDeviceFlushed(int threadId) const
{
  return fEventStates[threadId].load(std::memory_order_acquire) >= EventState::DeviceFlushed;
}

std::vector<TrackDataWithIDs> AsyncAdePTTransport::TakeReturnedTracks(int threadId)
{
  std::vector<TrackDataWithIDs> tracks;
  auto handle = fBuffer->getTracksFromDevice(threadId);
  tracks.swap(handle.tracks);
  return tracks;
}

void AsyncAdePTTransport::MarkLeakedTracksRetrieved(int threadId)
{
  fEventStates[threadId].store(EventState::LeakedTracksRetrieved, std::memory_order_release);
}

} // namespace AsyncAdePT
