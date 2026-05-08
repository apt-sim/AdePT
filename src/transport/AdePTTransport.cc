// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/transport/AdePTTransport.hh>
#include <AdePT/transport/queues/TrackBuffer.hh>

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

namespace adept::transport::detail {
void setDeviceLimits(int stackLimit = 0, int heapLimit = 0);
void CopySurfaceModelToGPU();
void InitWDTOnDevice(const adeptint::WDTHostPacked &, adeptint::WDTDeviceBuffers &, unsigned short);
void UploadG4HepEmToGPU(G4HepEmData *hepEmData, G4HepEmParameters *hepEmParameters);
std::thread LaunchGPUWorker(int, int, int, adept::transport::TrackBuffer &, adept::transport::GPUstate &,
                            std::vector<std::atomic<adept::transport::EventState>> &, std::condition_variable &, int,
                            int, bool, bool, unsigned short, const double, bool);
std::unique_ptr<adept::transport::GPUstate, adept::transport::GPUstateDeleter> InitializeGPU(
    int trackCapacity, int stepCapacity, int numThreads, adept::transport::TrackBuffer &trackBuffer,
    double CPUCapacityFactor, double CPUCopyFraction, std::string &generalBfieldFile,
    const std::vector<float> &uniformBfieldValues);
void FreeGPU(std::unique_ptr<adept::transport::GPUstate, adept::transport::GPUstateDeleter> &, std::thread &,
             adeptint::WDTDeviceBuffers &);
} // namespace adept::transport::detail

namespace adept::transport {

AdePTTransport::AdePTTransport(const AdePTTransportConfig &configuration,
                               std::unique_ptr<AdePTG4HepEmState> adeptG4HepEmState, adeptint::VolAuxData *auxData,
                               const adeptint::WDTHostPacked &wdtPacked, const std::vector<float> &uniformFieldValues)
    : fAdePTSeed{configuration.adeptSeed}, fNThread{configuration.numThreads},
      fTrackCapacity{configuration.trackCapacity}, fStepCapacity{configuration.stepCapacity},
      fDebugLevel{configuration.debugLevel}, fCUDAStackLimit{configuration.cudaStackLimit},
      fCUDAHeapLimit{configuration.cudaHeapLimit}, fLastNParticlesOnCPU{configuration.lastNParticlesOnCPU},
      fMaxWDTIter{configuration.maxWDTIter}, fAdePTG4HepEmState(std::move(adeptG4HepEmState)), fEventStates(fNThread),
      fReturnAllSteps{configuration.returnAllSteps}, fReturnFirstAndLastStep{configuration.returnFirstAndLastStep},
      fBfieldFile{configuration.bfieldFile}, fCPUCapacityFactor{configuration.cpuCapacityFactor},
      fCPUCopyFraction{configuration.cpuCopyFraction}, fStepBufferSafetyFactor{configuration.stepBufferSafetyFactor}
{
  if (fNThread > kMaxThreads)
    throw std::invalid_argument("AdePTTransport limited to " + std::to_string(kMaxThreads) + " threads");

  for (auto &eventState : fEventStates) {
    std::atomic_init(&eventState, EventState::DeviceFlushed);
  }

  Initialize(auxData, wdtPacked, uniformFieldValues);
}

AdePTTransport::~AdePTTransport()
{
  adept::transport::detail::FreeGPU(std::ref(fGPUstate), fGPUWorker, fWDTDev);
}

void AdePTTransport::AddTrack(int pdg, uint64_t trackId, uint64_t parentId, double energy, double x, double y, double z,
                              double dirx, double diry, double dirz, double globalTime, double localTime,
                              double properTime, float weight, unsigned short stepCounter, int threadId,
                              unsigned int eventId, vecgeom::NavigationState &&state)
{
  if (pdg != 11 && pdg != -11 && pdg != 22) {
    std::cerr << __FILE__ << ":" << __LINE__ << ": Only supporting EM tracks. Got pdgID=" << pdg << "\n";
    return;
  }

  adeptint::TrackData track{pdg,         trackId,
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

bool AdePTTransport::InitializeGeometry(const vecgeom::cxx::VPlacedVolume *world)
{
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  adept::transport::detail::setDeviceLimits(fCUDAStackLimit, fCUDAHeapLimit);

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
  adept::transport::detail::CopySurfaceModelToGPU();
#else
  cudaManager.LoadGeometry(world);
  auto world_dev = cudaManager.Synchronize();
  success        = world_dev != nullptr;
  InitBVH();
#endif
  return success;
}

bool AdePTTransport::InitializePhysics()
{
  if (!fAdePTG4HepEmState) {
    throw std::runtime_error("AdePTTransport::InitializePhysics: Missing AdePT-owned G4HepEm state.");
  }

  adept::transport::detail::UploadG4HepEmToGPU(fAdePTG4HepEmState->GetData(), fAdePTG4HepEmState->GetParameters());
  return true;
}

void AdePTTransport::Initialize(adeptint::VolAuxData *auxData, const adeptint::WDTHostPacked &wdtPacked,
                                const std::vector<float> &uniformFieldValues)
{
  if (vecgeom::GeoManager::Instance().GetRegisteredVolumesCount() == 0)
    throw std::runtime_error("AdePTTransport::Initialize: Number of geometry volumes is zero.");

  std::cout << "=== AdePTTransport: initializing geometry and physics\n";
  if (!vecgeom::GeoManager::Instance().IsClosed())
    throw std::runtime_error("AdePTTransport::Initialize: VecGeom geometry not closed.");

  const vecgeom::cxx::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (!InitializeGeometry(world))
    throw std::runtime_error("AdePTTransport::Initialize: Cannot initialize geometry on GPU");

  if (!InitializePhysics()) throw std::runtime_error("AdePTTransport::Initialize cannot initialize physics on GPU");

  const auto numVolumes   = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  auto &volAuxArray       = adeptint::VolAuxArray::GetInstance();
  volAuxArray.fNumVolumes = numVolumes;
  volAuxArray.fAuxData    = auxData;
  adept::transport::InitVolAuxArray(volAuxArray);

  fHasWDTRegions = !wdtPacked.regions.empty();

  adeptint::WDTDeviceBuffers wdtDev;
  adept::transport::detail::InitWDTOnDevice(wdtPacked, wdtDev, fMaxWDTIter);
  fWDTDev = wdtDev;

  std::cout << "\nAllocating " << 4 * 8192 * fNThread << " To-device buffer slots\n";
  fBuffer = std::make_unique<TrackBuffer>(4 * 8192 * fNThread);

  assert(fBuffer != nullptr);

  fGPUstate =
      adept::transport::detail::InitializeGPU(fTrackCapacity, fStepCapacity, fNThread, *fBuffer, fCPUCapacityFactor,
                                              fCPUCopyFraction, fBfieldFile, uniformFieldValues);
  fGPUWorker = adept::transport::detail::LaunchGPUWorker(fTrackCapacity, fStepCapacity, fNThread, *fBuffer, *fGPUstate,
                                                         fEventStates, fCV_G4Workers, fAdePTSeed, fDebugLevel,
                                                         fReturnAllSteps, fReturnFirstAndLastStep, fLastNParticlesOnCPU,
                                                         fStepBufferSafetyFactor, fHasWDTRegions);
}

void AdePTTransport::InitBVH()
{
  vecgeom::cxx::BVHManager::Init();
  vecgeom::cxx::BVHManager::DeviceInit();
}

void AdePTTransport::RequestFlush(int threadId)
{
  fEventStates[threadId].store(EventState::G4RequestsFlush, std::memory_order_release);
}

void AdePTTransport::WaitForFlushProgress()
{
  std::unique_lock lock{fMutex_G4Workers};
  using namespace std::chrono_literals;
  fCV_G4Workers.wait_for(lock, 1ms);
}

bool AdePTTransport::AreReturnedStepsFlushed(int threadId) const
{
  return fEventStates[threadId].load(std::memory_order_acquire) >= EventState::StepsFlushed;
}

void AdePTTransport::MarkHostFlushed(int threadId)
{
  fEventStates[threadId].store(EventState::DeviceFlushed, std::memory_order_release);
}

} // namespace adept::transport
