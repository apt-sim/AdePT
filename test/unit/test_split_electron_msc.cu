// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

// This fixture only launches existing kernels. Keep its includes out of the
// device pass to avoid redefining SlotManager's out-of-line device methods.
#ifndef __CUDA_ARCH__

#include <AdePT/transport/config/TransportKernelOptions.hh>
#include <AdePT/transport/kernels/AdePTSteppingActionSelector.cuh>
#include <AdePT/transport/queues/ParticleManager.cuh>
#include <AdePT/transport/state/DeviceGlobals.cuh>
#include <AdePT/transport/state/TransportStats.hh>

#include <G4HepEmElectronData.hh>
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmMaterialData.hh>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/management/GeoManager.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <new>
#include <stdexcept>

// Launch the specialization compiled into Transport. Including electrons_split.cuh
// here would define its out-of-line device functions a second time.
namespace adept::transport {
template <bool IsElectron, class SteppingActionT>
__global__ void ElectronHowFar(ParticleManager, G4HepEmElectronTrack *, adept::MParray *, Stats *,
                               const adept::SteppingAction::Params, AllowFinishOffEventArray,
                               const TransportKernelOptions);
}

std::uint64_t BuildSplitElectronMscGeometry(int &logicalId);

namespace {
void CheckCuda(cudaError_t status)
{
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T>
struct Managed {
  T *data = nullptr;
  std::size_t count;
  explicit Managed(std::size_t size = 1) : count(size)
  {
    CheckCuda(cudaMallocManaged(&data, count * sizeof(T)));
    for (std::size_t i = 0; i < count; ++i)
      new (data + i) T{};
  }
  ~Managed()
  {
    for (std::size_t i = 0; i < count; ++i)
      data[i].~T();
    cudaFree(data);
  }
  Managed(const Managed &)            = delete;
  Managed &operator=(const Managed &) = delete;
  T &operator*() { return *data; }
  T *operator->() { return data; }
};

struct ManagedQueue {
  void *storage = nullptr;
  adept::MParray *data;
  ManagedQueue()
  {
    CheckCuda(cudaMallocManaged(&storage, adept::MParray::SizeOfInstance(1)));
    data = adept::MParray::MakeInstanceAt(1, storage);
    data->clear();
  }
  ~ManagedQueue()
  {
    adept::MParray::ReleaseInstance(data);
    cudaFree(storage);
  }
};

struct GeometryCleanup {
  ~GeometryCleanup()
  {
    vecgeom::cxx::CudaManager::Instance().CleanGpu();
    vecgeom::GeoManager::Instance().Clear();
  }
};

// Only HowFar's tables are needed: constant range=1, transport MFP=.1, and
// zero discrete cross sections. E=.1 MeV avoids the inverse-range conversion.
struct PhysicsData {
  G4HepEmData data{};
  G4HepEmParameters parameters{};
  G4HepEmRegionParmeters region{};
  G4HepEmElectronData electron{};
  G4HepEmMaterialData materials{};
  G4HepEmMatData material{};
  G4HepEmMatCutData couples{};
  G4HepEmMCCData couple{};
  double energies[2]{.05, .2};
  double energyLoss[10]{1., 0., 1., 0.};
  double transportXsecs[4]{10., 0., 10., 0.};
  int xsecStarts[1]{0};
  double xsecs[22]{2., 100., 0., 0., 0., 10., 0., 0., 100., 0., 0., 2., 100., 0., 0., 0., 10., 0., 0., 100., 0., 0.};
  double nuclearThreshold[1]{100.};

  PhysicsData()
  {
    data.fTheElectronData   = &electron;
    data.fThePositronData   = &electron;
    data.fTheMaterialData   = &materials;
    data.fTheMatCutData     = &couples;
    materials.fMaterialData = &material;
    couples.fMatCutData     = &couple;
    couple.fHepEmMatIndex = couple.fG4RegionIndex = 0;
    parameters.fNumRegions                        = 1;
    parameters.fParametersPerRegion               = &region;
    region.fFinalRange                            = 0.;
    region.fMSCSafetyFactor                       = .1;
    material.fUMSCPar                             = .3;
    material.fUMSCStepMinPars[0]                  = 1.;
    material.fUMSCStepMinPars[1]                  = 1.;
    material.fZeffSqrt = material.fZeff23   = 1.;
    material.fDensity                       = 1.;
    electron.fELossEnergyGridSize           = 2;
    electron.fELossEnergyGrid               = energies;
    electron.fELossLogMinEkin               = std::log(energies[0]);
    electron.fELossEILDelta                 = 1. / std::log(energies[1] / energies[0]);
    electron.fELossData                     = energyLoss;
    electron.fTr1MacXSecData                = transportXsecs;
    electron.fResMacXSecStartIndexPerMatCut = xsecStarts;
    electron.fResMacXSecData                = xsecs;
    electron.fENucEnergyGrid                = nuclearThreshold;
  }
};

template <bool IsElectron>
void CheckDeferredInitialization(bool tinyFirstStep)
{
  using namespace adept::transport;
  GeometryCleanup cleanup;
  vecgeom::GeoManager::Instance().Clear();
  int logicalId;
  const auto navIndex = BuildSplitElectronMscGeometry(logicalId);
  ASSERT_TRUE(vecgeom::cxx::CudaManager::Instance().SynchronizeNavigationTable());
  Managed<adeptint::VolAuxData> aux(logicalId + 1);
  Managed<PhysicsData> physics;
  Managed<ChargedTrack> track;
  Managed<G4HepEmElectronTrack> hepEmTrack;
  Managed<SlotManager> slots;
  Managed<Stats> stats;
  Managed<unsigned int> inFlight;
  Managed<unsigned short> finishOff;
  ManagedQueue active, propagation;
  ASSERT_TRUE(active.data->push_back(0));
  track->navState.SetNavIndex(navIndex);
  track->rngState = RanluxppDouble{12345};
  track->eKin     = .1;
  track->pos      = {0., 0., 0.};
  track->threadId = 0;
  track->dir      = {1., 0., 0.};
  track->SetSafety(track->pos, tinyFirstStep ? .25 : .4);
  stats->perEventInFlightPrevious = inFlight.data;
  *inFlight                       = 1;
  ParticleManager manager{{track.data, slots.data, active.data, nullptr},
                          {track.data, slots.data, active.data, nullptr},
                          {nullptr, nullptr, nullptr, nullptr},
                          {nullptr, nullptr, nullptr, nullptr}};
  if (tinyFirstStep) physics->region.fDRoverRange = 1.e-8;
  CheckCuda(cudaMemcpyToSymbol(gVolAuxData, &aux.data, sizeof(aux.data)));
  CheckCuda(cudaMemcpyToSymbol(g4HepEmData, &physics->data, sizeof(physics->data)));
  CheckCuda(cudaMemcpyToSymbol(g4HepEmPars, &physics->parameters, sizeof(physics->parameters)));
  // The isolated executable starts with a null magnetic-field pointer.
  adept::SteppingAction::Params actionParameters{};
#if ADEPT_STEPACTION_TYPE == 2
  actionParameters.ecut = 0.;
#endif
  auto launch = [&] {
    propagation.data->clear();
    ElectronHowFar<IsElectron, adept::SteppingAction::Action>
        <<<1, 32>>>(manager, hepEmTrack.data, propagation.data, stats.data, actionParameters, {finishOff.data}, {});
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  };
  launch();
  auto *msc = hepEmTrack->GetMSCTrackData();
  ASSERT_EQ(propagation.data->size(), 1u);
  ASSERT_TRUE(track->hepEmTrackExists);
  ASSERT_TRUE(msc->fIsFirstStep);
  ASSERT_DOUBLE_EQ(msc->fInitialRange, 1.e21);

  // A shorter cached safety makes UseSafety eligible without crossing a
  // boundary. Both safeties exceed the .2 physics proposal, so navigation is
  // not needed. For tiny-step coverage, instead restore a normal proposal.
  if (tinyFirstStep) {
    physics->region.fDRoverRange = .2;
  } else {
    track->pos = {.15, 0., 0.};
    ASSERT_NEAR(track->GetSafety(track->pos), .25, 1.e-12);
  }
  ASSERT_FALSE(track->navState.IsOnBoundary());
  launch();
  ASSERT_EQ(propagation.data->size(), 1u);
  EXPECT_FALSE(msc->fIsFirstStep);
  EXPECT_DOUBLE_EQ(msc->fInitialRange, 1.);
  EXPECT_LT(hepEmTrack->GetPStepLength(), .1);
}
} // namespace

TEST(SplitElectronMSC, DefersSafetyInitializationForElectron)
{
  CheckDeferredInitialization<true>(false);
}
TEST(SplitElectronMSC, DefersSafetyInitializationForPositron)
{
  CheckDeferredInitialization<false>(false);
}
TEST(SplitElectronMSC, DefersInitializationPastTinyElectronStep)
{
  CheckDeferredInitialization<true>(true);
}
TEST(SplitElectronMSC, DefersInitializationPastTinyPositronStep)
{
  CheckDeferredInitialization<false>(true);
}

#endif // !__CUDA_ARCH__
