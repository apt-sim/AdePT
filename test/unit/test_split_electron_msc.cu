// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

// Check that deferred MSC initialization survives across split-kernel launches.
// Each case calls the production ElectronHowFar kernel twice for the same track:
//   1. G4HepEm skips MSC initialization and leaves fIsFirstStep true.
//   2. MSC becomes eligible and must initialize its limits before using them.
// The four cases cover both charges and two distinct reasons for the first skip.
// We control the inputs between launches; propagation and scattering kernels are
// not run. The behavior under test is the lifetime of the pending MSC state.

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

// Unified memory lets host setup/assertions and the production GPU kernel access
// the same objects. In particular, PhysicsData's pointers refer to its own arrays
// inside this managed allocation, so those tables are device-readable too.
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

// MParray uses a trailing variable-sized element array, so allocate its full
// buffer. One entry is enough for the single track in this test.
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

// Isolate the four cases when they run together in one GTest process. This guard
// is created first, so track/table allocations are freed before geometry cleanup.
struct GeometryCleanup {
  ~GeometryCleanup()
  {
    vecgeom::cxx::CudaManager::Instance().CleanGpu();
    vecgeom::GeoManager::Instance().Clear();
  }
};

// Minimal synthetic physics for one material, material-cuts couple, and region.
// Both charges share tables giving a constant range of 1 mm and transport mean
// free path of 0.1 mm. Discrete interactions impose no limit, so a normal call
// starts with the continuous-loss proposal of 0.2 mm and MSC can shorten it.
// E = 0.1 MeV selects a true-to-geometric conversion that needs no inverse-range
// table. These inputs make the initialization transition easy to distinguish.
struct PhysicsData {
  G4HepEmData data{};
  G4HepEmParameters parameters{};
  G4HepEmRegionParmeters region{};
  G4HepEmElectronData electron{};
  G4HepEmMaterialData materials{};
  G4HepEmMatData material{};
  G4HepEmMatCutData couples{};
  G4HepEmMCCData couple{};
  // The two grid energies bracket the test energy of 0.1 MeV.
  double energies[2]{.05, .2};
  // Range data alternate (value, spline second derivative): (1, 0) at each
  // grid point. The remaining entries in the energy-loss table are unused here.
  double energyLoss[10]{1., 0., 1., 0.};
  // Transport cross section = 10/mm, with zero spline second derivatives,
  // gives the constant transport mean free path 1/(10/mm) = 0.1 mm.
  double transportXsecs[4]{10., 0., 10., 0.};
  int xsecStarts[1]{0};
  // Ionization and bremsstrahlung each use a five-entry header followed by two
  // (energy, cross section, second derivative) triples. Their first grid energy
  // is 10 MeV, above the track energy, so G4HepEm returns zero cross sections.
  double xsecs[22]{2., 100., 0., 0., 0., 10., 0., 0., 100., 0., 0., 2., 100., 0., 0., 0., 10., 0., 0., 100., 0., 0.};
  // Likewise, the nuclear lookup returns before accessing any other table.
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
    // Default region parameters select UseSafety and an MSC range factor of
    // 0.04. With final range = 0 and DRoverRange = 0.2, the continuous proposal
    // is simply range * DRoverRange = 1 * 0.2 = 0.2 mm.
    region.fFinalRange      = 0.;
    region.fMSCSafetyFactor = .1;
    material.fUMSCPar       = .3;
    // These material coefficients keep the minimum MSC step below the 0.04 mm
    // range-based cap. Electron density stays zero to suppress positron
    // annihilation; nonzero mass density keeps the CMS stepping action alive.
    material.fUMSCStepMinPars[0] = 1.;
    material.fUMSCStepMinPars[1] = 1.;
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

// IsElectron selects the electron or positron production-kernel specialization.
// tinyFirstStep selects which G4HepEm early return defers initialization:
// false -> safety check in StepLimit; true -> tiny-step check in HowFarToMSC.
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
  // Slot 0 stays active for both launches. HowFar only fills the propagation
  // queue; no propagation kernel consumes it or changes the persistent track.
  ASSERT_TRUE(active.data->push_back(0));
  track->navState.SetNavIndex(navIndex);
  track->rngState = RanluxppDouble{12345};
  track->eKin     = .1;
  track->pos      = {0., 0., 0.};
  track->threadId = 0;
  track->dir      = {1., 0., 0.};
  track->SetSafety(track->pos, tinyFirstStep ? .25 : .4);
  // Keep the event active with finish-off disabled, so CPU handoff cannot
  // bypass the MSC calculation. Only the selected charge accesses this storage.
  stats->perEventInFlightPrevious = inFlight.data;
  *inFlight                       = 1;
  ParticleManager manager{{track.data, slots.data, active.data, nullptr},
                          {track.data, slots.data, active.data, nullptr},
                          {nullptr, nullptr, nullptr, nullptr},
                          {nullptr, nullptr, nullptr, nullptr}};
  // For the tiny-step case, range * DRoverRange = 1e-8 mm is below the
  // 5e-8 mm cutoff in HowFarToMSC. Safety = 0.25 mm would otherwise allow MSC.
  if (tinyFirstStep) physics->region.fDRoverRange = 1.e-8;
  // Point the production kernel globals at the managed fixture data. The same
  // tables and G4HepEm track remain available for the second launch.
  CheckCuda(cudaMemcpyToSymbol(gVolAuxData, &aux.data, sizeof(aux.data)));
  CheckCuda(cudaMemcpyToSymbol(g4HepEmData, &physics->data, sizeof(physics->data)));
  CheckCuda(cudaMemcpyToSymbol(g4HepEmPars, &physics->parameters, sizeof(physics->parameters)));
  // The isolated executable starts with a null magnetic-field pointer.
  adept::SteppingAction::Params actionParameters{};
#if ADEPT_STEPACTION_TYPE == 2
  // Keep this low-energy fixture alive in builds using the LHCb energy cut.
  actionParameters.ecut = 0.;
#endif
  auto launch = [&] {
    propagation.data->clear();
    ElectronHowFar<IsElectron, adept::SteppingAction::Action>
        <<<1, 32>>>(manager, hepEmTrack.data, propagation.data, stats.data, actionParameters, {finishOff.data}, {});
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  };
  // First launch: the safety case has range * fUMSCPar = 1 * 0.3 < 0.4 mm,
  // so StepLimit returns before initializing. The tiny-step case instead skips
  // StepLimit entirely. Both must leave the flag true and range at its sentinel.
  launch();
  auto *msc = hepEmTrack->GetMSCTrackData();
  ASSERT_EQ(propagation.data->size(), 1u);
  ASSERT_TRUE(track->hepEmTrackExists);
  ASSERT_TRUE(msc->fIsFirstStep);
  ASSERT_DOUBLE_EQ(msc->fInitialRange, 1.e21);

  // Prepare an eligible second call while reusing the same track and slot.
  // hepEmTrackExists stays true, exercising the branch that used to clear the
  // first-step flag prematurely.
  // In the safety case, changing the fixture position by 0.15 mm contracts the
  // cached safety from 0.4 to 0.25 mm, now below range * fUMSCPar = 0.3 mm.
  // This position change supplies a new input; no propagation kernel runs.
  // In the tiny-step case, restore the normal 0.2 mm physical proposal instead.
  // Safety is now 0.25 mm in both cases, above that proposal, so HowFar uses
  // the cache and does not recompute safety through the geometry navigator.
  if (tinyFirstStep) {
    physics->region.fDRoverRange = .2;
  } else {
    track->pos = {.15, 0., 0.};
    ASSERT_NEAR(track->GetSafety(track->pos), .25, 1.e-12);
  }
  // A boundary would initialize MSC independently of fIsFirstStep and could
  // hide the bug. The second launch must initialize during an interior step.
  ASSERT_FALSE(track->navState.IsOnBoundary());
  launch();
  ASSERT_EQ(propagation.data->size(), 1u);
  // Successful initialization clears the flag and stores max(range, MFP) =
  // max(1, 0.1) = 1 mm. The same persistent G4HepEm object must carry this state.
  EXPECT_FALSE(msc->fIsFirstStep);
  EXPECT_DOUBLE_EQ(msc->fInitialRange, 1.);
  // The nominal UseSafety cap is max(0.04 * 1, 0.1 * 0.25, minimum step) =
  // 0.04 mm. Allow its randomization while still rejecting the original bug:
  // prematurely clearing the flag leaves initialRange = 1e21 and step = 0.2 mm.
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
