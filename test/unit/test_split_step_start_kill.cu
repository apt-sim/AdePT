// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

// Exercise the real split kernels, stopping before physics/navigation. WDT
// requires a valid enclosing root for its initial admission check.
#include <cstdint>

#include <AdePT/transport/random/G4HepEmRandomEngineDeviceImpl.hh>
#include <AdePT/transport/tracks/TrackDebug.cuh>
#include <AdePT/transport/kernels/electrons_split.cuh>
#include <AdePT/transport/kernels/gammas_split.cuh>

#include <gtest/gtest.h>

#ifdef ADEPT_ENABLE_WDT
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/management/GeoManager.h>
#include <array>

void BuildWoodcockTestGeometry(std::array<std::uint64_t, 7> &, std::array<int, 4> &);
#endif

#include <new>
#include <string>

// This executable owns the kernel globals and links their dependencies directly:
// linking Transport would also bring a second copy of these kernel definitions.
namespace adept::transport {
__constant__ __device__ G4HepEmParameters g4HepEmPars;
__constant__ __device__ G4HepEmData g4HepEmData;
__constant__ __device__ adeptint::VolAuxData *gVolAuxData = nullptr;
__constant__ __device__ adeptint::WDTDeviceView gWDTData;
#ifdef ADEPT_USE_EXT_BFIELD
__constant__ __device__ GeneralMagneticField *gMagneticField = nullptr;
#else
__constant__ __device__ UniformMagneticField *gMagneticField = nullptr;
#endif
} // namespace adept::transport

namespace {
using namespace adept::transport;

enum class Species { Electron, Positron, Gamma, WoodcockGamma };
enum class KillCause { EnergyCut, MaxSteps, Stuck, Looper };

std::string SpeciesName(const testing::TestParamInfo<Species> &info)
{
  switch (info.param) {
  case Species::Electron:
    return "Electron";
  case Species::Positron:
    return "Positron";
  case Species::Gamma:
    return "Gamma";
  case Species::WoodcockGamma:
    return "WoodcockGamma";
  }
  return "Unknown";
}

// Adapt the selected kernel parameter type to the real LHCb action, so this
// regression also exercises LHCb's depositing cut in NONE/CMS/ATLAS builds.
struct LHCbEnergyCut : adept::SteppingAction::NoAction {
  __device__ static void ElectronAction(bool &alive, double &eKin, double &edep, const vecgeom::Vector3D<double> &pos,
                                        const double &time, const VolAuxData &aux, const G4HepEmData *data,
                                        const StepActionParam &)
  {
    adept::SteppingAction::LHCbAction::ElectronAction(alive, eKin, edep, pos, time, aux, data,
                                                      LHCbSteppingActionParams{});
  }

  __device__ static void GammaAction(bool &alive, double &eKin, double &edep, const vecgeom::Vector3D<double> &pos,
                                     const double &time, const VolAuxData &aux, const G4HepEmData *data,
                                     const StepActionParam &)
  {
    adept::SteppingAction::LHCbAction::GammaAction(alive, eKin, edep, pos, time, aux, data, LHCbSteppingActionParams{});
  }
};

template <class T>
class Managed {
public:
  Managed()
  {
    ADEPT_DEVICE_API_CALL(MallocManaged(&fValue, sizeof(T)));
    new (fValue) T{};
  }
  ~Managed()
  {
    fValue->~T();
    ADEPT_DEVICE_API_SYMBOL(Free)(fValue);
  }
  T *get() const { return fValue; }
  T &operator*() const { return *fValue; }
  T *operator->() const { return fValue; }
  Managed(const Managed &)            = delete;
  Managed &operator=(const Managed &) = delete;

private:
  T *fValue{nullptr};
};

struct TestState {
  ChargedTrack charged;
  NeutralTrack neutral;
  G4HepEmElectronTrack electron;
  G4HepEmGammaTrack gamma;
  VolAuxData aux;
#ifdef ADEPT_ENABLE_WDT
  adeptint::WDTRoot wdtRoot;
  adeptint::WDTRegion wdtRegion{0, 1, 0.f};
  int regionToWDT{0};
#endif
  SlotManager slots;
  unsigned int freedSlot{99};
  unsigned int stepCount{0};
  GPUStep step;
};

class SplitStepStartKill : public testing::TestWithParam<Species> {
public:
#ifdef ADEPT_ENABLE_WDT
  static void SetUpTestSuite()
  {
    std::array<std::uint64_t, 7> indices{};
    std::array<int, 4> logicalIds{};
    BuildWoodcockTestGeometry(indices, logicalIds);
    // The single auxiliary entry below represents the world's logical volume.
    ASSERT_EQ(logicalIds[0], 0);
    fWorldIndex = indices[6];
    ASSERT_TRUE(vecgeom::cxx::CudaManager::Instance().SynchronizeNavigationTable());
  }

  static void TearDownTestSuite()
  {
    vecgeom::cxx::CudaManager::Instance().CleanGpu();
    vecgeom::GeoManager::Instance().Clear();
  }
#endif

protected:
  void SetUp() override
  {
    ADEPT_DEVICE_API_CALL(MallocManaged(&fQueueStorage, adept::MParray::SizeOfInstance(1)));
    fQueue = adept::MParray::MakeInstanceAt(1, fQueueStorage);
    ASSERT_TRUE(fQueue->push_back(0));
  }

  void TearDown() override
  {
    adept::MParray::ReleaseInstance(fQueue);
    ADEPT_DEVICE_API_SYMBOL(Free)(fQueueStorage);
  }

  bool IsCharged() const { return GetParam() == Species::Electron || GetParam() == Species::Positron; }

  void Run(KillCause cause, bool sensitive, bool returnAll, bool returnLast, bool hasHostData, bool expectRecord,
           double kineticEnergy = 0.25 * copcore::units::MeV)
  {
    Managed<TestState> state;
    auto &track = IsCharged() ? static_cast<TrackBase &>(state->charged) : static_cast<TrackBase &>(state->neutral);
    track.eKin  = kineticEnergy;
    track.pos   = {1., 2., 3.};
    track.dir   = {0., 0., 1.};
    track.globalTime           = 12.;
    track.localTime            = 5.f;
    track.properTime           = 2.f;
    track.trackId              = 123;
    track.parentId             = 45;
    track.eventId              = 6;
    track.threadId             = 0;
    track.weight               = 0.5f;
    track.hasHostData          = hasHostData;
    track.hepEmTrackExists     = true;
    track.stepCounter          = cause == KillCause::MaxSteps ? 9999 : 7;
    track.zeroStepCounter      = cause == KillCause::Stuck ? 26 : 0;
    track.looperCounter        = cause == KillCause::Looper ? 501 : 0;
    state->aux.fSensIndex      = sensitive ? 0 : -1;
    state->slots.fFreeListSize = 1;
    state->slots.fToFreeList   = &state->freedSlot;
    // Seed persistent HepEm state with a previous step's length. The terminal
    // record must describe the current zero-length kill, not reuse this value.
    state->electron.GetTrack()->SetGStepLength(123.);
    state->gamma.GetTrack()->SetGStepLength(123.);
    state->gamma.GetTrack()->SetNumIALeft(1., 0);

#ifdef ADEPT_ENABLE_WDT
    if (GetParam() == Species::WoodcockGamma) {
      track.navState.SetNavIndex(fWorldIndex);
      state->aux.fGPUregionId = 0;
      state->wdtRoot          = {track.navState, 0};
      const adeptint::WDTDeviceView view{&state->wdtRoot, &state->wdtRegion, &state->regionToWDT, 1, 1, 1};
      ADEPT_DEVICE_API_CALL(MemcpyToSymbol(gWDTData, &view, sizeof(view)));
    }
#endif
    auto *aux = &state->aux;
    ADEPT_DEVICE_API_CALL(MemcpyToSymbol(gVolAuxData, &aux, sizeof(aux)));
    DeviceStepBufferView buffer{&state->step, &state->stepCount, 1, 1};
    ADEPT_DEVICE_API_CALL(MemcpyToSymbol(gDeviceStepBuffer, &buffer, sizeof(buffer)));
    ParticleManager manager{{&state->charged, &state->slots, fQueue, nullptr},
                            {&state->charged, &state->slots, fQueue, nullptr},
                            {&state->neutral, &state->slots, fQueue, nullptr},
                            {&state->neutral, &state->slots, fQueue, nullptr}};
    TransportKernelOptions options;
    options.returnAllSteps = returnAll;
    options.returnLastStep = returnLast;
    // All paths under test kill before consulting transport statistics or
    // enqueuing propagation. Null pointers make an unexpected survival fail.
    switch (GetParam()) {
    case Species::Electron:
      ElectronHowFar<true, LHCbEnergyCut><<<1, 1>>>(manager, &state->electron, nullptr, nullptr, {}, {}, options);
      break;
    case Species::Positron:
      ElectronHowFar<false, LHCbEnergyCut><<<1, 1>>>(manager, &state->electron, nullptr, nullptr, {}, {}, options);
      break;
    case Species::Gamma:
      GammaHowFar<LHCbEnergyCut><<<1, 1>>>(&state->gamma, manager, nullptr, nullptr, {}, {}, options);
      break;
    case Species::WoodcockGamma:
#ifdef ADEPT_ENABLE_WDT
      GammaWoodcock<LHCbEnergyCut><<<1, 1>>>(&state->gamma, manager, nullptr, nullptr, {}, {}, options);
#endif
      break;
    }
    ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());
    EXPECT_EQ(state->slots.fFreeCounter, 1u);
    EXPECT_EQ(state->freedSlot, 0u);
    EXPECT_EQ(track.eKin, 0.);
    ASSERT_EQ(state->stepCount, expectRecord ? 1u : 0u);
    if (!expectRecord) return;

    const auto &step       = state->step;
    const auto particle    = GetParam() == Species::Electron   ? ParticleType::Electron
                             : GetParam() == Species::Positron ? ParticleType::Positron
                                                               : ParticleType::Gamma;
    double expectedDeposit = kineticEnergy;
    if (GetParam() == Species::Positron && cause != KillCause::EnergyCut)
      expectedDeposit += 2 * copcore::units::kElectronMassC2;
    EXPECT_DOUBLE_EQ(step.fTotalEnergyDeposit, expectedDeposit);
    EXPECT_DOUBLE_EQ(step.fStepLength, 0.);
    EXPECT_DOUBLE_EQ(step.fPreStepPoint.fEKin, kineticEnergy);
    EXPECT_DOUBLE_EQ(step.fPostStepPoint.fEKin, 0.);
    EXPECT_EQ(step.fParticleType, particle);
    EXPECT_EQ(step.fTrackID, track.trackId);
    EXPECT_EQ(step.fParentID, track.parentId);
    EXPECT_EQ(step.fEventId, track.eventId);
    EXPECT_EQ(step.threadId, track.threadId);
    EXPECT_EQ(step.fStepCounter, track.stepCounter);
    EXPECT_EQ(step.fTrackWeight, track.weight);
    EXPECT_EQ(step.fStepLimProcessId, kAdePTTransportationProcess);
    EXPECT_TRUE(step.fLastStepOfTrack);
    EXPECT_EQ(step.fHasHostData, hasHostData);
    EXPECT_EQ(step.fNumSecondaries, 0);
    EXPECT_DOUBLE_EQ(step.fPreGlobalTime, track.globalTime);
    EXPECT_DOUBLE_EQ(step.fGlobalTime, track.globalTime);
    EXPECT_FLOAT_EQ(step.fLocalTime, track.localTime);
    EXPECT_FLOAT_EQ(step.fProperTime, track.properTime);
    for (int axis = 0; axis < 3; ++axis) {
      EXPECT_DOUBLE_EQ(step.fPreStepPoint.fPosition[axis], track.pos[axis]);
      EXPECT_DOUBLE_EQ(step.fPostStepPoint.fPosition[axis], track.pos[axis]);
      EXPECT_DOUBLE_EQ(step.fPreStepPoint.fMomentumDirection[axis], track.dir[axis]);
      EXPECT_DOUBLE_EQ(step.fPostStepPoint.fMomentumDirection[axis], track.dir[axis]);
    }
  }

private:
#ifdef ADEPT_ENABLE_WDT
  inline static std::uint64_t fWorldIndex{0};
#endif
  char *fQueueStorage{nullptr};
  adept::MParray *fQueue{nullptr};
};

TEST_P(SplitStepStartKill, SensitiveSafetyKillsDepositWithoutHostData)
{
  for (auto cause : {KillCause::MaxSteps, KillCause::Stuck, KillCause::Looper}) {
    if (cause == KillCause::Looper && !IsCharged()) continue;
    SCOPED_TRACE(static_cast<int>(cause));
    Run(cause, true, false, false, false, true);
  }
}

TEST_P(SplitStepStartKill, LHCbEnergyCutDepositsWithoutHostData)
{
  Run(KillCause::EnergyCut, true, false, false, false, true);
}

TEST_P(SplitStepStartKill, AllStepsRecordsNonSensitiveKills)
{
  Run(KillCause::EnergyCut, false, true, false, false, true);
  Run(KillCause::EnergyCut, false, true, false, false, true, 0.);
}

TEST_P(SplitStepStartKill, LastStepOptionRecordsNonSensitiveKills)
{
  Run(KillCause::EnergyCut, false, false, true, false, true);
}

TEST_P(SplitStepStartKill, HostDataRecordsNonSensitiveKills)
{
  Run(KillCause::EnergyCut, false, false, false, true, true);
}

TEST_P(SplitStepStartKill, UnrequestedNonSensitiveKillsDoNotRecord)
{
  Run(KillCause::EnergyCut, false, false, false, false, false);
}

TEST_P(SplitStepStartKill, ZeroDepositDoesNotCreateSensitiveHit)
{
  Run(KillCause::EnergyCut, true, false, false, false, false, 0.);
}

INSTANTIATE_TEST_SUITE_P(ParticleKernels, SplitStepStartKill,
                         testing::Values(Species::Electron, Species::Positron, Species::Gamma
#ifdef ADEPT_ENABLE_WDT
                                         ,
                                         Species::WoodcockGamma
#endif
                                         ),
                         SpeciesName);
} // namespace
