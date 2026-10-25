// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/returned_steps/HostTrackDataMapper.hh>
#include <AdePT/g4integration/tracking_managers/G4HepEmTrackingManagerSpecialized.hh>
#include <AdePT/g4integration/geometry/AdePTGeometryBridge.hh>
#include <AdePT/transport/steps/GPUStep.hh>

#include <G4Track.hh>

#include <memory>
#include <span>
#include <vector>

#include <AdePT/g4integration/returned_steps/AdePTGeant4Integration.hh>

#include <VecGeom/management/GeoManager.h>

#include <G4Box.hh>
#include <G4Electron.hh>
#include <G4EventManager.hh>
#include <G4Gamma.hh>
#include <G4GeometryManager.hh>
#include <G4LogicalVolume.hh>
#include <G4NistManager.hh>
#include <G4PVPlacement.hh>
#include <G4Positron.hh>
#include <G4StackManager.hh>
#include <G4Step.hh>
#include <G4SystemOfUnits.hh>
#include <G4TouchableHistory.hh>
#include <G4TransportationManager.hh>
#include <G4VSensitiveDetector.hh>

#include <gtest/gtest.h>

namespace {

struct ReplayGeometryCleanup {
  G4LogicalVolume *logical = nullptr;

  ~ReplayGeometryCleanup()
  {
    if (logical != nullptr) logical->SetSensitiveDetector(nullptr);
    G4GeometryManager::GetInstance()->OpenGeometry();
    vecgeom::GeoManager::Instance().Clear();
  }
};

class CountedReplayTrackInformation : public G4VUserTrackInformation {
public:
  CountedReplayTrackInformation(int value, int &counter) : tag(value), deletions(counter) {}
  ~CountedReplayTrackInformation() override { ++deletions; }

  void Print() const override {}

  int tag;

private:
  int &deletions;
};

class ReplayStepSensitiveDetector : public G4VSensitiveDetector {
public:
  ReplayStepSensitiveDetector() : G4VSensitiveDetector("returned_step_counter_sd") {}

  struct Observation {
    int stepNumber;
    int trackId;
    int informationTag;
    G4Track const *replayTrack;
    G4DynamicParticle const *replayParticle;
  };

  std::vector<Observation> observations;

protected:
  G4bool ProcessHits(G4Step *step, G4TouchableHistory *) override
  {
    auto const *track = step->GetTrack();
    auto const *info  = dynamic_cast<CountedReplayTrackInformation const *>(track->GetUserInformation());
    observations.push_back({track->GetCurrentStepNumber(), track->GetTrackID(), info != nullptr ? info->tag : -1, track,
                            track->GetDynamicParticle()});
    return true;
  }
};

} // namespace

TEST(AdePTGeant4Integration, ReplayParentStepNumberIsReset)
{
#ifndef ADEPT_HAS_G4TRACK_NSTEP_INCREMENT
  GTEST_SKIP() << "Geant4 lacks the counted step increment; replay counters remain unchanged";
#else
  ReplayGeometryCleanup cleanup;
  G4GeometryManager::GetInstance()->OpenGeometry();
  vecgeom::GeoManager::Instance().Clear();

  // ProcessGPUStep constructs all three replay species, even when a step has no secondaries.
  G4Electron::Definition();
  G4Positron::Definition();
  G4Gamma::Definition();

  auto *material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  ASSERT_NE(material, nullptr);
  auto *solid     = new G4Box("replay_counter_world_solid", 10.0 * mm, 10.0 * mm, 10.0 * mm);
  auto *logical   = new G4LogicalVolume(solid, material, "replay_counter_world_lv");
  auto *world     = new G4PVPlacement(nullptr, G4ThreeVector(), logical, "replay_counter_world", nullptr, false, 0);
  cleanup.logical = logical;
  ReplayStepSensitiveDetector detector;
  logical->SetSensitiveDetector(&detector);

  G4TransportationManager::GetTransportationManager()->SetWorldForTracking(world);
  G4GeometryManager::GetInstance()->CloseGeometry(true);
  AdePTGeometryBridge::CreateVecGeomWorld(world);
  auto const *vgWorld = vecgeom::GeoManager::Instance().GetWorld();
  ASSERT_NE(vgWorld, nullptr);
  vecgeom::NavigationState navigation;
  navigation.Push(vgWorld);
  navigation.SetBoundaryState(false);

  G4HepEmTrackingManagerSpecialized hepEm;
  AdePTGeant4Integration integration;
  integration.SetHepEmTrackingManager(&hepEm);
  auto &mapper = integration.GetHostTrackDataMapper();
  mapper.beginEvent(0);

  // Replay A:5, B:init, B:1, A:6 on each species' shared G4Track; A advances from step 5 to 6.
  for (auto const particleType : {ParticleType::Electron, ParticleType::Positron, ParticleType::Gamma}) {
    SCOPED_TRACE(static_cast<int>(particleType));
    detector.observations.clear();
    int deletionsA                   = 0;
    int deletionsB                   = 0;
    constexpr uint64_t trackA        = 101;
    constexpr uint64_t trackB        = 102;
    const int g4idA                  = mapper.create(trackA).g4id;
    mapper.get(trackA).userTrackInfo = new CountedReplayTrackInformation(1, deletionsA);
    const int g4idB                  = mapper.create(trackB).g4id;
    mapper.get(trackB).userTrackInfo = new CountedReplayTrackInformation(2, deletionsB);

    const auto replay = [&](uint64_t trackId, unsigned short stepNumber, bool lastStep) {
      GPUStep step;
      step.fTrackID                          = trackId;
      step.fParticleType                     = particleType;
      step.fStepLimProcessId                 = kAdePTTransportationProcess;
      step.fStepCounter                      = stepNumber;
      step.fLastStepOfTrack                  = lastStep;
      step.fHasHostData                      = true;
      step.fStepLength                       = 1.0 * mm;
      step.fTotalEnergyDeposit               = 0.1 * MeV;
      step.fPreStepPoint.fNavigationState    = navigation;
      step.fPreStepPoint.fPosition           = {0.0, 0.0, 0.0};
      step.fPreStepPoint.fMomentumDirection  = {0.0, 0.0, 1.0};
      step.fPreStepPoint.fEKin               = 1.0 * MeV;
      step.fPostStepPoint.fNavigationState   = navigation;
      step.fPostStepPoint.fPosition          = {0.0, 0.0, 1.0 * mm};
      step.fPostStepPoint.fMomentumDirection = {0.0, 0.0, 1.0};
      step.fPostStepPoint.fEKin              = 0.9 * MeV;
      if (stepNumber == 0) {
        step.fStepLength         = 0.0;
        step.fTotalEnergyDeposit = 0.0;
        step.fPostStepPoint      = step.fPreStepPoint;
      }
      integration.ProcessGPUStep(std::span<const GPUStep>(&step, 1), false, false, true);
    };

    replay(trackA, 5, false);
    EXPECT_EQ(deletionsA, 0);
    EXPECT_EQ(deletionsB, 0);

    replay(trackB, 0, false);
    ASSERT_EQ(detector.observations.size(), 1u); // Initializing records do not reach the sensitive detector.
    EXPECT_EQ(detector.observations[0].replayTrack->GetCurrentStepNumber(), 0);
    EXPECT_EQ(deletionsA, 0);
    EXPECT_EQ(deletionsB, 0);

    replay(trackB, 1, true);
    EXPECT_EQ(deletionsA, 0); // Reusing the parent must not delete A's mapper-owned information.
    EXPECT_EQ(deletionsB, 1);
    EXPECT_TRUE(mapper.contains(trackA));
    EXPECT_FALSE(mapper.contains(trackB));

    replay(trackA, 6, true);
    EXPECT_EQ(deletionsA, 1);
    EXPECT_EQ(deletionsB, 1);
    EXPECT_FALSE(mapper.contains(trackA));

    ASSERT_EQ(detector.observations.size(), 3u);
    EXPECT_EQ(detector.observations[0].stepNumber, 5);
    EXPECT_EQ(detector.observations[1].stepNumber, 1);
    EXPECT_EQ(detector.observations[2].stepNumber, 6);
    EXPECT_EQ(detector.observations[0].trackId, g4idA);
    EXPECT_EQ(detector.observations[1].trackId, g4idB);
    EXPECT_EQ(detector.observations[2].trackId, g4idA);
    EXPECT_EQ(detector.observations[0].informationTag, 1);
    EXPECT_EQ(detector.observations[1].informationTag, 2);
    EXPECT_EQ(detector.observations[2].informationTag, 1);
    EXPECT_EQ(detector.observations[0].replayTrack, detector.observations[1].replayTrack);
    EXPECT_EQ(detector.observations[0].replayTrack, detector.observations[2].replayTrack);
    EXPECT_EQ(detector.observations[0].replayParticle, detector.observations[1].replayParticle);
    EXPECT_EQ(detector.observations[0].replayParticle, detector.observations[2].replayParticle);
  }

  // CPU handoff creates a fresh track and retains its existing minimum of one step.
  ASSERT_EQ(G4EventManager::GetEventManager(), nullptr);
  G4EventManager eventManager;
  auto *stack = eventManager.GetStackManager();
  for (unsigned short const stepNumber : {0, 7}) {
    SCOPED_TRACE(stepNumber);
    int deletions          = 0;
    const uint64_t trackId = 200 + stepNumber;
    auto &hostData         = mapper.create(trackId);
    const int g4id         = hostData.g4id;
    auto *information      = new CountedReplayTrackInformation(3, deletions);
    hostData.userTrackInfo = information;
#ifdef ADEPT_USE_ORIGINNAVSTATE
    hostData.originNavState = navigation;
#endif

    GPUStep step;
    step.fTrackID                          = trackId;
    step.fParticleType                     = ParticleType::Gamma;
    step.fStepLimProcessId                 = kAdePTFinishOnCPUProcess;
    step.fStepCounter                      = stepNumber;
    step.fHasHostData                      = true;
    step.fPostStepPoint.fNavigationState   = navigation;
    step.fPostStepPoint.fPosition          = {0.0, 0.0, 0.0};
    step.fPostStepPoint.fMomentumDirection = {0.0, 0.0, 1.0};
    step.fPostStepPoint.fEKin              = 1.0 * MeV;
    integration.ReturnDeferredTrack(std::span<const GPUStep>(&step, 1), true);

    G4VTrajectory *trajectory = nullptr;
    std::unique_ptr<G4Track> returned(stack->PopNextTrack(&trajectory));
    ASSERT_NE(returned, nullptr);
    EXPECT_EQ(trajectory, nullptr);
    EXPECT_EQ(returned->GetCurrentStepNumber(), stepNumber == 0 ? 1 : stepNumber);
    EXPECT_EQ(returned->GetTrackID(), g4id);
    EXPECT_EQ(returned->GetUserInformation(), information);
    EXPECT_FALSE(mapper.contains(trackId));
    EXPECT_EQ(deletions, 0);
    returned.reset();
    EXPECT_EQ(deletions, 1);
  }
  mapper.beginEvent(1); // Discard the reverse mappings retained for possible CPU-to-GPU returns.
#endif
}
