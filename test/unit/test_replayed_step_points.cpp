// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/returned_steps/HostTrackDataMapper.hh>
#include <AdePT/g4integration/tracking_managers/G4HepEmTrackingManagerSpecialized.hh>
#include <AdePT/transport/steps/GPUStep.hh>

#include <G4Event.hh>
#include <G4EventManager.hh>
#include <G4Track.hh>

#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <vector>

#define private public
#include <AdePT/g4integration/returned_steps/AdePTGeant4Integration.hh>
#undef private

#include <G4Box.hh>
#include <G4Electron.hh>
#include <G4Gamma.hh>
#include <G4LogicalVolume.hh>
#include <G4NavigationHistory.hh>
#include <G4NistManager.hh>
#include <G4PVPlacement.hh>
#include <G4Positron.hh>
#include <G4Step.hh>
#include <G4SystemOfUnits.hh>
#include <G4TouchableHistory.hh>
#include <G4TransportationManager.hh>

#include <gtest/gtest.h>

TEST(AdePTGeant4Integration, ReplayedStepPointsKeepParticleMassAndRecordedWeight)
{
  auto *material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  ASSERT_NE(material, nullptr);
  auto solid   = std::make_unique<G4Box>("replay_points_world_solid", 10.0 * mm, 10.0 * mm, 10.0 * mm);
  auto logical = std::make_unique<G4LogicalVolume>(solid.get(), material, "replay_points_world_lv");
  auto world = std::make_unique<G4PVPlacement>(nullptr, G4ThreeVector(), logical.get(), "replay_points_world", nullptr,
                                               false, 0);
  G4TransportationManager::GetTransportationManager()->SetWorldForTracking(world.get());
  G4NavigationHistory history;
  history.SetFirstEntry(world.get());
  G4TouchableHandle preTouchable(new G4TouchableHistory(history));
  G4TouchableHandle postTouchable(new G4TouchableHistory(history));

  G4HepEmTrackingManagerSpecialized hepEm;
  AdePTGeant4Integration integration;
  integration.SetHepEmTrackingManager(&hepEm);

  const auto makeTrack = [](G4ParticleDefinition *definition) {
    return std::make_unique<G4Track>(new G4DynamicParticle(definition, G4ThreeVector(0., 0., 1.), 0.), 0.,
                                     G4ThreeVector());
  };
  auto electron = makeTrack(G4Electron::Definition());
  auto gamma    = makeTrack(G4Gamma::Definition());
  auto positron = makeTrack(G4Positron::Definition());
  G4Step step;
  for (auto *point : {step.GetPreStepPoint(), step.GetPostStepPoint()}) {
    point->SetMass(7.0 * MeV);
    point->SetWeight(99.);
  }

  struct Replay {
    G4Track *track;
    ParticleType type;
    G4double preEnergy;
    G4double postEnergy;
    float weight;
  };
  const std::array<Replay, 4> replays{{
      {electron.get(), ParticleType::Electron, 1.0 * MeV, 0.75 * MeV, 2.5f},
      {gamma.get(), ParticleType::Gamma, 3.0 * MeV, 2.0 * MeV, 8.0f},
      {positron.get(), ParticleType::Positron, 0.2 * MeV, 0.0 * MeV, 0.25f},
      {electron.get(), ParticleType::Electron, 0.5 * MeV, 0.25 * MeV, 1.0f},
  }};

  // The shared step points must be refreshed when species or statistical weights change.
  G4double previousWeight = 99.;
  for (auto const &replay : replays) {
    SCOPED_TRACE(static_cast<int>(replay.type));
    step.SetTrack(replay.track);
    // ProcessGPUStep calls FillG4Step before FillG4Track updates this reused track.
    replay.track->SetWeight(previousWeight);

    GPUStep gpuStep;
    gpuStep.fParticleType                     = replay.type;
    gpuStep.fStepCounter                      = 1;
    gpuStep.fStepLimProcessId                 = kAdePTTransportationProcess;
    gpuStep.fTrackWeight                      = replay.weight;
    gpuStep.fPreStepPoint.fPosition           = {0., 0., 0.};
    gpuStep.fPreStepPoint.fMomentumDirection  = {1., 0., 0.};
    gpuStep.fPreStepPoint.fEKin               = replay.preEnergy;
    gpuStep.fPostStepPoint.fPosition          = {1.0 * mm, 0., 0.};
    gpuStep.fPostStepPoint.fMomentumDirection = {0., 0., 1.};
    gpuStep.fPostStepPoint.fEKin              = replay.postEnergy;

    integration.FillG4Step(&gpuStep, &step, preTouchable, postTouchable, fUndefined, fUndefined, false, false);

    const G4double mass   = replay.track->GetParticleDefinition()->GetPDGMass();
    const auto checkPoint = [&](G4StepPoint const *point, G4double kineticEnergy, G4ThreeVector direction) {
      EXPECT_DOUBLE_EQ(point->GetMass(), mass);
      EXPECT_DOUBLE_EQ(point->GetWeight(), static_cast<G4double>(replay.weight));
      EXPECT_DOUBLE_EQ(point->GetTotalEnergy(), kineticEnergy + mass);
      const G4ThreeVector momentum = std::sqrt(kineticEnergy * (kineticEnergy + 2. * mass)) * direction;
      EXPECT_NEAR(point->GetMomentum().x(), momentum.x(), 1.e-12 * MeV);
      EXPECT_NEAR(point->GetMomentum().y(), momentum.y(), 1.e-12 * MeV);
      EXPECT_NEAR(point->GetMomentum().z(), momentum.z(), 1.e-12 * MeV);
    };
    {
      SCOPED_TRACE("pre-step point");
      checkPoint(step.GetPreStepPoint(), replay.preEnergy, G4ThreeVector(1., 0., 0.));
    }
    {
      SCOPED_TRACE("post-step point");
      checkPoint(step.GetPostStepPoint(), replay.postEnergy, G4ThreeVector(0., 0., 1.));
    }
    previousWeight = replay.weight;
  }
}
