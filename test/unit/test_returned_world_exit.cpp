// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/geometry/AdePTGeometryBridge.hh>
#include <AdePT/g4integration/returned_steps/AdePTGeant4Integration.hh>
#include <AdePT/g4integration/tracking_managers/G4HepEmTrackingManagerSpecialized.hh>
#include <AdePT/transport/steps/GPUStep.hh>

#include <VecGeom/management/GeoManager.h>

#include <G4Box.hh>
#include <G4Electron.hh>
#include <G4Gamma.hh>
#include <G4GeometryManager.hh>
#include <G4LogicalVolume.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4NistManager.hh>
#include <G4PVPlacement.hh>
#include <G4Positron.hh>
#include <G4ProductionCuts.hh>
#include <G4Step.hh>
#include <G4SystemOfUnits.hh>
#include <G4TouchableHistory.hh>
#include <G4Track.hh>
#include <G4TransportationManager.hh>
#include <G4VSensitiveDetector.hh>

#include <span>

#include <gtest/gtest.h>

namespace {

struct WorldExitGeometryCleanup {
  G4LogicalVolume *world    = nullptr;
  G4LogicalVolume *daughter = nullptr;

  ~WorldExitGeometryCleanup()
  {
    for (auto *logical : {world, daughter}) {
      if (logical != nullptr) {
        logical->SetSensitiveDetector(nullptr);
        logical->SetMaterialCutsCouple(nullptr);
      }
    }
    G4GeometryManager::GetInstance()->OpenGeometry();
    vecgeom::GeoManager::Instance().Clear();
  }
};

class WorldExitSensitiveDetector : public G4VSensitiveDetector {
public:
  WorldExitSensitiveDetector() : G4VSensitiveDetector("returned_world_exit_sd") {}

  G4VPhysicalVolume *expectedPre  = nullptr;
  G4VPhysicalVolume *expectedPost = nullptr;
  unsigned int calls              = 0;

protected:
  G4bool ProcessHits(G4Step *step, G4TouchableHistory *) override
  {
    ++calls;
    auto const *pre   = step->GetPreStepPoint();
    auto const *post  = step->GetPostStepPoint();
    auto const *track = step->GetTrack();

    // The touchables remain allocated when their volume is null, as in Geant4.
    EXPECT_NE(pre->GetTouchable(), nullptr);
    EXPECT_NE(post->GetTouchable(), nullptr);
    EXPECT_NE(track->GetTouchable(), nullptr);
    EXPECT_NE(track->GetNextTouchable(), nullptr);
    if (pre->GetTouchable() == nullptr || post->GetTouchable() == nullptr || track->GetTouchable() == nullptr ||
        track->GetNextTouchable() == nullptr) {
      return false;
    }

    EXPECT_EQ(pre->GetPhysicalVolume(), expectedPre);
    EXPECT_EQ(track->GetVolume(), expectedPre);
    EXPECT_EQ(post->GetPhysicalVolume(), expectedPost);
    EXPECT_EQ(track->GetNextVolume(), expectedPost);
    EXPECT_EQ(track->GetTouchable(), pre->GetTouchable());
    EXPECT_EQ(track->GetNextTouchable(), post->GetTouchable());
    EXPECT_NE(pre->GetTouchable(), post->GetTouchable());
    EXPECT_EQ(pre->GetMaterial(), expectedPre->GetLogicalVolume()->GetMaterial());
    EXPECT_EQ(pre->GetMaterialCutsCouple(), expectedPre->GetLogicalVolume()->GetMaterialCutsCouple());

    // G4Transportation clears material and cuts together with the world-exit volume.
    auto const *postLogical = expectedPost != nullptr ? expectedPost->GetLogicalVolume() : nullptr;
    EXPECT_EQ(post->GetMaterial(), postLogical != nullptr ? postLogical->GetMaterial() : nullptr);
    EXPECT_EQ(post->GetMaterialCutsCouple(), postLogical != nullptr ? postLogical->GetMaterialCutsCouple() : nullptr);
    EXPECT_EQ(post->GetStepStatus(), expectedPost != nullptr ? fUndefined : fWorldBoundary);
    return true;
  }
};

} // namespace

TEST(AdePTGeant4Integration, ReplayedWorldExitClearsPostTouchable)
{
  WorldExitGeometryCleanup cleanup;
  G4GeometryManager::GetInstance()->OpenGeometry();
  vecgeom::GeoManager::Instance().Clear();
  G4Electron::Definition();
  G4Positron::Definition();
  G4Gamma::Definition();

  auto *worldMaterial    = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  auto *daughterMaterial = G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
  ASSERT_NE(worldMaterial, nullptr);
  ASSERT_NE(daughterMaterial, nullptr);
  G4ProductionCuts cuts;
  G4MaterialCutsCouple worldCouple(worldMaterial, &cuts);
  G4MaterialCutsCouple daughterCouple(daughterMaterial, &cuts);

  auto *worldSolid   = new G4Box("world_exit_world_solid", 10.0 * mm, 10.0 * mm, 10.0 * mm);
  auto *worldLogical = new G4LogicalVolume(worldSolid, worldMaterial, "world_exit_world_lv");
  auto *world        = new G4PVPlacement(nullptr, G4ThreeVector(), worldLogical, "world_exit_world", nullptr, false, 0);
  cleanup.world      = worldLogical;
  worldLogical->SetMaterialCutsCouple(&worldCouple);
  auto *daughterSolid   = new G4Box("world_exit_daughter_solid", 2.0 * mm, 2.0 * mm, 2.0 * mm);
  auto *daughterLogical = new G4LogicalVolume(daughterSolid, daughterMaterial, "world_exit_daughter_lv");
  auto *daughter =
      new G4PVPlacement(nullptr, G4ThreeVector(), daughterLogical, "world_exit_daughter", worldLogical, false, 0);
  cleanup.daughter = daughterLogical;
  daughterLogical->SetMaterialCutsCouple(&daughterCouple);
  WorldExitSensitiveDetector detector;
  worldLogical->SetSensitiveDetector(&detector);
  daughterLogical->SetSensitiveDetector(&detector);

  G4TransportationManager::GetTransportationManager()->SetWorldForTracking(world);
  G4GeometryManager::GetInstance()->CloseGeometry(true);
  AdePTGeometryBridge::CreateVecGeomWorld(world);
  auto const *vgWorld = vecgeom::GeoManager::Instance().GetWorld();
  ASSERT_NE(vgWorld, nullptr);
  ASSERT_EQ(vgWorld->GetDaughters().size(), 1u);
  vecgeom::NavigationState worldNavigation;
  worldNavigation.Push(vgWorld);
  worldNavigation.SetBoundaryState(false);
  auto daughterNavigation = worldNavigation;
  daughterNavigation.Push(vgWorld->GetDaughters()[0]);
  vecgeom::NavigationState outsideNavigation;
  outsideNavigation.Clear();
  outsideNavigation.SetBoundaryState(true);
  ASSERT_TRUE(outsideNavigation.IsOutside());

  G4HepEmTrackingManagerSpecialized hepEm;
  for (auto const type : {ParticleType::Electron, ParticleType::Positron, ParticleType::Gamma}) {
    SCOPED_TRACE(static_cast<int>(type));
    AdePTGeant4Integration integration;
    integration.SetHepEmTrackingManager(&hepEm);
    unsigned int expectedCalls = detector.calls;

    const auto replay = [&](bool outside) {
      SCOPED_TRACE(outside ? "world exit" : "inside daughter");
      GPUStep step;
      step.fParticleType                    = type;
      step.fStepLimProcessId                = kAdePTTransportationProcess;
      step.fStepCounter                     = 1;
      step.fLastStepOfTrack                 = outside;
      step.fStepLength                      = 1.0 * mm;
      step.fPreStepPoint.fNavigationState   = outside ? worldNavigation : daughterNavigation;
      step.fPreStepPoint.fPosition          = {0.0, 0.0, outside ? 9.0 * mm : 0.0};
      step.fPreStepPoint.fMomentumDirection = {0.0, 0.0, 1.0};
      step.fPreStepPoint.fEKin              = 1.0 * MeV;
      step.fPostStepPoint                   = step.fPreStepPoint;
      step.fPostStepPoint.fNavigationState  = outside ? outsideNavigation : daughterNavigation;
      step.fPostStepPoint.fPosition         = {0.0, 0.0, outside ? 10.0 * mm : 1.0 * mm};
      detector.expectedPre                  = outside ? world : daughter;
      detector.expectedPost                 = outside ? nullptr : daughter;
      integration.ProcessGPUStep(std::span<const GPUStep>(&step, 1), false, false, false);
      EXPECT_EQ(detector.calls, ++expectedCalls);
    };

    replay(true); // The first replay must attach an outside touchable to both the step and track.
    replay(false);
    replay(true);  // Clear a cached post history whose depth is greater than the world's.
    replay(false); // Rebuild the cached inside touchable and restore its material and cuts.
    replay(true);
  }
}
