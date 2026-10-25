// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/geometry/AdePTGeometryBridge.hh>
#include <AdePT/g4integration/returned_steps/AdePTGeant4Integration.hh>
#include <AdePT/g4integration/returned_steps/HostTrackDataMapper.hh>
#include <AdePT/g4integration/tracking_managers/G4HepEmTrackingManagerSpecialized.hh>
#include <AdePT/transport/steps/GPUStep.hh>

#include <VecGeom/management/GeoManager.h>

#include <G4Box.hh>
#include <G4Electron.hh>
#include <G4EmProcessSubType.hh>
#include <G4Gamma.hh>
#include <G4GeometryManager.hh>
#include <G4LogicalVolume.hh>
#include <G4NistManager.hh>
#include <G4PVPlacement.hh>
#include <G4Positron.hh>
#include <G4Step.hh>
#include <G4SystemOfUnits.hh>
#include <G4TouchableHistory.hh>
#include <G4Track.hh>
#include <G4TransportationManager.hh>
#include <G4VProcess.hh>
#include <G4VSensitiveDetector.hh>

#include <span>
#include <vector>

#include <gtest/gtest.h>

namespace {

struct CreatorGeometryCleanup {
  G4LogicalVolume *logical = nullptr;

  ~CreatorGeometryCleanup()
  {
    if (logical != nullptr) logical->SetSensitiveDetector(nullptr);
    G4GeometryManager::GetInstance()->OpenGeometry();
    vecgeom::GeoManager::Instance().Clear();
  }
};

class CreatorProcessSensitiveDetector : public G4VSensitiveDetector {
public:
  CreatorProcessSensitiveDetector() : G4VSensitiveDetector("returned_creator_process_sd") {}

  int calls = 0;
  std::vector<int> creatorSubTypes;

protected:
  G4bool ProcessHits(G4Step *step, G4TouchableHistory *) override
  {
    ++calls;
    for (auto const *secondary : *step->GetSecondary()) {
      auto const *creator = secondary->GetCreatorProcess();
      creatorSubTypes.push_back(creator != nullptr ? creator->GetProcessSubType() : -1);
    }
    return true;
  }
};

} // namespace

TEST(AdePTGeant4Integration, ReplayedSecondariesUseParentCreatorProcessFamily)
{
  CreatorGeometryCleanup cleanup;
  G4GeometryManager::GetInstance()->OpenGeometry();
  vecgeom::GeoManager::Instance().Clear();
  G4Electron::Definition();
  G4Positron::Definition();
  G4Gamma::Definition();

  auto *material = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  ASSERT_NE(material, nullptr);
  auto *solid     = new G4Box("creator_world_solid", 10.0 * mm, 10.0 * mm, 10.0 * mm);
  auto *logical   = new G4LogicalVolume(solid, material, "creator_world_lv");
  auto *world     = new G4PVPlacement(nullptr, G4ThreeVector(), logical, "creator_world", nullptr, false, 0);
  cleanup.logical = logical;
  CreatorProcessSensitiveDetector detector;
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

  struct Secondary {
    ParticleType type;
    short creatorIndex;
    int expectedSubType;
  };
  struct Case {
    char const *name;
    ParticleType parentType;
    short parentProcessIndex;
    std::vector<Secondary> secondaries;
  };
  const std::vector<Case> cases = {
      {"gamma conversion",
       ParticleType::Gamma,
       0,
       {{ParticleType::Electron, 0, fGammaConversion}, {ParticleType::Positron, 0, fGammaConversion}}},
      {"Compton", ParticleType::Gamma, 1, {{ParticleType::Electron, 1, fComptonScattering}}},
      {"photoelectric", ParticleType::Gamma, 2, {{ParticleType::Electron, 2, fPhotoElectricEffect}}},
      {"electron bremsstrahlung", ParticleType::Electron, 1, {{ParticleType::Gamma, 1, fBremsstrahlung}}},
      {"positron bremsstrahlung", ParticleType::Positron, 1, {{ParticleType::Gamma, 1, fBremsstrahlung}}},
      // Stopped annihilation can follow ionization in the same record; its photons have a different creator index.
      {"positron ionization and stopped annihilation",
       ParticleType::Positron,
       0,
       {{ParticleType::Electron, 0, fIonisation},
        {ParticleType::Gamma, 2, fAnnihilation},
        {ParticleType::Gamma, 2, fAnnihilation}}}};

  for (auto const &testCase : cases) {
    SCOPED_TRACE(testCase.name);
    detector.calls = 0;
    detector.creatorSubTypes.clear();

    constexpr uint64_t parentId = 100;
    mapper.create(parentId);
    std::vector<GPUStep> steps(1 + testCase.secondaries.size());
    auto &parent                            = steps.front();
    parent.fTrackID                         = parentId;
    parent.fParticleType                    = testCase.parentType;
    parent.fStepLimProcessId                = testCase.parentProcessIndex;
    parent.fStepCounter                     = 1;
    parent.fLastStepOfTrack                 = true;
    parent.fHasHostData                     = true;
    parent.fNumSecondaries                  = static_cast<unsigned char>(testCase.secondaries.size());
    parent.fStepLength                      = 1.0 * mm;
    parent.fTotalEnergyDeposit              = 0.1 * MeV;
    parent.fPreStepPoint.fNavigationState   = navigation;
    parent.fPreStepPoint.fPosition          = {0.0, 0.0, 0.0};
    parent.fPreStepPoint.fMomentumDirection = {0.0, 0.0, 1.0};
    parent.fPreStepPoint.fEKin              = 10.0 * MeV;
    parent.fPostStepPoint                   = parent.fPreStepPoint;
    parent.fPostStepPoint.fPosition         = {0.0, 0.0, 1.0 * mm};
    parent.fPostStepPoint.fEKin             = 5.0 * MeV;

    std::vector<int> expectedSubTypes;
    for (std::size_t i = 0; i < testCase.secondaries.size(); ++i) {
      auto const &description        = testCase.secondaries[i];
      auto &secondary                = steps[i + 1];
      secondary.fTrackID             = parentId + i + 1;
      secondary.fParentID            = parentId;
      secondary.fParticleType        = description.type;
      secondary.fStepLimProcessId    = description.creatorIndex;
      secondary.fHasHostData         = true;
      secondary.fPostStepPoint       = parent.fPostStepPoint;
      secondary.fPostStepPoint.fEKin = 0.5 * MeV;
      secondary.fPreStepPoint        = secondary.fPostStepPoint;
      expectedSubTypes.push_back(description.expectedSubType);
    }

    integration.ProcessGPUStep(std::span<const GPUStep>(steps), false, false, true);
    EXPECT_EQ(detector.calls, 1);
    EXPECT_EQ(detector.creatorSubTypes, expectedSubTypes);
    EXPECT_FALSE(mapper.contains(parentId));
    for (std::size_t i = 1; i < steps.size(); ++i) {
      EXPECT_TRUE(mapper.contains(steps[i].fTrackID));
      mapper.removeTrack(steps[i].fTrackID);
    }
  }
}
