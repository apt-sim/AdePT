// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/geometry/AdePTGeometryBridge.hh>
#include <AdePT/g4integration/returned_steps/AdePTGeant4Integration.hh>
#include <AdePT/g4integration/tracking_managers/G4HepEmTrackingManagerSpecialized.hh>
#include <AdePT/transport/steps/GPUStep.hh>

#include <VecGeom/management/GeoManager.h>

#include <G4Box.hh>
#include <G4DynamicParticle.hh>
#include <G4Electron.hh>
#include <G4EventManager.hh>
#include <G4Gamma.hh>
#include <G4GeometryTolerance.hh>
#include <G4LogicalVolume.hh>
#include <G4NavigationHistory.hh>
#include <G4NistManager.hh>
#include <G4PVPlacement.hh>
#include <G4Positron.hh>
#include <G4ProductionCutsTable.hh>
#include <G4Region.hh>
#include <G4RunManager.hh>
#include <G4StackManager.hh>
#include <G4Step.hh>
#include <G4SystemOfUnits.hh>
#include <G4TouchableHistory.hh>
#include <G4Track.hh>
#include <G4UserSteppingAction.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPhysicsList.hh>

#include <gtest/gtest.h>

#include <initializer_list>
#include <memory>
#include <span>

namespace {

class HandoverDetector : public G4VUserDetectorConstruction {
public:
  G4VPhysicalVolume *Construct() override
  {
    auto *vacuum = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
    auto *worldLogical =
        new G4LogicalVolume(new G4Box("handover_world", 10 * mm, 10 * mm, 10 * mm), vacuum, "handover_world");
    world            = new G4PVPlacement(nullptr, {}, worldLogical, "handover_world", nullptr, false, 0);
    auto *cpuLogical = new G4LogicalVolume(new G4Box("handover_cpu", 1 * mm, 1 * mm, 1 * mm), vacuum, "handover_cpu");
    cpu              = new G4PVPlacement(nullptr, {}, cpuLogical, "handover_cpu", worldLogical, false, 0);
    auto *cpuRegion  = new G4Region("handover_cpu_region");
    cpuRegion->AddRootLogicalVolume(cpuLogical);
    return world;
  }

  G4VPhysicalVolume *world = nullptr;
  G4VPhysicalVolume *cpu   = nullptr;
};

class HandoverPhysics : public G4VUserPhysicsList {
public:
  void ConstructParticle() override
  {
    G4Electron::Definition();
    G4Positron::Definition();
    G4Gamma::Definition();
  }

  void ConstructProcess() override { AddTransportation(); }

  void SetCuts() override
  {
    SetCutValue(1 * mm, "gamma");
    SetCutValue(1 * mm, "e-");
    SetCutValue(1 * mm, "e+");
  }
};

class HandoverSteppingAction : public G4UserSteppingAction {
public:
  void UserSteppingAction(G4Step const *) override { ++steps; }

  int steps = 0;
};

struct ReturnPath {
  ParticleType particle;
  bool direct;
};

constexpr ReturnPath returnPaths[] = {{ParticleType::Electron, false},
                                      {ParticleType::Positron, false},
                                      {ParticleType::Gamma, false},
                                      {ParticleType::Gamma, true}};

class ReturnedTrackBoundary : public testing::Test {
protected:
  static void SetUpTestSuite()
  {
    // HepEm's master data is shared: initialize one manager for all test cases.
    runManager = std::make_unique<G4RunManager>();
    runManager->SetVerboseLevel(0);
    detector = new HandoverDetector;
    runManager->SetUserInitialization(detector);
    runManager->SetUserInitialization(new HandoverPhysics);
    steppingAction = new HandoverSteppingAction;
    runManager->SetUserAction(steppingAction);
    runManager->Initialize();
    G4ProductionCutsTable::GetProductionCutsTable()->UpdateCoupleTable(detector->world);
    AdePTGeometryBridge::CreateVecGeomWorld(detector->world);

    hepEm = std::make_unique<G4HepEmTrackingManagerSpecialized>();
    hepEm->SetVerbose(0);
    hepEm->ResetFinishEventOnCPUSize(1);
    hepEm->SetGPURegions({detector->world->GetLogicalVolume()->GetRegion()});
    for (auto *particle : std::initializer_list<G4ParticleDefinition *>{
             G4Electron::Definition(), G4Positron::Definition(), G4Gamma::Definition()}) {
      hepEm->PreparePhysicsTable(*particle);
      hepEm->BuildPhysicsTable(*particle);
    }
  }

  static void TearDownTestSuite()
  {
    hepEm.reset();
    vecgeom::GeoManager::Instance().Clear();
    runManager.reset();
    detector       = nullptr;
    steppingAction = nullptr;
  }

  void SetUp() override
  {
    hepEm->SetFinishEventOnCPU(0, -1);
    steppingAction->steps = 0;
    integration.SetHepEmTrackingManager(hepEm.get());
    integration.GetHostTrackDataMapper().beginEvent(0);
  }

  std::unique_ptr<G4Track> ReturnFromGPU(ReturnPath path, bool boundary, G4ThreeVector const &position,
                                         bool useHostData = true, short process = kAdePTOutOfGPURegionProcess)
  {
    auto const *world = vecgeom::GeoManager::Instance().GetWorld();
    vecgeom::NavigationState before;
    before.Push(world);
    before.SetBoundaryState(false);
    auto after = before;
    after.Push(world->GetLogicalVolume()->GetDaughters()[0]);
    after.SetBoundaryState(boundary);

    GPUStep step;
    step.fTrackID                         = nextTrackID++;
    step.fParticleType                    = path.particle;
    step.fStepLimProcessId                = process;
    step.fStepCounter                     = 7;
    step.fHasHostData                     = useHostData;
    step.fStepLength                      = 1 * mm;
    step.fPreStepPoint.fNavigationState   = before;
    step.fPreStepPoint.fPosition          = {2 * mm, 0., 0.};
    step.fPreStepPoint.fMomentumDirection = {1., 0., 0.};
    step.fPreStepPoint.fEKin              = 10 * MeV;
    step.fPostStepPoint                   = step.fPreStepPoint;
    step.fPostStepPoint.fNavigationState  = after;
    step.fPostStepPoint.fPosition         = {position.x(), position.y(), position.z()};
    if (useHostData) integration.GetHostTrackDataMapper().create(step.fTrackID);

    // Exercise both production GPU-return paths and Geant4's real stack. The
    // returned track must obtain its state from this record, not test metadata.
    if (path.direct) {
      integration.ReturnDeferredTrack(std::span<const GPUStep>(&step, 1), useHostData);
    } else {
      integration.ProcessGPUStep(std::span<const GPUStep>(&step, 1), false, false, useHostData);
    }
    G4VTrajectory *trajectory = nullptr;
    std::unique_ptr<G4Track> track(G4EventManager::GetEventManager()->GetStackManager()->PopNextTrack(&trajectory));
    EXPECT_EQ(trajectory, nullptr);
    return track;
  }

  static std::unique_ptr<G4Track> MakeInitialTrack(G4ParticleDefinition *particle)
  {
    auto *dynamic = new G4DynamicParticle(particle, G4ThreeVector(1, 0, 0), 10 * MeV);
    auto track    = std::make_unique<G4Track>(dynamic, 0., G4ThreeVector(2 * mm, 0., 0.));
    track->SetTrackID(1);
    G4NavigationHistory history;
    history.SetFirstEntry(detector->world);
    G4TouchableHandle touchable(new G4TouchableHistory(history));
    track->SetTouchableHandle(touchable);
    track->SetNextTouchableHandle(touchable);
    return track;
  }

  AdePTGeant4Integration integration;
  uint64_t nextTrackID = 1;
  inline static std::unique_ptr<G4RunManager> runManager;
  inline static std::unique_ptr<G4HepEmTrackingManagerSpecialized> hepEm;
  inline static HandoverDetector *detector             = nullptr;
  inline static HandoverSteppingAction *steppingAction = nullptr;
};

TEST_F(ReturnedTrackBoundary, ReturnedTrackMakesCPUProgressBeforeReoffload)
{
  const auto tolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
  // The GPU supplied a CPU destination touchable, but Geant4 locates this
  // coordinate back in GPU world. Returning immediately repeats the handover
  // forever. The CPU must complete a real step before allowing another offload.
  const G4ThreeVector position(1 * mm + 100 * tolerance, 0, 0);
  for (auto path : returnPaths) {
    for (bool boundary : {false, true}) {
      for (bool useHostData : {false, true}) {
        SCOPED_TRACE(testing::Message() << "particle=" << static_cast<int>(path.particle) << " direct=" << path.direct
                                        << " boundary=" << boundary << " hostData=" << useHostData);
        steppingAction->steps = 0;
        auto track            = ReturnFromGPU(path, boundary, position, useHostData);
        ASSERT_NE(track, nullptr);
        ASSERT_EQ(track->GetStep(), nullptr);
        ASSERT_EQ(track->GetNextVolume(), detector->cpu);
        const auto initialStep = track->GetCurrentStepNumber();

        hepEm->HandOverOneTrack(track.get());

        EXPECT_GT(track->GetCurrentStepNumber(), initialStep);
        EXPECT_GT(steppingAction->steps, 0);
        // There are no further GPU/CPU surfaces in the outgoing direction, so
        // the real CPU step exits the vacuum world rather than reoffloading.
        EXPECT_EQ(track->GetTrackStatus(), fStopAndKill);
        ASSERT_NE(track->GetStep(), nullptr);
        EXPECT_EQ(track->GetStep()->GetPostStepPoint()->GetStepStatus(), fWorldBoundary);
      }
    }
  }
}

TEST_F(ReturnedTrackBoundary, InitialGPUOffloadKeepsInteriorNonboundary)
{
  for (auto *particle : std::initializer_list<G4ParticleDefinition *>{
           G4Electron::Definition(), G4Positron::Definition(), G4Gamma::Definition()}) {
    SCOPED_TRACE(particle->GetParticleName());
    auto track = MakeInitialTrack(particle);
    EXPECT_EQ(track->GetCurrentStepNumber(), 0);

    hepEm->HandOverOneTrack(track.get());

    EXPECT_EQ(track->GetTrackStatus(), fAlive);
    EXPECT_EQ(track->GetCurrentStepNumber(), 0);
    ASSERT_NE(track->GetStep(), nullptr);
    EXPECT_EQ(track->GetStep()->GetPostStepPoint()->GetStepStatus(), fUndefined);
  }
}

TEST_F(ReturnedTrackBoundary, CPUStepSuppliesTheNewBoundaryStatus)
{
  auto track = ReturnFromGPU({ParticleType::Gamma, false}, false, {});
  ASSERT_NE(track, nullptr);
  const auto initialStep = track->GetCurrentStepNumber();

  hepEm->HandOverOneTrack(track.get());

  EXPECT_EQ(track->GetTrackStatus(), fAlive);
  EXPECT_GT(track->GetCurrentStepNumber(), initialStep);
  EXPECT_EQ(track->GetNextVolume(), detector->world);
  ASSERT_GT(steppingAction->steps, 0);
  ASSERT_NE(track->GetStep(), nullptr);
  EXPECT_EQ(track->GetStep()->GetPostStepPoint()->GetStepStatus(), fGeomBoundary);
}

TEST_F(ReturnedTrackBoundary, InteriorFinishOnCPUCompletesNormally)
{
  for (auto path : returnPaths) {
    SCOPED_TRACE(testing::Message() << "particle=" << static_cast<int>(path.particle) << " direct=" << path.direct);
    auto track = ReturnFromGPU(path, false, {}, true, kAdePTFinishOnCPUProcess);
    ASSERT_NE(track, nullptr);
    EXPECT_EQ(track->GetTrackStatus(), fStopButAlive);
    ASSERT_EQ(track->GetStep(), nullptr);
    const auto initialStep = track->GetCurrentStepNumber();

    // AdePTTrackingManager::ProcessTrack resets this handover sentinel before
    // asking HepEm to complete an event on the CPU.
    track->SetTrackStatus(fAlive);
    hepEm->SetFinishEventOnCPU(0, 0);
    hepEm->HandOverOneTrack(track.get());

    EXPECT_EQ(track->GetTrackStatus(), fStopAndKill);
    EXPECT_GT(track->GetCurrentStepNumber(), initialStep);
    ASSERT_NE(track->GetStep(), nullptr);
    EXPECT_EQ(track->GetStep()->GetPostStepPoint()->GetStepStatus(), fWorldBoundary);
  }
}

} // namespace
