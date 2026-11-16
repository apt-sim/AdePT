// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

// Handoff validation (/adept/validateHandoff): each outcome is produced through the real returned-track path, from a
// GPU step that leaves the GPU regions on a boundary.

#include <AdePT/g4integration/geometry/AdePTGeometryBridge.hh>
#include <AdePT/g4integration/returned_steps/AdePTGeant4Integration.hh>
#include <AdePT/g4integration/returned_steps/HandoffValidator.hh>
#include <AdePT/g4integration/tracking_managers/G4HepEmTrackingManagerSpecialized.hh>
#include <AdePT/transport/steps/GPUStep.hh>

#include <VecGeom/management/GeoManager.h>

#include <G4Box.hh>
#include <G4Electron.hh>
#include <G4EventManager.hh>
#include <G4Gamma.hh>
#include <G4GeometryManager.hh>
#include <G4LogicalVolume.hh>
#include <G4NistManager.hh>
#include <G4PVPlacement.hh>
#include <G4PVReplica.hh>
#include <G4Positron.hh>
#include <G4ProductionCutsTable.hh>
#include <G4Region.hh>
#include <G4RunManager.hh>
#include <G4StackManager.hh>
#include <G4SystemOfUnits.hh>
#include <G4Track.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPhysicsList.hh>

#include <gtest/gtest.h>

#include <initializer_list>
#include <iterator>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

// World (GPU region) with, in mm:
// - cpu: box [-1, 1]^3 in a CPU region, with two daughters:
//   - x: starts on the -x face of cpu (x in [-1, -0.6], |y|, |z| < 0.2), with a daughter x2 starting on the same
//     face (x in [-1, -0.9], |y|, |z| < 0.1)
//   - y: starts 5e-8 mm behind that face (x in [-1 + 5e-8, -0.8 + 5e-8], y in [0.57, 0.67]), within the handoff push
// - ovl: overlaps cpu (x in [-1.3, -0.7], y in [0.85, 0.95], |z| < 0.3)
// - rep: box [3, 5] x [-1, 1]^2, in the CPU region, divided into 4 replicas along x
class HandoffDetector : public G4VUserDetectorConstruction {
public:
  G4VPhysicalVolume *Construct() override
  {
    auto *vacuum = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
    auto box     = [&](char const *name, double dx, double dy, double dz) {
      return new G4LogicalVolume(new G4Box(name, dx, dy, dz), vacuum, name);
    };
    auto *worldLV = box("hv_world", 10 * mm, 10 * mm, 10 * mm);
    world         = new G4PVPlacement(nullptr, {}, worldLV, "hv_world", nullptr, false, 0);
    auto *cpuLV   = box("hv_cpu", 1 * mm, 1 * mm, 1 * mm);
    new G4PVPlacement(nullptr, {}, cpuLV, "hv_cpu", worldLV, false, 0);
    auto *xLV = box("hv_x", 0.2 * mm, 0.2 * mm, 0.2 * mm);
    new G4PVPlacement(nullptr, {-0.8 * mm, 0, 0}, xLV, "hv_x", cpuLV, false, 0);
    new G4PVPlacement(nullptr, {-0.15 * mm, 0, 0}, box("hv_x2", 0.05 * mm, 0.1 * mm, 0.1 * mm), "hv_x2", xLV, false, 0);
    new G4PVPlacement(nullptr, {(-0.9 + 5e-8) * mm, 0.62 * mm, 0}, box("hv_y", 0.1 * mm, 0.05 * mm, 0.2 * mm), "hv_y",
                      cpuLV, false, 0);
    new G4PVPlacement(nullptr, {-1 * mm, 0.9 * mm, 0}, box("hv_ovl", 0.3 * mm, 0.05 * mm, 0.3 * mm), "hv_ovl", worldLV,
                      false, 0);
    auto *repLV = box("hv_rep", 1 * mm, 1 * mm, 1 * mm);
    new G4PVPlacement(nullptr, {4 * mm, 0, 0}, repLV, "hv_rep", worldLV, false, 0);
    new G4PVReplica("hv_slice", box("hv_slice", 0.25 * mm, 1 * mm, 1 * mm), repLV, kXAxis, 4, 0.5 * mm);
    auto *cpuRegion = new G4Region("hv_cpu_region");
    cpuRegion->AddRootLogicalVolume(cpuLV);
    cpuRegion->AddRootLogicalVolume(repLV);
    return world;
  }

  G4VPhysicalVolume *world = nullptr;
};

class HandoffPhysics : public G4VUserPhysicsList {
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

/// GPU navigation state of the world followed by the given daughter indices.
vecgeom::NavigationState State(std::initializer_list<int> daughters, bool boundary)
{
  auto const *volume = vecgeom::GeoManager::Instance().GetWorld();
  vecgeom::NavigationState state;
  state.Push(volume);
  for (int index : daughters) {
    volume = volume->GetLogicalVolume()->GetDaughters()[index];
    state.Push(volume);
  }
  state.SetBoundaryState(boundary);
  return state;
}

struct Handoff {
  char const *name;
  std::initializer_list<int> from, to;
  G4ThreeVector end, direction;
  HandoffOutcome outcome;
  char const *locatedVolume; ///< volume of the returned track's touchable (the GPU path), as a check of the setup
};

const Handoff kHandoffs[] = {
    {"confirmed", {}, {0}, {-1 * mm, 0.5 * mm, 0}, {1, 0, 0}, HandoffOutcome::Confirmed, "hv_cpu"},
    {"coincident surface", {}, {0}, {-1 * mm, 0.15 * mm, 0}, {1, 0, 0}, HandoffOutcome::CoincidentSurface, "hv_cpu"},
    // Two nested volumes start on the crossed surface: Geant4 enters both.
    {"nested coincident surfaces", {}, {0}, {-1 * mm, 0, 0}, {1, 0, 0}, HandoffOutcome::CoincidentSurface, "hv_cpu"},
    {"push crossing", {}, {0}, {-1 * mm, 0.62 * mm, 0}, {1, 0, 0}, HandoffOutcome::PushCrossing, "hv_cpu"},
    {"direction reversed", {}, {0}, {-1 * mm, 0.5 * mm, 0}, {-1, 0, 0}, HandoffOutcome::DirectionReversed, "hv_cpu"},
    // A real path disagreement: the GPU reached cpu 1 um before its surface, Geant4 relocates the track in the world.
    {"band error", {}, {0}, {-1.001 * mm, 0.5 * mm, 0}, {1, 0, 0}, HandoffOutcome::BandError, "hv_cpu"},
    // Band error proven to come from an overlap: the end point is strictly inside ovl, the volume being left.
    {"overlap", {1}, {0}, {-1 * mm, 0.9 * mm, 0}, {1, 0, 0}, HandoffOutcome::Overlap, "hv_cpu"},
    // Geant4 locates a replica slice: its solid is not checked.
    {"replicas", {}, {2}, {3 * mm, 0, 0}, {1, 0, 0}, HandoffOutcome::Replicas, "hv_rep"},
};

class HandoffValidation : public testing::Test {
protected:
  static void SetUpTestSuite()
  {
    runManager = std::make_unique<G4RunManager>();
    runManager->SetVerboseLevel(0);
    detector = new HandoffDetector;
    runManager->SetUserInitialization(detector);
    runManager->SetUserInitialization(new HandoffPhysics);
    runManager->Initialize();
    G4ProductionCutsTable::GetProductionCutsTable()->UpdateCoupleTable(detector->world);
    AdePTGeometryBridge::CreateVecGeomWorld(detector->world);
    // As at the start of a run: Geant4 needs the voxels of the replicas to navigate.
    G4GeometryManager::GetInstance()->CloseGeometry();

    hepEm = std::make_unique<G4HepEmTrackingManagerSpecialized>();
    hepEm->SetVerbose(0);
    hepEm->ResetFinishEventOnCPUSize(1);
    for (auto *particle : std::initializer_list<G4ParticleDefinition *>{
             G4Electron::Definition(), G4Positron::Definition(), G4Gamma::Definition()}) {
      hepEm->PreparePhysicsTable(*particle);
      hepEm->BuildPhysicsTable(*particle);
    }
  }

  static void TearDownTestSuite()
  {
    hepEm.reset();
    G4GeometryManager::GetInstance()->OpenGeometry();
    vecgeom::GeoManager::Instance().Clear();
    runManager.reset();
    detector = nullptr;
  }

  void SetUp() override
  {
    integration.SetHepEmTrackingManager(hepEm.get());
    integration.GetHostTrackDataMapper().beginEvent(0);
  }

  /// Return a track to Geant4 through the production path (`direct`: ReturnDeferredTrack, gammas only).
  std::unique_ptr<G4Track> Return(Handoff const &handoff, ParticleType particle, bool direct, bool boundary = true,
                                  short process = kAdePTOutOfGPURegionProcess)
  {
    GPUStep step;
    step.fTrackID                         = nextTrackID++;
    step.fParticleType                    = particle;
    step.fStepLimProcessId                = process;
    step.fStepCounter                     = 7;
    step.fHasHostData                     = true;
    step.fStepLength                      = 0.5 * mm;
    step.fPreStepPoint.fNavigationState   = State(handoff.from, false);
    step.fPreStepPoint.fPosition          = {handoff.end.x() - 0.5 * mm * handoff.direction.x(),
                                             handoff.end.y() - 0.5 * mm * handoff.direction.y(),
                                             handoff.end.z() - 0.5 * mm * handoff.direction.z()};
    step.fPreStepPoint.fMomentumDirection = {handoff.direction.x(), handoff.direction.y(), handoff.direction.z()};
    step.fPreStepPoint.fEKin              = 10 * MeV;
    step.fPostStepPoint                   = step.fPreStepPoint;
    step.fPostStepPoint.fNavigationState  = State(handoff.to, boundary);
    step.fPostStepPoint.fPosition         = {handoff.end.x(), handoff.end.y(), handoff.end.z()};
    integration.GetHostTrackDataMapper().create(step.fTrackID);
    if (direct)
      integration.ReturnDeferredTrack(std::span<const GPUStep>(&step, 1), true);
    else
      integration.ProcessGPUStep(std::span<const GPUStep>(&step, 1), false, false, true);
    G4VTrajectory *trajectory = nullptr;
    return std::unique_ptr<G4Track>(G4EventManager::GetEventManager()->GetStackManager()->PopNextTrack(&trajectory));
  }

  AdePTGeant4Integration integration;
  uint64_t nextTrackID = 1;
  inline static std::unique_ptr<G4RunManager> runManager;
  inline static std::unique_ptr<G4HepEmTrackingManagerSpecialized> hepEm;
  inline static HandoffDetector *detector = nullptr;
};

TEST_F(HandoffValidation, EachOutcomeThroughTheReturnedTrackPath)
{
  integration.EnableHandoffValidation(1);
  auto const &counts = integration.GetHandoffValidator()->GetCounts();
  struct Path {
    ParticleType particle;
    bool direct;
  };
  for (auto const &handoff : kHandoffs) {
    for (auto path :
         {Path{ParticleType::Electron, false}, Path{ParticleType::Gamma, false}, Path{ParticleType::Gamma, true}}) {
      SCOPED_TRACE(testing::Message() << handoff.name << ", particle " << int(path.particle) << ", direct "
                                      << path.direct);
      const HandoffCounts before = counts;
      auto track                 = Return(handoff, path.particle, path.direct);
      ASSERT_NE(track, nullptr);
      ASSERT_EQ(track->GetNextVolume()->GetName(), handoff.locatedVolume);
      EXPECT_EQ(counts.Total(), before.Total() + 1);
      EXPECT_EQ(counts[handoff.outcome], before[handoff.outcome] + 1)
          << "expected " << HandoffOutcomeName(handoff.outcome);
    }
  }
}

TEST_F(HandoffValidation, OnlyBoundaryExitsFromTheGPURegionsAreChecked)
{
  integration.EnableHandoffValidation(1);
  auto const &counts = integration.GetHandoffValidator()->GetCounts();
  auto track         = Return(kHandoffs[0], ParticleType::Electron, false, /*boundary=*/false);
  ASSERT_NE(track, nullptr);
  track = Return(kHandoffs[0], ParticleType::Electron, false, true, kAdePTFinishOnCPUProcess);
  ASSERT_NE(track, nullptr);
  EXPECT_EQ(counts.Total(), 0u);
}

TEST_F(HandoffValidation, OffByDefault)
{
  EXPECT_EQ(integration.GetHandoffValidator(), nullptr);
  auto track = Return(kHandoffs[0], ParticleType::Electron, false);
  ASSERT_NE(track, nullptr);
  EXPECT_EQ(track->GetNextVolume()->GetName(), "hv_cpu");
}

TEST(HandoffCounts, VerdictNeedsAConclusiveHandoffAndNoUnexpectedOne)
{
  using Verdict = HandoffCounts::Verdict;
  HandoffCounts counts;
  EXPECT_EQ(counts.GetVerdict(), Verdict::NoConclusiveHandoff);
  counts[HandoffOutcome::Replicas] = 100;
  counts[HandoffOutcome::Overlap]  = 3;
  EXPECT_EQ(counts.GetVerdict(), Verdict::NoConclusiveHandoff);
  counts[HandoffOutcome::CoincidentSurface] = 1;
  EXPECT_EQ(counts.GetVerdict(), Verdict::Passed);
  for (auto unexpected : {HandoffOutcome::BandError, HandoffOutcome::DirectionReversed, HandoffOutcome::OtherPath}) {
    HandoffCounts failed = counts;
    failed[unexpected]   = 1;
    EXPECT_EQ(failed.GetVerdict(), Verdict::Failed) << HandoffOutcomeName(unexpected);
  }
}

TEST(HandoffRunTotals, SumsAllThreadsThenStartsOver)
{
  HandoffRunTotals totals;
  HandoffCounts a, b, total;
  a[HandoffOutcome::Confirmed] = 5;
  b[HandoffOutcome::Confirmed] = 2;
  b[HandoffOutcome::BandError] = 1;
  EXPECT_FALSE(totals.Add(a, 2, total));
  ASSERT_TRUE(totals.Add(b, 2, total));
  EXPECT_EQ(total[HandoffOutcome::Confirmed], 7u);
  EXPECT_EQ(total[HandoffOutcome::BandError], 1u);
  EXPECT_FALSE(totals.Add(a, 2, total)); // next run
}

TEST_F(HandoffValidation, EndOfRunPrintsTheVerdictAndResets)
{
  integration.EnableHandoffValidation(1);
  auto *validator = integration.GetHandoffValidator();
  auto track      = Return(kHandoffs[0], ParticleType::Electron, false);
  track           = Return(kHandoffs[std::size(kHandoffs) - 1], ParticleType::Electron, false); // replicas
  HandoffRunTotals totals;
  std::ostringstream out;
  validator->EndOfRun(totals, out);
  EXPECT_NE(out.str().find("HandoffValidation: PASSED (1 of 2 handoffs checked, 0 unexpected)"), std::string::npos)
      << out.str();
  EXPECT_EQ(validator->GetCounts().Total(), 0u);

  std::ostringstream empty;
  validator->EndOfRun(totals, empty);
  EXPECT_NE(empty.str().find("HandoffValidation: NO CONCLUSIVE HANDOFF"), std::string::npos) << empty.str();
}

} // namespace
