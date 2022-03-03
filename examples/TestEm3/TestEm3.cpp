// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "TestEm3.h"

#include <AdePT/ArgParser.h>
#include <CopCore/SystemOfUnits.h>

#include <G4NistManager.hh>
#include <G4Material.hh>

#include <G4Box.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>

#include <G4ParticleTable.hh>
#include <G4Electron.hh>
#include <G4Positron.hh>
#include <G4Gamma.hh>
#include <G4Proton.hh>

#include <G4ProductionCuts.hh>
#include <G4Region.hh>
#include <G4ProductionCutsTable.hh>
#include <G4TransportationManager.hh>

#include <G4EmParameters.hh>

#include <G4SystemOfUnits.hh>

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/volumes/UnplacedBox.h>

#include <iostream>

const char *WorldMaterial    = "G4_Galactic";
const char *GapMaterial      = "G4_Pb";
const char *AbsorberMaterial = "G4_lAr";

enum MaterialCutCouples {
  WorldMC = 0,
  GapMC,
  AbsorberMC,
};

constexpr G4double ProductionCut = 0.7 * mm;

constexpr double CalorSizeYZ       = 40 * copcore::units::cm;
constexpr int NbOfLayers           = 50;
constexpr int NbOfAbsorbers        = 2;
constexpr double GapThickness      = 2.3 * copcore::units::mm;
constexpr double AbsorberThickness = 5.7 * copcore::units::mm;

constexpr double LayerThickness = GapThickness + AbsorberThickness;
constexpr double CalorThickness = NbOfLayers * LayerThickness;

constexpr double WorldSizeX  = 1.2 * CalorThickness;
constexpr double WorldSizeYZ = 1.2 * CalorSizeYZ;

static void InitGeant4()
{
  // --- Create materials.
  G4Material *world    = G4NistManager::Instance()->FindOrBuildMaterial(WorldMaterial);
  G4Material *gap      = G4NistManager::Instance()->FindOrBuildMaterial(GapMaterial);
  G4Material *absorber = G4NistManager::Instance()->FindOrBuildMaterial(AbsorberMaterial);
  //
  // --- Define a world.
  G4double worldDim          = 1 * m;
  G4Box *worldBox            = new G4Box("world", worldDim, worldDim, worldDim);
  G4LogicalVolume *worldLog  = new G4LogicalVolume(worldBox, world, "world");
  G4PVPlacement *worldPlaced = new G4PVPlacement(nullptr, {}, worldLog, "world", nullptr, false, 0);
  // --- Define two boxes to use the materials.
  G4double boxDim              = 0.25 * m;
  G4double gapBoxPos           = 0.5 * boxDim;
  G4double absorberBoxPos      = 1.5 * boxDim;
  G4Box *box                   = new G4Box("", boxDim, boxDim, boxDim);
  G4LogicalVolume *gapLog      = new G4LogicalVolume(box, gap, "gap");
  G4LogicalVolume *absorberLog = new G4LogicalVolume(box, absorber, "absorber");
  new G4PVPlacement(nullptr, {gapBoxPos, gapBoxPos, gapBoxPos}, gapLog, "gap", worldLog, false, 0);
  new G4PVPlacement(nullptr, {absorberBoxPos, absorberBoxPos, absorberBoxPos}, absorberLog, "absorber", worldLog, false,
                    0);
  //
  // --- Create particles that have secondary production threshold.
  G4Gamma::Gamma();
  G4Electron::Electron();
  G4Positron::Positron();
  G4Proton::Proton();
  G4ParticleTable *partTable = G4ParticleTable::GetParticleTable();
  partTable->SetReadiness();
  //
  // --- Create production - cuts object and set the secondary production threshold.
  G4ProductionCuts *productionCuts = new G4ProductionCuts();
  productionCuts->SetProductionCut(ProductionCut);
  //
  // --- Register a region for the world.
  G4Region *reg = new G4Region("default");
  reg->AddRootLogicalVolume(worldLog);
  reg->UsedInMassGeometry(true);
  reg->SetProductionCuts(productionCuts);
  //
  // --- Update the couple tables.
  G4ProductionCutsTable *theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  theCoupleTable->UpdateCoupleTable(worldPlaced);
  //
  // --- Set the world volume to fix initialization of G4SafetyHelper (used by G4UrbanMscModel)
  G4TransportationManager::GetTransportationManager()->SetWorldForTracking(worldPlaced);
  //
  // --- Set MSC range factor to match G4HepEm physics lists.
  G4EmParameters *param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetMscRangeFactor(0.04);
}

const void CreateVecGeomWorld()
{
  auto worldSolid = new vecgeom::UnplacedBox(0.5 * WorldSizeX, 0.5 * WorldSizeYZ, 0.5 * WorldSizeYZ);
  auto worldLogic = new vecgeom::LogicalVolume("World", worldSolid);
  vecgeom::VPlacedVolume *worldPlaced = worldLogic->Place();

  //
  // Calorimeter
  //
  auto calorSolid = new vecgeom::UnplacedBox(0.5 * CalorThickness, 0.5 * CalorSizeYZ, 0.5 * CalorSizeYZ);
  auto calorLogic = new vecgeom::LogicalVolume("Calorimeter", calorSolid);
  vecgeom::Transformation3D origin;
  worldLogic->PlaceDaughter(calorLogic, &origin);

  //
  // Layers
  //
  auto layerSolid = new vecgeom::UnplacedBox(0.5 * LayerThickness, 0.5 * CalorSizeYZ, 0.5 * CalorSizeYZ);

  //
  // Absorbers
  //
  auto gapSolid = new vecgeom::UnplacedBox(0.5 * GapThickness, 0.5 * CalorSizeYZ, 0.5 * CalorSizeYZ);
  auto gapLogic = new vecgeom::LogicalVolume("Gap", gapSolid);
  vecgeom::Transformation3D gapPlacement(-0.5 * LayerThickness + 0.5 * GapThickness, 0, 0);

  auto absorberSolid = new vecgeom::UnplacedBox(0.5 * AbsorberThickness, 0.5 * CalorSizeYZ, 0.5 * CalorSizeYZ);
  auto absorberLogic = new vecgeom::LogicalVolume("Absorber", absorberSolid);
  vecgeom::Transformation3D absorberPlacement(0.5 * LayerThickness - 0.5 * AbsorberThickness, 0, 0);

  // Create a new LogicalVolume per layer, we need unique IDs for scoring.
  double xCenter = -0.5 * CalorThickness + 0.5 * LayerThickness;
  for (int i = 0; i < NbOfLayers; i++) {
    auto layerLogic = new vecgeom::LogicalVolume("Layer", layerSolid);
    vecgeom::Transformation3D placement(xCenter, 0, 0);
    calorLogic->PlaceDaughter(layerLogic, &placement);

    layerLogic->PlaceDaughter(gapLogic, &gapPlacement);
    layerLogic->PlaceDaughter(absorberLogic, &absorberPlacement);

    xCenter += LayerThickness;
  }

  vecgeom::GeoManager::Instance().SetWorldAndClose(worldPlaced);
}

void PrintDaughters(const vecgeom::VPlacedVolume *placed, int level = 0)
{
  for (auto *daughter : placed->GetDaughters()) {
    std::cout << std::setw(level * 2) << "";
    auto *trans = daughter->GetTransformation();
    std::cout << " ID " << daughter->id() << " @ " << trans->Translation() << std::endl;
    PrintDaughters(daughter, level + 1);
  }
}

void PrintScoringPerVolume(const vecgeom::VPlacedVolume *placed, const ScoringPerVolume *scoring, int level = 0)
{
  for (auto *daughter : placed->GetDaughters()) {
    std::cout << std::setw(level * 2) << "";
    auto id = daughter->id();
    std::cout << " ID " << id << " Charged-TrakL " << scoring->chargedTrackLength[id] / copcore::units::mm
              << " mm; Energy-Dep " << scoring->energyDeposit[id] / copcore::units::MeV << " MeV" << std::endl;
    PrintScoringPerVolume(daughter, scoring, level + 1);
  }
}

void InitBVH()
{
  vecgeom::cxx::BVHManager::Init();
  vecgeom::cxx::BVHManager::DeviceInit();
}

int main(int argc, char *argv[])
{
#ifndef VECGEOM_USE_NAVINDEX
  std::cout << "### VecGeom must be compiled with USE_NAVINDEX support to run this.\n";
  return 1;
#endif

  OPTION_INT(cache_depth, 0); // 0 = full depth
  OPTION_INT(particles, 1000);
  OPTION_DOUBLE(energy, 10); // entered in GeV
  energy *= copcore::units::GeV;
  OPTION_INT(batch, -1);

  InitGeant4();
  CreateVecGeomWorld();

  const vecgeom::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
#ifdef VERBOSE
  std::cout << "World (ID " << world->id() << ")" << std::endl;
  PrintDaughters(world);
#endif

  // Map VecGeom volume IDs to Geant4 material-cuts couples.
  constexpr int NumVolumes = 1 + 1 + NbOfLayers * (1 + NbOfAbsorbers);
  int MCIndex[NumVolumes];
  // Fill world and calorimeter.
  MCIndex[0] = MCIndex[1] = WorldMC;
  for (int i = 2; i < NumVolumes; i += (1 + NbOfAbsorbers)) {
    MCIndex[i]     = WorldMC;
    MCIndex[i + 1] = GapMC;
    MCIndex[i + 2] = AbsorberMC;
  }
  // Place particles between the world boundary and the calorimeter.
  double startX = -0.25 * (WorldSizeX + CalorThickness);
  double chargedTrackLength[NumVolumes];
  double energyDeposit[NumVolumes];
  ScoringPerVolume scoringPerVolume;
  scoringPerVolume.chargedTrackLength = chargedTrackLength;
  scoringPerVolume.energyDeposit      = energyDeposit;
  GlobalScoring globalScoring;

  TestEm3(world, particles, energy, batch, startX, MCIndex, &scoringPerVolume, NumVolumes, &globalScoring);

  std::cout << std::endl;
  std::cout << std::endl;
  double meanEnergyDeposit = globalScoring.energyDeposit / particles;
  std::cout << "Mean energy deposit          " << (meanEnergyDeposit / copcore::units::GeV) << " GeV\n"
            << "Mean number of gamma         " << ((double)globalScoring.numGammas / particles) << "\n"
            << "Mean number of e-            " << ((double)globalScoring.numElectrons / particles) << "\n"
            << "Mean number of e+            " << ((double)globalScoring.numPositrons / particles) << "\n"
            << "Mean number of charged steps " << ((double)globalScoring.chargedSteps / particles) << "\n"
            << "Mean number of neutral steps " << ((double)globalScoring.neutralSteps / particles) << "\n"
            << "Mean number of hits          " << ((double)globalScoring.hits / particles) << "\n";
  if (globalScoring.numKilled > 0) {
    std::cout << "Total killed particles       " << globalScoring.numKilled << "\n";
  }
  std::cout << std::endl;

  // Average charged track length and energy deposit per particle.
  for (int i = 0; i < NumVolumes; i++) {
    chargedTrackLength[i] /= particles;
    energyDeposit[i] /= particles;
  }

  std::cout << std::scientific;
#ifdef VERBOSE
  std::cout << std::endl;
  const int id = world->id();
  std::cout << "World (ID " << world->id() << ") Charged-TrakL " << chargedTrackLength[id] / copcore::units::mm
            << " Energy-Dep " << energyDeposit[id] / copcore::units::MeV << " MeV" << std::endl;
  PrintScoringPerVolume(world, &scoringPerVolume);
#endif

  // Accumulate per material.
  double energyDepGap = 0, energyDepAbsorber = 0;
  for (int i = 2; i < NumVolumes; i += (1 + NbOfAbsorbers)) {
    energyDepGap += energyDeposit[i + 1];
    energyDepAbsorber += energyDeposit[i + 2];
  }

  std::cout << std::endl;
  std::cout << " " << GapMaterial << ": " << energyDepGap / copcore::units::GeV << " GeV" << std::endl;
  std::cout << " " << AbsorberMaterial << ": " << energyDepAbsorber / copcore::units::GeV << " GeV" << std::endl;
  // Accumulate per layer.
  std::cout << std::endl;
  std::cout << "  Layer   Charged-TrakL [mm]   Energy-Dep [MeV]" << std::endl;
  for (int i = 0; i < NbOfLayers; i++) {
    int layerVolume        = 2 + i * (1 + NbOfAbsorbers);
    double chargedTrackLen = chargedTrackLength[layerVolume + 1] + chargedTrackLength[layerVolume + 2];
    double energyDep       = energyDeposit[layerVolume + 1] + energyDeposit[layerVolume + 2];
    std::cout << std::setw(5) << i << std::setw(20) << chargedTrackLen / copcore::units::mm << std::setw(20)
              << energyDep / copcore::units::MeV << std::endl;
  }
}
