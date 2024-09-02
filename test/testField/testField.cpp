// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "testField.h"

#include <globals.hh>
#include <G4SystemOfUnits.hh>
#include <G4EmParameters.hh>
#include <G4GDMLParser.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4ProductionCutsTable.hh>
#include <G4RegionStore.hh>

#include <G4Gamma.hh>
#include <G4Electron.hh>
#include <G4Positron.hh>
#include <G4Proton.hh>
#include <G4ParticleTable.hh>

#include <G4HepEmData.hh>
#include <G4HepEmState.hh>
#include <G4HepEmStateInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmMatCutData.hh>

#include <VecGeom/base/Stopwatch.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/gdml/Frontend.h>
#include <VecGeom/management/CudaManager.h>
#ifdef ADEPT_USE_SURF
#include <VecGeom/surfaces/BrepHelper.h>
#endif

#include <AdePT/copcore/SystemOfUnits.h>
#include <AdePT/base/ArgParser.h>

static constexpr double DefaultCut = 0.7 * mm;

float BzFieldValue_host = 1.0 * copcore::units::tesla;
float *BzFieldValue_dev = nullptr;

const G4VPhysicalVolume *InitGeant4(const std::string &gdml_file)
{
  auto defaultRegion = new G4Region("DefaultRegionForTheWorld"); // deleted by store
  auto pcuts         = G4ProductionCutsTable::GetProductionCutsTable()->GetDefaultProductionCuts();
  pcuts->SetProductionCut(DefaultCut, "gamma");
  pcuts->SetProductionCut(DefaultCut, "e-");
  pcuts->SetProductionCut(DefaultCut, "e+");
  pcuts->SetProductionCut(DefaultCut, "proton");
  defaultRegion->SetProductionCuts(pcuts);

  // Read the geometry, regions and cuts from the GDML file
  std::cout << "reading " << gdml_file << " transiently on CPU for Geant4 ...\n";
  G4GDMLParser parser;
  parser.Read(gdml_file, false); // turn off schema checker
  G4VPhysicalVolume *world = parser.GetWorldVolume();

  if (world == nullptr) {
    std::cerr << "testField: World volume not set properly check your setup selection criteria or GDML input!\n";
    return world;
  }

  // - PHYSICS
  // --- Create particles that have secondary production threshold.
  G4Gamma::Gamma();
  G4Electron::Electron();
  G4Positron::Positron();
  G4Proton::Proton();
  G4ParticleTable *partTable = G4ParticleTable::GetParticleTable();
  partTable->SetReadiness();

  // - REGIONS
  if (world->GetLogicalVolume()->GetRegion() == nullptr) {
    // Add default region if none available
    defaultRegion->AddRootLogicalVolume(world->GetLogicalVolume());
  }
  for (auto region : *G4RegionStore::GetInstance()) {
    region->UsedInMassGeometry(true); // make sure all regions are marked as used
    region->UpdateMaterialList();
  }

  // - UPDATE COUPLES
  std::cout << "updating material-cut couples based on " << G4RegionStore::GetInstance()->size() << " regions ...\n";
  G4ProductionCutsTable *theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  theCoupleTable->UpdateCoupleTable(world);

  // --- Set the world volume to fix initialization of G4SafetyHelper (used by G4UrbanMscModel)
  G4TransportationManager::GetTransportationManager()->SetWorldForTracking(world);

  // --- Set MSC range factor to match G4HepEm physics lists.
  G4EmParameters *param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetMscRangeFactor(0.04);

  return world;
}

const vecgeom::cxx::VPlacedVolume *InitVecGeom(const std::string &gdml_file, int cache_depth)
{
  // Import the gdml file into VecGeom
  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(cache_depth);
  vgdml::Parser vgdmlParser;
  auto middleWare = vgdmlParser.Load(gdml_file.c_str(), false, copcore::units::mm);
  if (middleWare == nullptr) {
    std::cerr << "Failed to read geometry from GDML file '" << gdml_file << "'" << std::endl;
    return nullptr;
  }

  const vecgeom::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (world == nullptr) {
    std::cerr << "GeoManager world volume is nullptr" << std::endl;
    return nullptr;
  }
  return world;
}

int *CreateMCCindex(const G4VPhysicalVolume *g4world, const vecgeom::VPlacedVolume *world,
                    const G4HepEmState &hepEmState)
{
  const int *g4tohepmcindex = hepEmState.fData->fTheMatCutData->fG4MCIndexToHepEmMCIndex;

  // - FIND vecgeom::LogicalVolume corresponding to each and every G4LogicalVolume
  int nphysical = 0;

  int nvolumes   = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  int *mcc_index = new int[nvolumes];
  memset(mcc_index, 0, nvolumes * sizeof(int));

  // recursive geometry visitor lambda matching one by one Geant4 and VecGeom logical volumes
  // (we need to make sure we set the right MCC index to the right volume)
  typedef std::function<void(G4LogicalVolume const *, vecgeom::LogicalVolume const *)> func_t;
  func_t visitAndSetMCindex = [&](G4LogicalVolume const *g4vol, vecgeom::LogicalVolume const *vol) {
    int nd         = g4vol->GetNoDaughters();
    auto daughters = vol->GetDaughters();
    if (nd != daughters.size()) throw std::runtime_error("Mismatch in number of daughters");
    // Check the couples
    if (g4vol->GetMaterialCutsCouple() == nullptr)
      throw std::runtime_error("G4LogicalVolume " + std::string(g4vol->GetName()) +
                               std::string(" has no material-cuts couple"));
    int g4mcindex    = g4vol->GetMaterialCutsCouple()->GetIndex();
    int hepemmcindex = g4tohepmcindex[g4mcindex];
    // Check consistency with G4HepEm data
    if (hepEmState.fData->fTheMatCutData->fMatCutData[hepemmcindex].fG4MatCutIndex != g4mcindex)
      throw std::runtime_error("Mismatch between Geant4 mcindex and corresponding G4HepEm index");
    if (vol->id() >= nvolumes) throw std::runtime_error("Volume id larger than number of volumes");

    // All OK, now fill the index in the array
    mcc_index[vol->id()] = hepemmcindex;
    nphysical++;

    // Now do the daughters
    for (int id = 0; id < nd; ++id) {
      auto g4pvol = g4vol->GetDaughter(id);
      auto pvol   = daughters[id];
      // VecGeom does not strip pointers from logical volume names
      if (std::string(pvol->GetLogicalVolume()->GetName()).rfind(g4pvol->GetLogicalVolume()->GetName(), 0) != 0)
        throw std::runtime_error("Volume names " + std::string(pvol->GetLogicalVolume()->GetName()) + " and " +
                                 std::string(g4pvol->GetLogicalVolume()->GetName()) + " mismatch");
      visitAndSetMCindex(g4pvol->GetLogicalVolume(), pvol->GetLogicalVolume());
    }
  };

  visitAndSetMCindex(g4world->GetLogicalVolume(), world->GetLogicalVolume());
  std::cout << "Visited " << nphysical << " matching physical volumes\n";
  return mcc_index;
}

void FreeG4HepEm(G4HepEmState *state)
{
  FreeG4HepEmData(state->fData);
}

void InitBVH()
{
  vecgeom::cxx::BVHManager::Init();
#ifndef ADEPT_USE_SURF
  vecgeom::cxx::BVHManager::DeviceInit();
#endif
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

int main(int argc, char *argv[])
{
  // Only outputs are the data file(s)
  // Separate for now, but will want to unify
  OPTION_STRING(gdml_file, "cms2018.gdml"); //  "trackML.gdml"); //  "cms2018.gdml");
  OPTION_INT(cache_depth, 0);               // 0 = full depth
  OPTION_INT(particles, 100);
  OPTION_DOUBLE(energy, 10); // entered in GeV
  energy *= copcore::units::GeV;
  OPTION_INT(batch, -1);
  OPTION_BOOL(rotatingParticleGun, false);

  OPTION_DOUBLE(bzValue, -1.00); // entered in tesla
  BzFieldValue_host = bzValue * copcore::units::tesla;

  std::cout << "INFO: Particles to simulate = " << particles << " \n";
  std::cout << "INFO: Bz field value = " << bzValue << " T (tesla) \n";

  constexpr int stackLimit = 8 * 1024; // 8192
  printf("testField.cpp/main(): Setting Cuda Device Stack Limit to %6d \n", stackLimit);
  cudaError_t error = cudaDeviceSetLimit(cudaLimitStackSize, stackLimit);
  if (error != cudaSuccess) {
    printf("cudaDeviceSetLimit failed with %d, line(%d)\n", error, __LINE__);
    exit(EXIT_FAILURE);
  }

  vecgeom::Stopwatch timer;
  timer.Start();

  // Initialize Geant4
  auto g4world = InitGeant4(gdml_file);
  if (!g4world) return 3;

  // Initialize VecGeom
  std::cout << "reading " << gdml_file << " transiently on CPU for VecGeom ...\n";
  auto world = InitVecGeom(gdml_file, cache_depth);
  if (!world) return 3;

  // Construct and initialize the G4HepEmState data/tables
  std::cout << "initializing G4HepEm state ...\n";
  G4HepEmState hepEmState;
  InitG4HepEmState(&hepEmState);

  // Initialize G4HepEm material-cut couple array indexed by VecGeom volume id.
  // (In future we should connect the index directly to the VecGeom logical volume)
  std::cout << "initializing material-cut couple indices ...\n";
  int *MCCindex = nullptr;

  try {
    MCCindex = CreateMCCindex(g4world, world, hepEmState);
  } catch (const std::runtime_error &ex) {
    std::cerr << "*** CreateMCCindex: " << ex.what() << "\n";
    return 1;
  }

  // Load and synchronize the geometry on the GPU
  std::cout << "synchronizing VecGeom geometry to GPU ...\n";
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
#ifdef ADEPT_USE_SURF
#ifdef ADEPT_USE_SURF_SINGLE
  using BrepHelper = vgbrep::BrepHelper<float>;
#else
  using BrepHelper = vgbrep::BrepHelper<double>;
#endif
#endif

#ifndef ADEPT_USE_SURF
  cudaManager.LoadGeometry(world);
  cudaManager.Synchronize();
#else
  if (!BrepHelper::Instance().Convert()) return 1;
  BrepHelper::Instance().PrintSurfData();
  cudaManager.SynchronizeNavigationTable();
#endif
  InitBVH();

  auto time_cpu = timer.Stop();
  std::cout << "Initialization took: " << time_cpu << " sec\n";

  int NumVolumes = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  int NumPlaced  = vecgeom::GeoManager::Instance().GetPlacedVolumesCount();

  // Scoring is done per placed volume (for now...)
  double *chargedTrackLength = new double[NumPlaced];
  double *energyDeposit      = new double[NumPlaced];
  ScoringPerVolume scoringPerVolume;
  scoringPerVolume.chargedTrackLength = chargedTrackLength;
  scoringPerVolume.energyDeposit      = energyDeposit;
  GlobalScoring globalScoring;

  testField(particles, energy, batch, MCCindex, &scoringPerVolume, &globalScoring, NumVolumes, NumPlaced, &hepEmState,
            rotatingParticleGun);

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
  for (int i = 0; i < NumPlaced; i++) {
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
  FreeG4HepEm(&hepEmState);
  delete[] MCCindex;
  delete[] chargedTrackLength;
  delete[] energyDeposit;
  return 0;
}
