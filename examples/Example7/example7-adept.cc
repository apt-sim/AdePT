// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <iomanip>
#include <unordered_set>

#include <AdePT/ArgParser.h>
#include <CopCore/SystemOfUnits.h>

#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Frontend.h>

#include <G4HepEmState.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmData.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmDataJsonIO.hh>

// Just to get sizes so we know where to start particles
//#include "example7-geometry.h"

#include "TestEm3.h"

/// Find and return Auxiliary with given "type" attribute in passed collection
/// Return default (null) instance in case Auxiliary is not found.
template <typename Collection>
vgdml::Auxiliary find_by_type(const Collection& c, const std::string& type_value) {
  auto iter = std::find_if(c.begin(), c.end(), [&](const vgdml::Auxiliary& a) {
    return a.GetType() == type_value;
  });

  return iter == c.end() ? vgdml::Auxiliary{} : *iter;
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

// We shoot particles along {1,0,0} starting from {-x, 0, 0}
// Calculate -x as distance to world boundary plus a small tolerance
double GetStartX(const vecgeom::VPlacedVolume *volume) {
  auto dist = volume->DistanceToOut({0,0,0}, {-1,0,0});
  return -1.0*dist + 1.0*copcore::units::mm;
}

int main(int argc, char* argv[])
{
#ifndef VECGEOM_USE_NAVINDEX
  std::cout << "### VecGeom must be compiled with USE_NAVINDEX support to run this.\n";
  return 1;
#endif

  // Command line options
  // G4HepEMData file path is stored as an auxiliary
  OPTION_STRING(gdml_file, "");

  // Remainder of options follow TestEm3 example
  OPTION_INT(cache_depth, 0); // 0 = full depth
  OPTION_INT(particles, 1000);
  OPTION_DOUBLE(energy, 10); // entered in GeV
  energy *= copcore::units::GeV;
  OPTION_INT(batch, -1);

  // File arg must not be empty
  if(gdml_file.empty()) {
    std::cerr << "required gdml_file argument is empty\n";
    return 2;
  }
  // Main task in this example is to reconstitute GDML and G4HepEMData
  // - GDML
  // Inputs: File path, Cache depth
  // Output: vgdml::Middleware, vecgeom::VPlacedVolume (world volume)
  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(cache_depth);
  vgdml::Parser vgdmlParser;
  auto middleWare = vgdmlParser.Load(gdml_file.c_str(), false, copcore::units::mm);
  if (middleWare == nullptr)
  {
    std::cerr << "Failed to read geometry from GDML file '" << gdml_file << "'" << std::endl;
    return 3;
  }

  const vecgeom::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (world == nullptr) {
    std::cerr << "GeoManager world volume is nullptr" << std::endl;
    return 3;
  }
#ifdef VERBOSE
  std::cout << "World (ID " << world->id() << ")" << std::endl;
  PrintDaughters(world);
#endif

  // G4HepEmData
  // Input: file path (here from vgdml::Middleware)
  // Output: G4HepEmState pointer (owned)
  const auto& userInfo = middleWare->GetUserInfo();
  vgdml::Auxiliary g4hepemFileAux = find_by_type(userInfo, "g4hepemfile");

  if (g4hepemFileAux == vgdml::Auxiliary{}) {
    std::cerr << "GDML file '" << gdml_file << "' has no 'g4hepemfile' user info auxiliary" << std::endl;
    return 4;
  }

  std::string g4hepem_file = g4hepemFileAux.GetValue();
  std::ifstream jsonIS{ g4hepem_file.c_str() };

  G4HepEmState* inState = G4HepEmStateFromJson(jsonIS);
  if(inState == nullptr)
  {
    std::cerr << "Failed to read G4HepEmState from " << g4hepem_file << std::endl;
    return 4;
  }

  // Connect up VecGeom _physical_ volume ids to G4HepEMData material cut couple indexes
  // via Geant4 material cut couple index supplied as a volume auxiliary in the GDML
  // Inputs: vecgeom::GeoManager, vgdml::Middleware, G4HepEmData
  // Output: Array of ints. Array indices are VecGeom PV id numbers, value at that index is G4HepEmData MCC index
  // Visting all PVs walks the tree, so we get multiple copies in the vector. Thus reduced to a
  // set of pointers afterwards to uniq the list.
  std::vector<vecgeom::VPlacedVolume*> tmpPlacedVolumes;
  vecgeom::GeoManager::Instance().getAllPlacedVolumes(tmpPlacedVolumes);
  std::unordered_set<vecgeom::VPlacedVolume*> allPlacedVolumes{tmpPlacedVolumes.begin(), tmpPlacedVolumes.end()};

  const auto NumVolumes = allPlacedVolumes.size();
  int* MCCIndex = new int[NumVolumes];

  const auto& lvAuxInfo = middleWare->GetVolumeAuxiliaryInfo();

  G4HepEmData* inData = inState->fData;

  for (const vecgeom::VPlacedVolume* pv : allPlacedVolumes) {
    const auto lvId = pv->GetLogicalVolume()->id();

    auto lvAuxForThisLV = lvAuxInfo.find(lvId);
    if(lvAuxForThisLV == lvAuxInfo.end()) {
      std::cerr << "Logical volume " << lvId << " has no auxiliary data" << std::endl;
      return 5;
    }

    // Auxiliary to find has type "g4matcutcouple"
    vgdml::Auxiliary mccAux = find_by_type(lvAuxForThisLV->second, "g4matcutcouple");
    if (mccAux == vgdml::Auxiliary{}) {
      std::cerr << "Logical volume " << lvId << " has no auxiliary of type 'g4matcutcouple'" << std::endl;
      return 5;
    }

    // Value is of auxiliary is Geant4 mcc index, so use G4HepEm to get index conversion
    int geant4MCCIdx = std::stoi(mccAux.GetValue());
    MCCIndex[pv->id()] = inData->fTheMatCutData->fG4MCIndexToHepEmMCIndex[geant4MCCIdx];
  }

  // Place particles between the world boundary and the calorimeter.
  double startX = GetStartX(world);
  GlobalScoring globalScoring;

  // NB: Scoring per Volume retained for interface compatibility with TestEm3 but it doesn't work here yet
  // - Geometry here is nested, so we have
  //   + World(1x)
  //     + Calorimeter(1x)
  //       + Layer (50x)
  //         + Pb (1x)
  //         + Ar (1x)
  // - Scoring per volume therefore needs a copy number of the Pb/Ar volumes from the layer mother (TODO)
  // - Implementation safe to retain as NumVolumes is correct size, but intepretation as scoring per
  //   individual Pb/Lar layer no longer correct.
  double chargedTrackLength[NumVolumes];
  double energyDeposit[NumVolumes];
  ScoringPerVolume scoringPerVolume;
  scoringPerVolume.chargedTrackLength = chargedTrackLength;
  scoringPerVolume.energyDeposit      = energyDeposit;

  TestEm3(world, particles, energy, batch, startX, MCCIndex, &scoringPerVolume, NumVolumes, &globalScoring, inState);

  // -- Report global scoring
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
  std::cout << std::endl;

  // TODO: reimplement scoring-per-volume
  // --

  // Free directly allocated data that we own
  delete [] MCCIndex;
  delete inState->fParameters;
  FreeG4HepEmData(inState->fData); // NB: this also frees any device memory!

  return 0;
}