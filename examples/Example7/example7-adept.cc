// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include <AdePT/ArgParser.h>

#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Frontend.h>

int main(int argc, char* argv[])
{
  // Only inputs are the data file(s)?
  // Separate for now, but will want to unify
  OPTION_STRING(gdml_file, "");
  OPTION_STRING(g4hepem_file, "");
  // Keep cache_depth from example6
  OPTION_INT(cache_depth, 0); // 0 = full depth

  // Args must not be empty
  if(gdml_file.empty()) {
    std::cerr << "required gdml_file argument is empty\n";
    return 2;
  }

  if(g4hepem_file.empty()) {
    std::cerr << "required hepem_file argument is empty\n";
    return 2;
  }

  // Main task is to reconstitute GDML and G4HepEMData
  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(cache_depth);
  bool load = vgdml::Frontend::Load(gdml_file.c_str(), false);
  if (!load) return 3;

  const vecgeom::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (!world) return 4;

  return 0;
}