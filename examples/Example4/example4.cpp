// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example4.h"

#include <AdePT/ArgParser.h>

#include <VecGeom/base/Config.h>
#include <VecGeom/management/GeoManager.h>
#ifdef VECGEOM_GDML
#include <VecGeom/gdml/Frontend.h>
#endif

void example4();

int main(int argc, char *argv[])
{
#ifndef VECGEOM_GDML
  std::cout << "### VecGeom must be compiled with GDML support to run this.\n";
  return 1;
#endif
#ifndef VECGEOM_USE_NAVINDEX
  std::cout << "### VecGeom must be compiled with USE_NAVINDEX support to run this.\n";
  return 2;
#endif

  OPTION_STRING(gdml_name, "trackML.gdml");
  OPTION_INT(cache_depth, 0); // 0 = full depth

#ifdef VECGEOM_GDML
  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(cache_depth);
  // The vecgeom millimeter unit is the last parameter of vgdml::Frontend::Load
  bool load = vgdml::Frontend::Load(gdml_name.c_str(), false, 1);
  if (!load) return 3;
#endif

  const vecgeom::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (!world) return 4;

  example4(world);
}
