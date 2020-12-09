// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
/*
 * RaytraceBenchmark.cpp
 *
 *  Created on: Feb 13, 2020
 *      Author: andrei.gheata@cern.ch
 */
/// Adapted from VecGeom for AdePT by antonio.petre@spacescience.ro

#include "Raytracer.h"
#include "RaytraceBenchmark.hpp"

#include <CopCore/Global.h>
#include <AdePT/BlockData.h>
#include <AdePT/LoopNavigator.h>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NavStatePath.h>
#include <VecGeom/base/Stopwatch.h>

#ifdef VECGEOM_GDML
#include <VecGeom/gdml/Frontend.h>
#endif

int executePipelineGPU(const vecgeom::cxx::VPlacedVolume *world, int argc, char *argv[]);

int executePipelineCPU(const vecgeom::cxx::VPlacedVolume *world, int argc, char *argv[])
{
  int result = runSimulation<copcore::BackendType::CPU>(world, argc, argv);
  return result;
}

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

  OPTION_INT(on_gpu, 1); // run on GPU

#ifdef VECGEOM_GDML
  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(cache_depth);
  bool load = vgdml::Frontend::Load(gdml_name.c_str(), false);
  if (!load) return 2;
#endif

  const vecgeom::cxx::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (!world) return 3;

  auto ierr = 0;

  if (on_gpu) {
    ierr = executePipelineGPU(world, argc, argv);
  } else {
    ierr = executePipelineCPU(world, argc, argv);
  }
  if (ierr) std::cout << "TestNavIndex FAILED\n";

  return ierr;
}
