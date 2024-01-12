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
#include "Material.h"
#include <vector>

#include <CopCore/Global.h>
#include <AdePT/base/BlockData.h>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NavStatePath.h>
#include <VecGeom/base/Stopwatch.h>

#ifdef VECGEOM_GDML
#include <VecGeom/gdml/Frontend.h>
#endif

// Note: The function below needs to be in a .cpp file,
// otherwise the vecgeom::cxx namespace is not accessible.

void InitBVH(bool on_gpu)
{
  vecgeom::cxx::BVHManager::Init();

  if (on_gpu) vecgeom::cxx::BVHManager::DeviceInit();
}

namespace cuda {
struct MyMediumProp;
} // namespace cuda

int executePipelineGPU(const cuda::MyMediumProp *volume_container, const vecgeom::cxx::VPlacedVolume *world, int argc,
                       char *argv[]);

int executePipelineCPU(const MyMediumProp *volume_container, const vecgeom::cxx::VPlacedVolume *world, int argc,
                       char *argv[])
{

  int result = runSimulation<copcore::BackendType::CPU>(volume_container, world, argc, argv);
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

  OPTION_STRING(gdml_file, "trackML.gdml");
  OPTION_INT(cache_depth, 0); // 0 = full depth

  OPTION_INT(on_gpu, 1); // run on GPU

#ifdef VECGEOM_GDML
  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(cache_depth);

  // The vecgeom millimeter unit is the last parameter of vgdml::Frontend::Load
  bool load = vgdml::Frontend::Load(gdml_file.c_str(), false, 1);
  if (!load) return 2;
#endif

  const vecgeom::cxx::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
  if (!world) return 3;

  auto ierr               = 0;
  const int maxno_volumes = 1000;

  // Allocate material structure
  static MyMediumProp *volume_container;
  cudaMallocManaged(&volume_container, maxno_volumes * sizeof(MyMediumProp));

  std::vector<vecgeom::LogicalVolume *> logicalvolumes;
  vecgeom::GeoManager::Instance().GetAllLogicalVolumes(logicalvolumes);

  SetMaterialStruct(volume_container, logicalvolumes, on_gpu);

  if (on_gpu) {
    auto volume_container_cuda = reinterpret_cast<cuda::MyMediumProp *>(volume_container);
    ierr                       = executePipelineGPU(volume_container_cuda, world, argc, argv);
  } else {
    ierr = executePipelineCPU(volume_container, world, argc, argv);
  }
  if (ierr) std::cout << "TestNavIndex FAILED\n";

  return ierr;
}
