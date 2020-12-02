// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
/*
 * RaytraceBenchmark.cpp
 *
 *  Created on: Feb 13, 2020
 *      Author: andrei.gheata@cern.ch
 */
/// Adapted from VecGeom for AdePT by antonio.petre@spacescience.ro

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NavStatePath.h>
#include <VecGeom/base/Stopwatch.h>

#include "ArgParser.h"
#include "examples/Raytracer_Benchmark/Raytracer.h"
#include <CopCore/Global.h>
#include <AdePT/BlockData.h>
#include "kernels.h"
#include "examples/Raytracer_Benchmark/LoopNavigator.h"

#ifdef VECGEOM_GDML
#include <VecGeom/gdml/Frontend.h>
#endif

// forward declarations
int RaytraceBenchmarkCPU(cxx::RaytracerData_t &rtdata);

#ifdef VECGEOM_ENABLE_CUDA
namespace cuda {
struct RaytracerData_t;
} // namespace cuda

int RaytraceBenchmarkGPU(cuda::RaytracerData_t *, bool, int);
#endif

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

  // geometry file name and global transformation cache depth (affects size of navigation index table)
  OPTION_STRING(gdml_name, "trackML.gdml");
  OPTION_INT(cache_depth, 0); // 0 = full depth

  // image size in pixels
  OPTION_INT(px, 1840);
  OPTION_INT(py, 512);

  // RT model as in { kRTxray = 0, kRTspecular, kRTtransparent, kRTdiffuse };
  OPTION_INT(model, 2);

  // RT view as in { kRTVparallel = 0, kRTVperspective };
  OPTION_INT(view, 1);

  // zoom w.r.t to the default view mode
  OPTION_DOUBLE(zoom, 3.5);

  // Screen position in world coordinates
  OPTION_DOUBLE(screenx, -5000);
  OPTION_DOUBLE(screeny, 0);
  OPTION_DOUBLE(screenz, 0);

  // Up vector (no need to be normalized)
  OPTION_DOUBLE(upx, 0);
  OPTION_DOUBLE(upy, 1);
  OPTION_DOUBLE(upz, 0);
  vecgeom::Vector3D<double> up(upx, upy, upz);

  // Light color, object color (no color per volume yet) - in RGBA chars compressed into an unsigned integer
  OPTION_INT(bkgcol, 0xFF0000FF); // red
  OPTION_INT(objcol, 0x0000FFFF); // blue
  OPTION_INT(vdepth, 4);          // visible depth

  OPTION_INT(on_gpu, 1);     // run on GPU
  OPTION_INT(use_tiles, 0);  // run on GPU in tiled mode
  OPTION_INT(block_size, 8); // run on GPU in tiled mode

// Try to open the input file
#ifdef VECGEOM_GDML
  vecgeom::GeoManager::Instance().SetTransformationCacheDepth(cache_depth);
  bool load = vgdml::Frontend::Load(gdml_name.c_str(), false);
  if (!load) return 2;
#endif

  auto world = vecgeom::GeoManager::Instance().GetWorld();
  if (!world) return 3;

  RaytracerData_t rtdata;

  rtdata.fScreenPos.Set(screenx, screeny, screenz);
  rtdata.fUp.Set(upx, upy, upz);
  rtdata.fZoom     = zoom;
  rtdata.fModel    = (ERTmodel)model;
  rtdata.fView     = (ERTView)view;
  rtdata.fSize_px  = px;
  rtdata.fSize_py  = py;
  rtdata.fBkgColor = bkgcol;
  rtdata.fObjColor = objcol;
  rtdata.fVisDepth = vdepth;
  rtdata.fMaxDepth = vecgeom::GeoManager::Instance().getMaxDepth();

  Raytracer::InitializeModel(world, rtdata);
  rtdata.Print();

  auto ierr = 0;
  if (on_gpu) {
#ifdef VECGEOM_ENABLE_CUDA
    auto rtdata_cuda = reinterpret_cast<cuda::RaytracerData_t *>(&rtdata);
    ierr             = RaytraceBenchmarkGPU(rtdata_cuda, use_tiles, block_size);
#else
    std::cout << "=== Cannot run RaytracerBenchmark on GPU since VecGeom CUDA support not compiled.\n";
    return 1;
#endif
  } else {
    ierr = RaytraceBenchmarkCPU(rtdata);
  }
  if (ierr) std::cout << "TestNavIndex FAILED\n";

  return ierr;
}

int RaytraceBenchmarkCPU(cxx::RaytracerData_t &rtdata)
{
  using RayBlock     = adept::BlockData<Ray_t>;
  using RayAllocator = copcore::VariableSizeObjAllocator<RayBlock, copcore::BackendType::CPU>;
  using Launcher_t   = copcore::Launcher<copcore::BackendType::CPU>;
  using StreamStruct = copcore::StreamType<copcore::BackendType::CPU>;
  using Stream_t     = typename StreamStruct::value_type;

  // initialize BlockData of Ray_t structure
  int capacity = 1 << 20;
  RayAllocator hitAlloc(capacity);
  RayBlock *rays = hitAlloc.allocate(1);

  // Boilerplate to get the pointers to the device functions to be used
  COPCORE_CALLABLE_DECLARE(generateFunc, generateRays);

  // Create a stream to work with.
  Stream_t stream;
  StreamStruct::CreateStream(stream);

  // Allocate slots for the BlockData
  Launcher_t generate(stream);
  generate.Run(generateFunc, capacity, {0, 0}, rays);

  generate.WaitStream();

  // Allocate and initialize all rays on the host
  size_t raysize = Ray_t::SizeOfInstance();
  printf("=== Allocating %.3f MB of ray data on the host\n", (float)rtdata.fNrays * raysize / 1048576);
  unsigned char *input_buffer  = new unsigned char[rtdata.fNrays * raysize];
  unsigned char *output_buffer = new unsigned char[4 * rtdata.fNrays * sizeof(char)];

  // Initialize the navigation state for the view point
  vecgeom::NavStateIndex vpstate;
  LoopNavigator::LocatePointIn(rtdata.fWorld, rtdata.fStart, vpstate, true);

  rtdata.fVPstate = vpstate;

  // Construct rays in place
  for (int iray = 0; iray < rtdata.fNrays; ++iray)
    Ray_t::MakeInstanceAt(input_buffer + iray * raysize);

  // Run the CPU propagation kernel
  vecgeom::Stopwatch timer;
  timer.Start();
  Raytracer::PropagateRays(rays, rtdata, input_buffer, output_buffer);
  auto time_cpu = timer.Stop();
  std::cout << "Run time on CPU: " << time_cpu << "\n";

  // Write the output
  write_ppm("output.ppm", output_buffer, rtdata.fSize_px, rtdata.fSize_py);

  return 0;
}
