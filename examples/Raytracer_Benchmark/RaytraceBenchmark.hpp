// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NavStatePath.h>
#include <VecGeom/base/Stopwatch.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/base/Global.h>

#include "ArgParser.h"
#include "examples/Raytracer_Benchmark/Raytracer.h"
#include <CopCore/Global.h>
#include <AdePT/BlockData.h>
#include "kernels.h"
#include "examples/Raytracer_Benchmark/LoopNavigator.h"

#ifdef VECGEOM_GDML
#include <VecGeom/gdml/Frontend.h>
#endif

void initiliazeCudaWorld(RaytracerData_t *rtdata);

void RenderTiledImage(adept::BlockData<Ray_t> *rays, RaytracerData_t *rtdata, NavIndex_t *output_buffer,
                      int block_size);

template <copcore::BackendType backend>
void InitRTdata(RaytracerData_t *rtdata)
{

  if (backend == copcore::BackendType::CUDA) {
    initiliazeCudaWorld((RaytracerData_t *)rtdata);
  } else {
    vecgeom::NavStateIndex vpstate;
    LoopNavigator::LocatePointIn(rtdata->fWorld, rtdata->fStart, vpstate, true);
    rtdata->fVPstate = vpstate;
  }
}

template <copcore::BackendType backend>
int runSimulation(const vecgeom::cxx::VPlacedVolume *world, int argc, char *argv[])
{

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

  OPTION_INT(use_tiles, 0);  // run on GPU in tiled mode
  OPTION_INT(block_size, 8); // run on GPU in tiled mode

  copcore::Allocator<RaytracerData_t, backend> rayAlloc;
  RaytracerData_t *rtdata = rayAlloc.allocate(1);

  rtdata->fScreenPos.Set(screenx, screeny, screenz);
  rtdata->fUp.Set(upx, upy, upz);
  rtdata->fZoom     = zoom;
  rtdata->fModel    = (ERTmodel)model;
  rtdata->fView     = (ERTView)view;
  rtdata->fSize_px  = px;
  rtdata->fSize_py  = py;
  rtdata->fBkgColor = bkgcol;
  rtdata->fObjColor = objcol;
  rtdata->fVisDepth = vdepth;
  rtdata->fMaxDepth = vecgeom::GeoManager::Instance().getMaxDepth();

  Raytracer::InitializeModel((Raytracer::VPlacedVolumePtr_t)world, *rtdata);

  InitRTdata<backend>(rtdata);

  rtdata->Print();

  using RayBlock     = adept::BlockData<Ray_t>;
  using RayAllocator = copcore::VariableSizeObjAllocator<RayBlock, backend>;
  using Launcher_t   = copcore::Launcher<backend>;
  using StreamStruct = copcore::StreamType<backend>;
  using Stream_t     = typename StreamStruct::value_type;

  // initialize BlockData of Ray_t structure
  int capacity = 1 << 20;
  RayAllocator hitAlloc(capacity);
  RayBlock *rays = hitAlloc.allocate(1);

  // Boilerplate to get the pointers to the device functions to be used
  COPCORE_CALLABLE_DECLARE(generateFunc, generateRays);
  COPCORE_CALLABLE_DECLARE(renderkernelFunc, renderKernels);

  // Create a stream to work with.
  Stream_t stream;
  StreamStruct::CreateStream(stream);

  // Allocate slots for the BlockData
  Launcher_t generate(stream);
  generate.Run(generateFunc, capacity, {0, 0}, rays);

  generate.WaitStream();

  // Allocate and initialize all rays on the host
  size_t raysize = Ray_t::SizeOfInstance();
  printf("=== Allocating %.3f MB of ray data on the %s\n", (float)rtdata->fNrays * raysize / 1048576,
         copcore::BackendName(backend));

  copcore::Allocator<NavIndex_t, backend> charAlloc;
  NavIndex_t *input_buffer = charAlloc.allocate(rtdata->fNrays * raysize * sizeof(NavIndex_t));

  copcore::Allocator<NavIndex_t, backend> ucharAlloc;
  NavIndex_t *output_buffer = ucharAlloc.allocate(4 * rtdata->fNrays * sizeof(NavIndex_t));

  // Construct rays in place
  for (int iray = 0; iray < rtdata->fNrays; ++iray)
    Ray_t::MakeInstanceAt(input_buffer + iray * raysize);

  vecgeom::Stopwatch timer;
  timer.Start();

  if (backend == copcore::BackendType::CUDA && use_tiles) {
    RenderTiledImage(rays, (RaytracerData_t *)rtdata, output_buffer, block_size);
  } else {
    Launcher_t renderKernel(stream);
    renderKernel.Run(renderkernelFunc, rays->GetNused(), {0, 0}, rays, *rtdata, input_buffer, output_buffer);
    renderKernel.WaitStream();
  }

  auto time_cpu = timer.Stop();
  std::cout << "Run time: " << time_cpu << "\n";

  // Write the output
  write_ppm("output.ppm", output_buffer, rtdata->fSize_px, rtdata->fSize_py);

  return 0;
}
