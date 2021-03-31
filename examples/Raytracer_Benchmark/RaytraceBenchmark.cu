// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/// \file Raytracer.cu
/// \author Guilherme Amadio. Rewritten to use navigation from common code by Andrei Gheata.
/// Adapted from VecGeom for AdePT by antonio.petre@spacescience.ro

#include "Raytracer.h"
#include "RaytraceBenchmark.hpp"

#include <CopCore/Global.h>
#include <AdePT/BlockData.h>
#include <AdePT/LoopNavigator.h>

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/navigation/NavStateIndex.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/base/Stopwatch.h>
#include <VecGeom/base/Global.h>

#include <cassert>
#include <cstdio>
#include <vector>

__global__ void RenderTile(RaytracerData_t rtdata, int offset_x, int offset_y,
                           int tile_size_x, int tile_size_y, unsigned char *tile_in, unsigned char *tile_out, int generation)
{
  int local_px = threadIdx.x + blockIdx.x * blockDim.x;
  int local_py = threadIdx.y + blockIdx.y * blockDim.y;

  if (local_px >= tile_size_x || local_py >= tile_size_y) return;

  int pixel_index = 4 * (local_py * tile_size_x + local_px);

  int global_px = offset_x + local_px;
  int global_py = offset_y + local_py;

  int ray_index = global_py*tile_size_x + global_px;

  if (!(rtdata.sparse_rays)[generation]->is_used(ray_index)) return;
  Ray_t *ray = &(*rtdata.sparse_rays[generation])[ray_index];

  auto pixel_color = Raytracer::RaytraceOne(rtdata, *ray, generation);

  tile_out[pixel_index + 0] = pixel_color.fComp.red;
  tile_out[pixel_index + 1] = pixel_color.fComp.green;
  tile_out[pixel_index + 2] = pixel_color.fComp.blue;
  tile_out[pixel_index + 3] = 255;
}

// subdivide image in 16 tiles and launch each tile on a separate CUDA stream
void RenderTiledImage(cuda::RaytracerData_t *rtdata, NavIndex_t *output_buffer, int generation, int block_size)
{
  cudaStream_t streams[4];

  unsigned char *tile_host[16];
  unsigned char *tile_device_in[16];
  unsigned char *tile_device_out[16];

  int tile_size_x = rtdata->fSize_px / 4 + 1;
  int tile_size_y = rtdata->fSize_py / 4 + 1;

  // subdivide each tile in block_size x block_size thread blocks
  dim3 threads(block_size, block_size);
  dim3 blocks(tile_size_x / block_size + 1, tile_size_y / block_size + 1);

  for (int i = 0; i < 4; ++i)
    COPCORE_CUDA_CHECK(cudaStreamCreate(&streams[i]));

  for (int i = 0; i < 16; ++i) {
    // allocate tile on host and device
    COPCORE_CUDA_CHECK(cudaMalloc((void **)&tile_device_in[i], tile_size_x * tile_size_y * sizeof(cuda::Ray_t)));
    COPCORE_CUDA_CHECK(cudaMalloc((void **)&tile_device_out[i], 4 * tile_size_x * tile_size_y));
    // CUDA streams require host memory to be pinned
    COPCORE_CUDA_CHECK(cudaMallocHost(&tile_host[i], 4 * tile_size_x * tile_size_y));
  }

  // wait for memory to reach GPU before launching kernels
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  // call kernels to render each tile
  for (int ix = 0; ix < 4; ++ix) {
    for (int iy = 0; iy < 4; ++iy) {
      int idx      = 4 * ix + iy;
      int offset_x = ix * tile_size_x;
      int offset_y = iy * tile_size_y;

      RenderTile<<<blocks, threads, 0, streams[iy]>>>(*rtdata, offset_x, offset_y, tile_size_x, tile_size_y,
                                                      tile_device_in[idx], tile_device_out[idx], generation);
    }
  }

  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  // copy back rendered tile to system memory
  for (int ix = 0; ix < 4; ++ix) {
    for (int iy = 0; iy < 4; ++iy) {
      int idx = 4 * ix + iy;
      COPCORE_CUDA_CHECK(cudaMemcpyAsync(tile_host[idx], tile_device_out[idx], (size_t)4 * tile_size_x * tile_size_y,
                                         cudaMemcpyDeviceToHost, streams[iy]));
      COPCORE_CUDA_CHECK(cudaFree(tile_device_in[idx]));
      COPCORE_CUDA_CHECK(cudaFree(tile_device_out[idx]));
    }
  }

  // ensure all tiles have been copied back
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  COPCORE_CUDA_CHECK(cudaGetLastError());

  for (int ix = 0; ix < 4; ++ix) {
    for (int iy = 0; iy < 4; ++iy) {
      int idx      = 4 * ix + iy;
      int offset_x = ix * tile_size_x;
      int offset_y = iy * tile_size_y;

      // copy each tile into the final destination
      for (int i = 0; i < tile_size_x; ++i) {
        for (int j = 0; j < tile_size_y; ++j) {
          int px          = offset_x + i;
          int py          = offset_y + j;
          int tile_index  = 4 * (j * tile_size_x + i);
          int pixel_index = 4 * (py * rtdata->fSize_px + px);

          if ((px >= rtdata->fSize_px) || (py >= rtdata->fSize_py)) continue;

          output_buffer[pixel_index + 0] = tile_host[idx][tile_index + 0];
          output_buffer[pixel_index + 1] = tile_host[idx][tile_index + 1];
          output_buffer[pixel_index + 2] = tile_host[idx][tile_index + 2];
          output_buffer[pixel_index + 3] = tile_host[idx][tile_index + 3];
        }
      }
      COPCORE_CUDA_CHECK(cudaFreeHost(tile_host[idx]));
    }
  }
  COPCORE_CUDA_CHECK(cudaGetLastError());
}

__global__ void getAllLogicalVolumes(vecgeom::VPlacedVolume *currentVolume, vecgeom::Vector<vecgeom::LogicalVolume *> *container)
{
  
  auto lvol = (vecgeom::LogicalVolume *)currentVolume->GetLogicalVolume();
  bool cond = true;

  for (auto it = container->begin(); it != container->end(); it++) {
    if (lvol == *it)
      cond = false;
  }

  if (cond) {
    container->reserve(1*sizeof(vecgeom::LogicalVolume *));
    container->push_back(lvol);
  }

  const vecgeom::Vector<vecgeom::VPlacedVolume const *> placedvolumes = lvol->GetDaughters();

  for (auto crt: placedvolumes) {
    getAllLogicalVolumes<<<1,1>>>((vecgeom::VPlacedVolume *)  crt, container);
  }
}


// Attach material structure to all logical volumes
__global__ void AttachRegions(const MyMediumProp *volume_container, vecgeom::Vector<vecgeom::LogicalVolume *> *container)
{
  int i = 0;
  for (auto x: *container) {
    // x->Print();
    x->SetBasketManagerPtr((void *)&volume_container[i]);
    i++;
  }
}


void initiliazeCudaWorld(cuda::RaytracerData_t *rtdata, const MyMediumProp *volume_container) {
  
  // Load and synchronize the geometry on the GPU
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  cudaManager.LoadGeometry((vecgeom::cxx::VPlacedVolume *)rtdata->fWorld);
  cudaManager.Synchronize();

  auto gpu_world = cudaManager.world_gpu();
  assert(gpu_world && "GPU world volume is a null pointer");

  // Initialize the navigation state for the view point
  vecgeom::NavStateIndex vpstate;
  LoopNavigator::LocatePointIn(rtdata->fWorld, rtdata->fStart, vpstate, true);
  rtdata->fVPstate = vpstate;
  rtdata->fWorld   = gpu_world;

  vecgeom::Vector<vecgeom::LogicalVolume *> *container;
  cudaMallocManaged(&container, sizeof(vecgeom::Vector<vecgeom::LogicalVolume *>));

  getAllLogicalVolumes<<<1,1>>>((vecgeom::VPlacedVolume *)gpu_world, container);

  cudaDeviceSynchronize();

  AttachRegions<<<1,1>>>(volume_container, container);
}

int executePipelineGPU(const MyMediumProp *volume_container, const vecgeom::cxx::VPlacedVolume *world, int argc, char *argv[])
{
  int result;
  result = runSimulation<copcore::BackendType::CUDA>(volume_container, world, argc, argv);
  return result;
}