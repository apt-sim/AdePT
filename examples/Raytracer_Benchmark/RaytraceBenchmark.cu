// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/// \file Raytracer.cu
/// \author Guilherme Amadio. Rewritten to use navigation from common code by Andrei Gheata.
/// Adapted from VecGeom for AdePT by antonio.petre@spacescience.ro

#include "Raytracer.h"
#include <CopCore/Global.h>
#include <AdePT/BlockData.h>
#include "examples/Raytracer_Benchmark/LoopNavigator.h"
#include "RaytraceBenchmark.hpp"

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/navigation/NavStateIndex.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/base/Stopwatch.h>
#include <VecGeom/base/Global.h>

#include <cuda_profiler_api.h>

#include <cassert>
#include <cstdio>

__global__ void RenderTile(adept::BlockData<Ray_t> *rays, RaytracerData_t rtdata, int offset_x, int offset_y,
                           int tile_size_x, int tile_size_y, unsigned char *tile_in, unsigned char *tile_out)
{
  int local_px = threadIdx.x + blockIdx.x * blockDim.x;
  int local_py = threadIdx.y + blockIdx.y * blockDim.y;

  if (local_px >= tile_size_x || local_py >= tile_size_y) return;

  int ray_index   = local_py * tile_size_x + local_px;
  int pixel_index = 4 * (local_py * tile_size_x + local_px);

  int global_px = offset_x + local_px;
  int global_py = offset_y + local_py;

  Ray_t *ray = (Ray_t *)(tile_in + ray_index * sizeof(Ray_t));
  ray->index = ray_index;

  (*rays)[ray_index]         = *ray;
  adept::Color_t pixel_color = Raytracer::RaytraceOne(rtdata, rays, global_px, global_py, ray->index);

  tile_out[pixel_index + 0] = pixel_color.fComp.red;
  tile_out[pixel_index + 1] = pixel_color.fComp.green;
  tile_out[pixel_index + 2] = pixel_color.fComp.blue;
  tile_out[pixel_index + 3] = 255;
}

// subdivide image in 16 tiles and launch each tile on a separate CUDA stream
void RenderTiledImage(adept::BlockData<Ray_t> *rays, cuda::RaytracerData_t *rtdata, NavIndex_t *output_buffer,
                      int block_size)
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

      RenderTile<<<blocks, threads, 0, streams[iy]>>>(rays, *rtdata, offset_x, offset_y, tile_size_x, tile_size_y,
                                                      tile_device_in[idx], tile_device_out[idx]);
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

void initiliazeCudaWorld(cuda::RaytracerData_t *rtdata) {
  
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

}

int executePipelineGPU(const vecgeom::cxx::VPlacedVolume *world, int argc, char *argv[])
{
  int result;
  result = runSimulation<copcore::BackendType::CUDA>(world, argc, argv);
  return result;
}