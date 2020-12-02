// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/// \file Raytracer.cu
/// \author Guilherme Amadio. Rewritten to use navigation from common code by Andrei Gheata.
/// Adapted from VecGeom for AdePT by antonio.petre@spacescience.ro

#include "Raytracer.h"
#include <CopCore/Global.h>
#include <AdePT/BlockData.h>
#include "kernels.h"
#include "examples/Raytracer_Benchmark/LoopNavigator.h"

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/navigation/NavStateIndex.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/base/Stopwatch.h>


#include <cuda_profiler_api.h>

#include <cassert>
#include <cstdio>


void check_cuda_err(cudaError_t result, char const *const func, const char *const file, int const line)
{
  if (result) {
    fprintf(stderr, "CUDA error = %s at %s:%d\n", cudaGetErrorString(result), file, line);
    cudaDeviceReset();
    exit(1);
  }
}

#define checkCudaErrors(val) check_cuda_err((val), #val, __FILE__, __LINE__)

__global__ void RenderKernel(adept::BlockData<Ray_t> *rays, RaytracerData_t rtdata, char *input_buffer,
                             unsigned char *output_buffer)
{
  int px = threadIdx.x + blockIdx.x * blockDim.x;
  int py = threadIdx.y + blockIdx.y * blockDim.y;

  if ((px >= rtdata.fSize_px) || (py >= rtdata.fSize_py)) return;

  int ray_index = py * rtdata.fSize_px + px;

  Ray_t *ray = (Ray_t *)(input_buffer + ray_index * sizeof(Ray_t));
  ray->index = ray_index;

  (*rays)[ray_index] = *ray;
  
  adept::Color_t pixel_color = Raytracer::RaytraceOne(rtdata, rays, px, py, ray->index);

  int pixel_index = 4 * ray_index;

  output_buffer[pixel_index + 0] = pixel_color.fComp.red;
  output_buffer[pixel_index + 1] = pixel_color.fComp.green;
  output_buffer[pixel_index + 2] = pixel_color.fComp.blue;
  output_buffer[pixel_index + 3] = 255;
}

__global__ void RenderLine(RaytracerData_t rtdata, int py, unsigned char *line)
{
  int px = threadIdx.x + blockIdx.x * blockDim.x;

  if (px >= rtdata.fSize_px) return;

  Ray_t ray;
  adept::BlockData<Ray_t> *rays;
  adept::Color_t pixel_color = Raytracer::RaytraceOne(rtdata, rays, px, py, 0);

  line[4 * px + 0] = pixel_color.fComp.red;
  line[4 * px + 1] = pixel_color.fComp.green;
  line[4 * px + 2] = pixel_color.fComp.blue;
  line[4 * px + 3] = 255;
}

__global__ void RenderInterlaced(RaytracerData_t rtdata, int offset, int width, unsigned char *output)
{
  Ray_t ray;
  adept::BlockData<Ray_t> *rays;

  int px = threadIdx.x + blockIdx.x * blockDim.x;

  if (px >= rtdata.fSize_px) return;

  for (int py = offset; py < rtdata.fSize_py; py += width) {
    unsigned char *line = &output[4 * py * rtdata.fSize_px];

    adept::Color_t pixel_color = Raytracer::RaytraceOne(rtdata, rays, px, py, 0);

    line[4 * px + 0] = pixel_color.fComp.red;
    line[4 * px + 1] = pixel_color.fComp.green;
    line[4 * px + 2] = pixel_color.fComp.blue;
    line[4 * px + 3] = 255;
  }
}

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

  (*rays)[ray_index] = *ray;
  adept::Color_t pixel_color = Raytracer::RaytraceOne(rtdata, rays, global_px, global_py, ray->index);

  tile_out[pixel_index + 0] = pixel_color.fComp.red;
  tile_out[pixel_index + 1] = pixel_color.fComp.green;
  tile_out[pixel_index + 2] = pixel_color.fComp.blue;
  tile_out[pixel_index + 3] = 255;
}

void RenderImageLines(cuda::RaytracerData_t *rtdata, unsigned char *output)
{
#define NSTREAMS 4
  cudaStream_t streams[NSTREAMS];
  unsigned char *buffer = nullptr;

  for (int i = 0; i < NSTREAMS; ++i)
    checkCudaErrors(cudaStreamCreate(&streams[i]));

  checkCudaErrors(cudaMalloc((void **)&buffer, 4 * sizeof(unsigned char) * rtdata->fSize_px * rtdata->fSize_py));
  checkCudaErrors(cudaDeviceSynchronize());

  dim3 blocks((rtdata->fSize_px >> 5) + 1), threads(32);

  for (int iy = 0; iy < rtdata->fSize_py; ++iy)
    RenderLine<<<blocks, threads, 0, streams[iy % NSTREAMS]>>>(*rtdata, iy, buffer + 4 * iy * rtdata->fSize_px);

  checkCudaErrors(cudaMemcpy(output, buffer, 4 * sizeof(unsigned char) * rtdata->fSize_px * rtdata->fSize_py,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(buffer));
  checkCudaErrors(cudaGetLastError());
}

void RenderImageInterlaced(cuda::RaytracerData_t *rtdata, unsigned char *output)
{
#define NSTREAMS 4
  cudaStream_t streams[NSTREAMS];
  unsigned char *buffer = nullptr;

  for (int i = 0; i < NSTREAMS; ++i)
    checkCudaErrors(cudaStreamCreate(&streams[i]));

  checkCudaErrors(cudaMalloc((void **)&buffer, 4 * sizeof(unsigned char) * rtdata->fSize_px * rtdata->fSize_py));
  checkCudaErrors(cudaDeviceSynchronize());

  dim3 blocks((rtdata->fSize_px >> 5) + 1), threads(32);

  for (int i = 0; i < NSTREAMS; ++i)
    RenderInterlaced<<<blocks, threads, 0, streams[i % NSTREAMS]>>>(*rtdata, i, NSTREAMS, buffer);

  checkCudaErrors(cudaMemcpy(output, buffer, 4 * sizeof(unsigned char) * rtdata->fSize_px * rtdata->fSize_py,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(buffer));
  checkCudaErrors(cudaGetLastError());
}

// subdivide image in 16 tiles and launch each tile on a separate CUDA stream
void RenderTiledImage(adept::BlockData<Ray_t> *rays, cuda::RaytracerData_t *rtdata, unsigned char *output_buffer,
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
    checkCudaErrors(cudaStreamCreate(&streams[i]));

  for (int i = 0; i < 16; ++i) {
    // allocate tile on host and device
    checkCudaErrors(cudaMalloc((void **)&tile_device_in[i], tile_size_x * tile_size_y * sizeof(cuda::Ray_t)));
    checkCudaErrors(cudaMalloc((void **)&tile_device_out[i], 4 * tile_size_x * tile_size_y));
    // CUDA streams require host memory to be pinned
    checkCudaErrors(cudaMallocHost(&tile_host[i], 4 * tile_size_x * tile_size_y));
  }

  // wait for memory to reach GPU before launching kernels
  checkCudaErrors(cudaDeviceSynchronize());

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

  checkCudaErrors(cudaDeviceSynchronize());

  // copy back rendered tile to system memory
  for (int ix = 0; ix < 4; ++ix) {
    for (int iy = 0; iy < 4; ++iy) {
      int idx = 4 * ix + iy;
      checkCudaErrors(cudaMemcpyAsync(tile_host[idx], tile_device_out[idx], (size_t)4 * tile_size_x * tile_size_y,
                                      cudaMemcpyDeviceToHost, streams[iy]));
      checkCudaErrors(cudaFree(tile_device_in[idx]));
      checkCudaErrors(cudaFree(tile_device_out[idx]));
    }
  }

  // ensure all tiles have been copied back
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

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
      checkCudaErrors(cudaFreeHost(tile_host[idx]));
    }
  }
  checkCudaErrors(cudaGetLastError());
}

int RaytraceBenchmarkGPU(cuda::RaytracerData_t *rtdata, bool use_tiles, int block_size)
{
  using RayBlock     = adept::BlockData<Ray_t>;
  using RayAllocator = copcore::VariableSizeObjAllocator<RayBlock, copcore::BackendType::CUDA>;
  using Launcher_t     = copcore::Launcher<copcore::BackendType::CUDA>;
  using StreamStruct   = copcore::StreamType<copcore::BackendType::CUDA>;
  using Stream_t       = typename StreamStruct::value_type;

  // Allocate ray data and output data on the device
  size_t statesize = vecgeom::NavStateIndex::SizeOfInstance(rtdata->fMaxDepth);
  size_t raysize   = Ray_t::SizeOfInstance();
  printf(" State size is %lu, ray size is %lu\n", statesize, raysize);

  printf("=== Allocating %.3f MB of ray data on the device\n", (float)rtdata->fNrays * raysize / 1048576);
  // char *input_buffer_gpu = nullptr;
  char *input_buffer = new char[rtdata->fNrays * raysize];
  checkCudaErrors(cudaMalloc((void **)&input_buffer, rtdata->fNrays * raysize));

  unsigned char *output_buffer = nullptr;
  checkCudaErrors(cudaMalloc((void **)&output_buffer, 4 * sizeof(unsigned char) * rtdata->fSize_px * rtdata->fSize_py));

  // Load and synchronize the geometry on the GPU
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  cudaManager.LoadGeometry((vecgeom::cxx::VPlacedVolume *)rtdata->fWorld);
  cudaManager.Synchronize();

  // CudaManager is altering the stack size... setting an appropriate value
  size_t def_stack_limit = 0, def_heap_limit = 0;
  cudaDeviceGetLimit(&def_stack_limit, cudaLimitStackSize);
  cudaDeviceGetLimit(&def_heap_limit, cudaLimitMallocHeapSize);
  std::cout << "=== cudaLimitStackSize = " << def_stack_limit << "  cudaLimitMallocHeapSize = " << def_heap_limit
            << std::endl;
  auto err = cudaDeviceSetLimit(cudaLimitStackSize, 8192);
  cudaDeviceGetLimit(&def_stack_limit, cudaLimitStackSize);
  std::cout << "=== CUDA thread stack size limit set now to: " << def_stack_limit << std::endl;

  auto gpu_world = cudaManager.world_gpu();
  assert(gpu_world && "GPU world volume is a null pointer");

  // initialize BlockData of Ray_t structure
  int capacity = 1 << 20;
  RayAllocator hitAlloc(capacity);
  RayBlock *rays = hitAlloc.allocate(1);

  // Boilerplate to get the pointers to the device functions to be used
  COPCORE_CALLABLE_DECLARE(generateFunc, generateRays);

  // Create a stream to work with. On the CPU backend, this will be equivalent with: int stream = 0;
  Stream_t stream;
  StreamStruct::CreateStream(stream);

  // Allocate slots for the BlockData
  Launcher_t generate(stream);
  generate.Run(generateFunc, capacity, {0, 0}, rays);

  generate.WaitStream();

  // Initialize the navigation state for the view point
  vecgeom::NavStateIndex vpstate;
  LoopNavigator::LocatePointIn(rtdata->fWorld, rtdata->fStart, vpstate, true);
  rtdata->fVPstate = vpstate;
  rtdata->fWorld   = gpu_world;

  rtdata->Print();

  unsigned char *image_buffer = new unsigned char[4 * rtdata->fSize_px * rtdata->fSize_py];

  vecgeom::Stopwatch timer;
  timer.Start();

  cudaProfilerStart();

  if (use_tiles) {
    RenderTiledImage(rays, rtdata, image_buffer, block_size);
  } else {
    dim3 threads(block_size, block_size);
    dim3 blocks(rtdata->fSize_px / block_size + 1, rtdata->fSize_py / block_size + 1);
    RenderKernel<<<blocks, threads>>>(rays, *rtdata, input_buffer, output_buffer);
    checkCudaErrors(
        cudaMemcpy(image_buffer, output_buffer, 4 * rtdata->fSize_px * rtdata->fSize_py, cudaMemcpyDeviceToHost));
  }

  cudaProfilerStop();

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  auto time_gpu = timer.Stop();
  std::cout << "Time on GPU: " << time_gpu << "\n";

  write_ppm("output.ppm", image_buffer, rtdata->fSize_px, rtdata->fSize_py);

  delete[] image_buffer;

  checkCudaErrors(cudaFree(input_buffer));
  checkCudaErrors(cudaFree(output_buffer));
  return 0;
}
