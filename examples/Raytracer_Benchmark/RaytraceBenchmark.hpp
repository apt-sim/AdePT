// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "kernels.h"
#include "Raytracer.h"

#include <CopCore/Global.h>
#include <AdePT/ArgParser.h>
#include <AdePT/SparseVector.h>
#include <AdePT/BVHNavigator.h>

#include <VecGeom/base/Global.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/base/Stopwatch.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/navigation/NavStatePath.h>

#include <atomic>
#include <vector>

#ifdef VECGEOM_GDML
#include <VecGeom/gdml/Frontend.h>
#endif

unsigned int size_used_cuda(Vector_t *sparse_vector); 
unsigned int size_used_cuda(Vector_t_int *sparse_vector); 
    
void clear_sparse_vector_cuda(Vector_t_int *vector);

void add_indices_cuda(Vector_t_int *sparse_vector, unsigned *sel_vector_d, unsigned *nselected_hd);

void print_vector_cuda(RaytracerData_t *rtdata, int no_generations);

void initiliazeCudaWorld(RaytracerData_t *rtdata, const MyMediumProp *volume_container,
                         std::vector<vecgeom::cxx::LogicalVolume *> logicalvolume);

void RenderTiledImage(RaytracerData_t *rtdata, NavIndex_t *output_buffer, int generation, int block_size);

bool check_used_cuda(RaytracerData_t *rtdata, int no_generations);

void InitBVH(bool);

template <copcore::BackendType backend>
void InitRTdata(RaytracerData_t *rtdata, const MyMediumProp *volume_container, int no_generations,
                std::vector<vecgeom::cxx::LogicalVolume *> logicalvolumes)
{
  Vector_t *x;
  Vector_t_int *sparse_int_copy, *sparse_int;

  if (backend == copcore::BackendType::CUDA) {
    initiliazeCudaWorld((RaytracerData_t *)rtdata, volume_container, logicalvolumes);
    COPCORE_CUDA_CHECK(cudaMalloc(&x, no_generations * sizeof(Vector_t)));

    COPCORE_CUDA_CHECK(cudaMalloc(&sparse_int_copy, no_generations * sizeof(Vector_t_int))); 
    COPCORE_CUDA_CHECK(cudaMalloc(&sparse_int, no_generations * sizeof(Vector_t_int)));

  } else {
    vecgeom::NavStateIndex vpstate;
    BVHNavigator::LocatePointIn(rtdata->fWorld, rtdata->fStart, vpstate, true);
    rtdata->fVPstate = vpstate;

    // COPCORE_CUDA_CHECK(cudaMallocManaged(&x, no_generations * sizeof(Vector_t)));
    x = (Vector_t *)malloc(no_generations * sizeof(Vector_t));  
    sparse_int_copy = (Vector_t_int *)malloc(no_generations * sizeof(Vector_t_int)); 
    sparse_int          = (Vector_t_int *)malloc(no_generations * sizeof(Vector_t_int));
  }

  for (int i = 0; i < no_generations; ++i) {
    Vector_t::MakeInstanceAt(&x[i]);
    Vector_t_int::MakeInstanceAt(&sparse_int_copy[i]);
    Vector_t_int::MakeInstanceAt(&sparse_int[i]);
  }

  rtdata->sparse_rays = x;
  rtdata->sparse_int_copy  = sparse_int_copy; 
  rtdata->sparse_int = sparse_int;
}

// Print information about containers
template <copcore::BackendType backend>
void print(RaytracerData_t *rtdata, int no_generations)
{
  if (backend == copcore::BackendType::CUDA) {
    print_vector_cuda(rtdata, no_generations);
  } else {
    for (int i = 0; i < no_generations; ++i)
      print_vector(&rtdata->sparse_rays[i]);
  }
}

// Add elements from sel_vector in sparse_vector
template <copcore::BackendType backend> 
void add_indices(Vector_t_int *sparse_vector, unsigned *sel_vector, unsigned *nselected_hd)  
{ 
  if (backend == copcore::BackendType::CUDA) {  
    add_indices_cuda(sparse_vector, sel_vector, nselected_hd); 
  } else {  
    add_indices_cpp(sparse_vector, sel_vector, nselected_hd);
  } 
} 
    
// Return number of elements in use
template <copcore::BackendType backend> 
unsigned int size_used(Vector_t *sparse_vector) 
{ 
  if (backend == copcore::BackendType::CUDA) {  
    return size_used_cuda(sparse_vector); 
  } else {  
  return (unsigned int) sparse_vector->size_used(); 
  } 
} 
    
// Return number of elements in use
template <copcore::BackendType backend> 
unsigned int size_used(Vector_t_int *sparse_vector) 
{ 
  if (backend == copcore::BackendType::CUDA) {  
    return size_used_cuda(sparse_vector); 
  } else {  
    return (unsigned int) sparse_vector->size_used(); 
  } 
}

// Clear sparse vector
template <copcore::BackendType backend> 
void clear_sparse_vector(Vector_t_int *vector) {  
  if (backend == copcore::BackendType::CUDA) {  
    clear_sparse_vector_cuda(vector);  
  }   
  else {  
    vector->clear();
  } 
}  

// Check if there are rays in containers
template <copcore::BackendType backend>
bool check_used(RaytracerData_t *rtdata, int no_generations)
{
  if (backend == copcore::BackendType::CUDA) {
    return check_used_cuda(rtdata, no_generations);
  } else {
    return (rtdata->sparse_int->size() > 0);
  }
}

template <copcore::BackendType backend>
int runSimulation(const MyMediumProp *volume_container, const vecgeom::cxx::VPlacedVolume *world,
                  std::vector<vecgeom::cxx::LogicalVolume *> logicalvolumes, int argc, char *argv[])
{
  // image size in pixels
  OPTION_INT(px, 1840);
  OPTION_INT(py, 512);

  // RT model as in { kRTxray = 0, kRTspecular, kRTtransparent, kRTdiffuse };
  OPTION_INT(model, 2);

  // RT view as in { kRTVparallel = 0, kRTVperspective };
  OPTION_INT(view, 1);

  // Use reflection
  OPTION_BOOL(reflection, 0);

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

  // Light color - in RGBA chars compressed into an unsigned integer
  OPTION_INT(bkgcol, 0xFFFFFF80); // white (keep 80 as alpha channel for correct color blending)

  OPTION_INT(use_tiles, 0);  // run on GPU in tiled mode
  OPTION_INT(block_size, 8); // run on GPU in tiled mode

  copcore::Allocator<RaytracerData_t, backend> rayAlloc;
  RaytracerData_t *rtdata = rayAlloc.allocate(1);

  rtdata->fScreenPos.Set(screenx, screeny, screenz);
  rtdata->fUp.Set(upx, upy, upz);
  rtdata->fZoom       = zoom;
  rtdata->fModel      = (ERTmodel)model;
  rtdata->fView       = (ERTView)view;
  rtdata->fSize_px    = px;
  rtdata->fSize_py    = py;
  rtdata->fBkgColor   = bkgcol;
  rtdata->fReflection = reflection;

  Raytracer::InitializeModel((Raytracer::VPlacedVolumePtr_t)world, *rtdata);

  constexpr int VectorSize = 1 << 22;
  const int no_pixels      = rtdata->fSize_py * rtdata->fSize_px;

  using Launcher_t   = copcore::Launcher<backend>;
  using StreamStruct = copcore::StreamType<backend>;
  using Stream_t     = typename StreamStruct::value_type;
  using Vector_t     = adept::SparseVector<Ray_t, VectorSize>; // 1<<16 is the default vector size if parameter omitted
  using VectorInterface = adept::SparseVectorInterface<Ray_t>;
  using Vector_t_int    = adept::SparseVector<int, VectorSize>; 
  using VectorInterface_int = adept::SparseVectorInterface<int>;

  int no_generations = 1;
  // if (rtdata->fReflection) no_generations = 10;
  int rays_per_pixel = 10; // maximum number of rays per pixel

  InitRTdata<backend>(rtdata, volume_container, no_generations, logicalvolumes);

  rtdata->Print();

  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  bool on_gpu = backend == copcore::BackendType::CUDA;
  InitBVH(on_gpu); // note: needs to be called after geometry has been uploaded to the GPU

  // Boilerplate to get the pointers to the device functions to be used
  COPCORE_CALLABLE_DECLARE(generateRaysFunc, generateRays);
  COPCORE_CALLABLE_DECLARE(renderkernelFunc, renderKernels);
  COPCORE_CALLABLE_DECLARE(sortColorFunc, sortColor); 
    
  unsigned *sel_vector_d; 
  COPCORE_CUDA_CHECK(cudaMallocManaged(&sel_vector_d, VectorSize * sizeof(unsigned)));  
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());  
    
  unsigned *nselected_hd; 
  COPCORE_CUDA_CHECK(cudaMallocManaged(&nselected_hd, sizeof(unsigned))); 
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  // Create a stream to work with.
  Stream_t stream;
  StreamStruct::CreateStream(stream);

  Launcher_t generate(stream);

  // Allocate and initialize all rays on the host
  size_t raysize = Ray_t::SizeOfInstance();
  printf("=== Allocating %.3f MB of ray data on the %s\n", (float)rtdata->fNrays * raysize / 1048576,
         copcore::BackendName(backend));

  copcore::Allocator<NavIndex_t, backend> ucharAlloc;
  NavIndex_t *output_buffer = ucharAlloc.allocate(4 * rtdata->fNrays * sizeof(NavIndex_t));

  using Atomic_t = adept::Atomic_t<unsigned int>;

  adept::Atomic_t<unsigned int> *color;
  COPCORE_CUDA_CHECK(cudaMallocManaged(&color, rays_per_pixel * no_pixels * sizeof(adept::Atomic_t<unsigned int>)));

  vecgeom::Stopwatch timer;
  timer.Start();

  // Add initial rays in container
  generate.Run(generateRaysFunc, no_pixels, {0, 0}, *rtdata);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  int step = 0; 
  int size = 0;

  if (backend == copcore::BackendType::CUDA && use_tiles) {
    RenderTiledImage((RaytracerData_t *)rtdata, output_buffer, 0, block_size);
  } else {
    Launcher_t renderKernel(stream);
    while (check_used<backend>(rtdata, no_generations)) {  

      size = size_used<backend>(rtdata->sparse_int);

      renderKernel.Run(renderkernelFunc, size, {0, 0}, *rtdata, step, rays_per_pixel, color);
      COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

      std::swap(rtdata->sparse_int, rtdata->sparse_int_copy);  // swap the pointers

      clear_sparse_vector<backend>(rtdata->sparse_int_copy);
      COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

      if (rtdata->fReflection) {

        // Get indices of reflected rays
        auto select_func2 = [] __device__(int i, const VectorInterface *arr) { return ((*arr)[i].reflection == true); };
        VectorInterface::select(&(rtdata->sparse_rays[0]), select_func2, sel_vector_d, nselected_hd);
        COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

        if (*nselected_hd > 0)
          add_indices<backend>(rtdata->sparse_int, sel_vector_d, nselected_hd);
      }
      
      COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

      step++;
    }
  }

  // Sort colors (for every index)  
  generate.Run(sortColorFunc, no_pixels, {0, 0}, rays_per_pixel, color);  
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < no_pixels; i++) {
    int pixel_index                = 4 * i;
    adept::Color_t pixel           = mix_vector_color(i, color, rays_per_pixel);
    output_buffer[pixel_index + 0] = pixel.fComp.red;
    output_buffer[pixel_index + 1] = pixel.fComp.green;
    output_buffer[pixel_index + 2] = pixel.fComp.blue;
    output_buffer[pixel_index + 3] = 255;
  }

  // Print basic information about containers
  print<backend>(rtdata, no_generations);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  auto time_cpu = timer.Stop();
  std::cout << "Run time: " << time_cpu << "\n";

  // Write the output
  const char *output_name = on_gpu ? "output-gpu.ppm" : "output-cpu.ppm";
  write_ppm(output_name, output_buffer, rtdata->fSize_px, rtdata->fSize_py);

  COPCORE_CUDA_CHECK(cudaFree(rtdata->sparse_rays));
  COPCORE_CUDA_CHECK(cudaFree(color));
  COPCORE_CUDA_CHECK(cudaFree(sel_vector_d));
  COPCORE_CUDA_CHECK(cudaFree(nselected_hd));
  COPCORE_CUDA_CHECK(cudaFree(rtdata));

  return 0;
}
