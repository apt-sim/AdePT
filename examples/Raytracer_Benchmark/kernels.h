// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "Raytracer.h"

#include <AdePT/BlockData.h>
#include <AdePT/SparseVector.h>

#include <VecGeom/base/Global.h>

inline namespace COPCORE_IMPL {

constexpr int VectorSize = 1 << 22;
using Vector_t           = adept::SparseVector<Ray_t, VectorSize>;
using Vector_t_int    = adept::SparseVector<int, VectorSize>;
using VectorInterface_int = adept::SparseVectorInterface<int>;

// Add color in the vector of color
__host__ __device__ void add_color(int index, int rays_per_pixel, adept::Atomic_t<unsigned int> *color,
                                   unsigned int pixel_color)
  {
    for (int i = index; i < index + rays_per_pixel; ++i) {
      unsigned int x = color[i].load();
      if (x == 0)
        if (color[i].compare_exchange_strong(x, pixel_color))
          return;
    }
  }

// Sort colors (basic implementation)
__host__ __device__ void sortColor(int id, int rays_per_pixel, adept::Atomic_t<unsigned int> *color)
{
  int index = id*rays_per_pixel;
  
  for (int i = index; i < index + rays_per_pixel; ++i) {
    if (color[i].load() == 0) {
      return;
    }
    for (int j = i+1; j < index + rays_per_pixel; ++j)
    {
  
      if (color[j].load() == 0) {
        break;
      }
      else if (color[i].load() > color[j].load() && color[j].load() > 0) {
        unsigned int x = color[i].load();
        color[i].store(color[j].load());
        color[j].store(x);
      }
    }
  }
}
  
COPCORE_CALLABLE_FUNC(sortColor)  

// Add initial rays in SparseVector
__host__ __device__ void generateRays(int id, const RaytracerData_t &rtdata)
{
  Ray_t *ray      = rtdata.sparse_rays[0].next_free();
  rtdata.sparse_int->next_free(id);
  ray->index      = id;
  ray->generation = 0;
  ray->intensity.store(1.);

  Raytracer::InitRay(rtdata, *ray);
}

COPCORE_CALLABLE_FUNC(generateRays)

__host__ __device__ void renderKernels(int id, const RaytracerData_t &rtdata, int generation, int rays_per_pixel,
                                       adept::Atomic_t<unsigned int> *color)
{

  int index = rtdata.sparse_int[0][id];

  Ray_t &ray = rtdata.sparse_rays[0][index];
  
  auto pixel_color = Raytracer::RaytraceOne(rtdata, ray, generation);

  // if ray is still alive, add the index in the container
  if (!ray.fDone)
    rtdata.sparse_int_copy->next_free(rtdata.sparse_int[0][id]);

  // if ray is not alive, add the color in container
  else if (pixel_color.fColor != 0)
    add_color(rays_per_pixel * ray.index, rays_per_pixel, color, pixel_color.fColor);
}

COPCORE_CALLABLE_FUNC(renderKernels)

// Print information about containers
__host__ __device__ void print_vector(Vector_t *vect)
{
  printf("=== vect: fNshared=%lu/%lu fNused=%lu fNbooked=%lu - shared=%.1f%% sparsity=%.1f%%\n", vect->size(),
         vect->capacity(), vect->size_used(), vect->size_booked(), 100. * vect->get_shared_fraction(),
         100. * vect->get_sparsity());
}

// Print information about containers
__host__ __device__ void print_vector(Vector_t_int *vect)
{
  printf("=== vect: fNshared=%lu/%lu fNused=%lu fNbooked=%lu - shared=%.1f%% sparsity=%.1f%%\n", vect->size(),
         vect->capacity(), vect->size_used(), vect->size_booked(), 100. * vect->get_shared_fraction(),
         100. * vect->get_sparsity());
}

// Add elements from sel_vector_d in container sparse_vector
__host__ void add_indices_cpp(Vector_t_int *sparse_vector, unsigned *sel_vector_d, unsigned *nselected_hd)
{
  for (int j = 0; j < *nselected_hd; ++j)
    sparse_vector->next_free(sel_vector_d[j]);
}

} // End namespace COPCORE_IMPL