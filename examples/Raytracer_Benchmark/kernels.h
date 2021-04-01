// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "Raytracer.h"

#include <AdePT/BlockData.h>
#include <AdePT/SparseVector.h>

#include <VecGeom/base/Global.h>

inline namespace COPCORE_IMPL {

constexpr int VectorSize = 1 << 22;
using Vector_t           = adept::SparseVector<Ray_t, VectorSize>;

// Add color in the vector of color (ascending order)
__host__ __device__ void add_color(int index, int rays_per_pixel, adept::Atomic_t<unsigned int> *color,
                                   unsigned int pixel_color)
{
  for (int i = index; i < index + rays_per_pixel; ++i) {
    if (color[i].load() == 0) {
      color[i].store(pixel_color);
      return;
    } else if (pixel_color < color[i].load()) {
      unsigned int x = color[i].load();
      color[i].store(pixel_color);
      pixel_color = x;
    }
  }
}

// Add initial rays in SparseVector
__host__ __device__ void generateRays(int id, const RaytracerData_t &rtdata)
{
  Ray_t *ray      = rtdata.sparse_rays[0].next_free();
  ray->index      = id;
  ray->generation = 0;
  ray->intensity.store(1.);

  Raytracer::InitRay(rtdata, *ray);
}

COPCORE_CALLABLE_FUNC(generateRays)

__host__ __device__ void renderKernels(int id, const RaytracerData_t &rtdata, int generation, int rays_per_pixel,
                                       adept::Atomic_t<unsigned int> *color)
{
  // Propagate all rays and write out the image on the backend
  if (!(rtdata.sparse_rays)[generation].is_used(id)) return;

  Ray_t &ray = rtdata.sparse_rays[generation][id];

  auto pixel_color = Raytracer::RaytraceOne(rtdata, ray, generation);
  if (pixel_color.fColor == 0) return;

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

COPCORE_CALLABLE_FUNC(print_vector)

// Check if there are rays in containers
__host__ bool check_used_cpp(const RaytracerData_t &rtdata, int no_generations)
{

  auto rays_containers = rtdata.sparse_rays;

  for (int i = 0; i < no_generations; ++i) {
    if (rays_containers[i].size_used() > 0) return true;
  }

  return false;
}

} // End namespace COPCORE_IMPL