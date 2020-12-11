// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "Raytracer.h"

#include <AdePT/BlockData.h>

#include <VecGeom/base/Global.h>

inline namespace COPCORE_IMPL {

// Alocate slots for the BlockData
__host__ __device__
void generateRays(int id, adept::BlockData<Ray_t> *rays)
{
  auto ray = rays->NextElement();
  if (!ray) COPCORE_EXCEPTION("generateRays: Not enough space for rays");

  ray->index = id;
}

COPCORE_CALLABLE_FUNC(generateRays)

__host__ __device__
void renderKernels(int id, adept::BlockData<Ray_t> *rays, const RaytracerData_t &rtdata, NavIndex_t *input_buffer,
                   NavIndex_t *output_buffer)
{
  // Propagate all rays and write out the image on the backend
  // size_t n10  = 0.1 * rtdata.fNrays;

  int ray_index = id;

  int px = 0;
  int py = 0;

  if (ray_index) {
    px = ray_index % rtdata.fSize_px;
    py = ray_index / rtdata.fSize_px;
  }

  if ((px >= rtdata.fSize_px) || (py >= rtdata.fSize_py)) return;

  // fprintf(stderr, "P3\n%d %d\n255\n", fSize_px, fSize_py);
  // if ((ray_index % n10) == 0) printf("%lu %%\n", 10 * ray_index / n10);
  Ray_t *ray = (Ray_t *)(input_buffer + ray_index * sizeof(Ray_t));
  ray->index = ray_index;

  (*rays)[ray_index] = *ray;

  auto pixel_color = Raytracer::RaytraceOne(rtdata, rays, px, py, ray->index);

  int pixel_index                = 4 * ray_index;
  output_buffer[pixel_index + 0] = pixel_color.fComp.red;
  output_buffer[pixel_index + 1] = pixel_color.fComp.green;
  output_buffer[pixel_index + 2] = pixel_color.fComp.blue;
  output_buffer[pixel_index + 3] = 255;
}
COPCORE_CALLABLE_FUNC(renderKernels)

} // End namespace COPCORE_IMPL
