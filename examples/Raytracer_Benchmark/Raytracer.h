// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/// \file Raytracer.h
/// \author Andrei Gheata (andrei.gheata@cern.ch)
/// Adapted from VecGeom for AdePT by antonio.petre@spacescience.ro

#ifndef RT_BENCHMARK_RAYTRACER_H_
#define RT_BENCHMARK_RAYTRACER_H_

#include "Color.h"

#include <CopCore/Global.h>
#include <CopCore/PhysicalConstants.h>
#include <AdePT/base/BlockData.h>
#include <AdePT/base/SparseVector.h>

#include <VecGeom/base/Global.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavStateIndex.h>

#ifdef COPCORE_CUDA_COMPILER
#include <VecGeom/backend/cuda/Interface.h>
#endif

inline namespace COPCORE_IMPL {

enum ERTmodel { kRTxray = 0, kRTspecular, kRTtransparent, kRTfresnel };
enum ERTView { kRTVparallel = 0, kRTVperspective };

struct MyMediumProp {
  ERTmodel material         = kRTxray;
  adept::Color_t fObjColor  = 0;
  float refr_index          = 0;
  float transparency_per_cm = 0;
};

struct Ray_t {
  using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const *;
  using Precision          = vecgeom::Precision;

  vecgeom::Vector3D<Precision> fPos;
  vecgeom::Vector3D<Precision> fDir;
  vecgeom::NavStateIndex fCrtState;     ///< navigation state for the current volume
  vecgeom::NavStateIndex fNextState;    ///< navigation state for the next volume
  VPlacedVolumePtr_t fVolume = nullptr; ///< current volume
  int fNcrossed              = 0;       ///< number of crossed boundaries
  adept::Color_t fColor      = 0;       ///< pixel color
  bool fDone                 = false;   ///< done flag
  int index                  = -1;      ///< index flag
  adept::Atomic_t<float> intensity;     ///< intensity flag
  int generation  = -1;                 ///< generation flag (used for kRTfresnel model)
  bool reflection = false;              ///< mark reflected ray

  __host__ __device__ static Ray_t *MakeInstanceAt(void *addr) { return new (addr) Ray_t(); }

  __host__ __device__ Ray_t() {}

  __host__ __device__ static size_t SizeOfInstance() { return sizeof(Ray_t); }

  __host__ __device__ vecgeom::Vector3D<Precision> Reflect(vecgeom::Vector3D<Precision> const &normal)
  {
    return (fDir - 2 * fDir.Dot(normal) * normal);
  }

  __host__ __device__ void Print()
  {
    printf("ray %d: done=%d gen=%d ncrossed=%d refl=%d point=(%g, %g, %g) dir=(%g, %g, %g) vol=%p color=%u crt:", index,
           int(fDone), generation, fNcrossed, int(reflection), fPos[0], fPos[1], fPos[2], fDir[0], fDir[1], fDir[2],
           fVolume, fColor.GetColor());
    fCrtState.Print();
  }

  __host__ __device__ void UpdateToNextVolume()
  {
    auto tmpstate = fCrtState;
    fCrtState     = fNextState;
    fNextState    = tmpstate;
    fVolume       = fCrtState.Top();
  }

  __host__ __device__ vecgeom::Vector3D<Precision> Refract(vecgeom::Vector3D<Precision> const &normal, float ior1,
                                                           float ior2, bool &totalreflect)
  {
    // ior1, ior2 are the refraction indices of the exited and entered volumes respectively
    float cosi = fDir.Dot(normal);
    float eta  = ior1 / ior2;

    if (cosi >= 0) eta = 1. / eta;

    vecgeom::Vector3D<Precision> n = (cosi < 0) ? normal : -normal;
    cosi                           = vecCore::math::Abs(cosi);
    float k                        = 1 - eta * eta * (1 - cosi * cosi);
    vecgeom::Vector3D<Precision> refracted;
    if (k < 0) {
      totalreflect = true;
    } else {
      totalreflect = false;
      refracted    = eta * fDir + (eta * cosi - vecCore::math::Sqrt(k)) * n;
    }
    return refracted;
  }

  __host__ __device__ void TraverseTransparentLayer(float transparency_per_cm, float step, adept::Color_t object_color)
  {
    // Calculate transmittance = I/I0 = exp(ktr * step);
    // where: ktr = log(transparency_per_cm)
    float ktr           = log(transparency_per_cm);
    float transmittance = exp(ktr * step / copcore::units::cm);
    auto blend_color    = object_color;
    blend_color *= (1 - transmittance);
    fColor += blend_color;
    float x = intensity.load();
    x *= transmittance;
    intensity.store(x);
  }

  __host__ __device__ void Fresnel(vecgeom::Vector3D<Precision> const &normal, float ior1, float ior2, float &kr)
  {
    float cosi = fDir.Dot(normal);
    if (cosi > 0) {
      float x = ior1;
      ior1    = ior2;
      ior2    = x;
    }
    float eta = ior1 / ior2;

    // Vector3D<Precision> n = (cosi < 0) ? normal : -normal;
    cosi = vecCore::math::Abs(cosi);
    // Compute sini using Snell's law
    float sint = eta * vecCore::math::Sqrt(vecCore::math::Max(0.f, 1.f - cosi * cosi));
    // Total internal reflection
    if (sint >= 1) {
      kr = 1;
    } else {
      float cost = vecCore::math::Sqrt(1 - sint * sint);
      float Rs   = ((ior2 * cosi) - (ior1 * cost)) / ((ior2 * cosi) + (ior1 * cost));
      float Rp   = ((ior1 * cosi) - (ior2 * cost)) / ((ior1 * cosi) + (ior2 * cost));
      kr         = (Rs * Rs + Rp * Rp) / 2;
    }
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;
  }
};

struct RaytracerData_t {

  using VPlacedVolumePtr_t    = vecgeom::VPlacedVolume const *;
  static const int VectorSize = 1 << 22;
  using Vector_t              = adept::SparseVector<Ray_t, VectorSize>;
  using Vector_t_int          = adept::SparseVector<int, VectorSize>;

  Precision fScale     = 0;                   ///< Scaling from pixels to world coordinates
  Precision fShininess = 1.;                  ///< Shininess exponent in the specular model
  Precision fZoom      = 1.;                  ///< Zoom with respect to the default view
  vecgeom::Vector3D<Precision> fSourceDir;    ///< Light source direction
  vecgeom::Vector3D<Precision> fScreenPos;    ///< Screen position
  vecgeom::Vector3D<Precision> fStart;        ///< Eye position in perspectove mode
  vecgeom::Vector3D<Precision> fDir;          ///< Start direction of all rays in parallel view mode
  vecgeom::Vector3D<Precision> fUp;           ///< Up vector in the shooting rectangle plane
  vecgeom::Vector3D<Precision> fRight;        ///< Right vector in the shooting rectangle plane
  vecgeom::Vector3D<Precision> fLeftC;        ///< left-down corner of the ray shooting rectangle
  int fVerbosity           = 0;               ///< Verbosity level
  int fNrays               = 0;               ///< Number of rays left to propagate
  int fSize_px             = 1024;            ///< Image pixel size in x
  int fSize_py             = 1024;            ///< Image pixel size in y
  adept::Color_t fBkgColor = 0xFFFFFF80;      ///< Background color
  ERTmodel fModel          = kRTxray;         ///< Selected RT model
  ERTView fView            = kRTVperspective; ///< View type
  bool fReflection         = false;           ///< Reflection model
  bool fApproach           = false;           ///< approach solid to bbox before calculating distance

  VPlacedVolumePtr_t fWorld = nullptr;     ///< World volume
  vecgeom::NavStateIndex fVPstate;         ///< Navigation state corresponding to the viewpoint
  MyMediumProp const *fMediaProp{nullptr}; ///< Array of media properties per volume

  Vector_t *sparse_rays         = nullptr; ///< pointer to the rays containers
  Vector_t_int *sparse_int      = nullptr; ///<
  Vector_t_int *sparse_int_copy = nullptr; ///<

  __host__ __device__ void Print();
};

/// \brief Raytracing a logical volume content using a given model
///
/// In order to run a benchmark, a world volume must be provided to the
/// raytracer.
namespace Raytracer {

using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const *;

/// \param world Mother volume containing daughters that will be benchmarked.
/// \param screen_position position of the screen in world coordinates, rays are starting from a plane passing through
/// it, normal vector pointing to the origin of the world reference frame \param up_vector the projection of this
/// vector on the camera plane determines the 'up' direction of the image \param img_size_px image size on X in pixels
/// \param img_size_py image size on Y in pixels
__host__ __device__ void InitializeModel(VPlacedVolumePtr_t world, RaytracerData_t &data);

__host__ __device__ void ApplyRTmodel(Ray_t &ray, Precision step, RaytracerData_t const &rtdata, int generation);

/// \brief Entry point to propagate all rays
// __host__ __device__ void PropagateRays(adept::BlockData<Ray_t> *rays, RaytracerData_t &data, unsigned char
// *rays_buffer, unsigned char *output_buffer);

__host__ __device__ void InitRay(RaytracerData_t const &rtdata, Ray_t &ray);

__host__ __device__ adept::Color_t RaytraceOne(RaytracerData_t const &rtdata, Ray_t &ray, int generation);

} // End namespace Raytracer

} // End namespace COPCORE_IMPL

void write_ppm(std::string filename, NavIndex_t *buffer, int px, int py);
// void RenderCPU(VPlacedVolume const *const world, int px, int py, int maxdepth);

#endif // RT_BENCHMARK_RAYTRACER_H_
