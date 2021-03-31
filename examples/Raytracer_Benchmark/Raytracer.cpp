// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/// \file Raytracer.cpp
/// \author Andrei Gheata (andrei.gheata@cern.ch)
/// Adapted from VecGeom for AdePT by antonio.petre@spacescience.ro

#include "Raytracer.h"
#include "Color.h"

#include <CopCore/Global.h>
#include <AdePT/BlockData.h>
#include <AdePT/LoopNavigator.h>

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/base/Stopwatch.h>

#include <VecGeom/navigation/NavStateIndex.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <VecGeom/management/GeoManager.h>

#ifdef VECGEOM_CUDA_INTERFACE
#include <VecGeom/backend/cuda/Backend.h>
#include <VecGeom/management/CudaManager.h>
#endif

#include <random>
#include <sstream>
#include <fstream>
#include <utility>

inline namespace COPCORE_IMPL {

void RaytracerData_t::Print()
{
  printf("  screen_pos(%g, %g, %g) screen_size(%d, %d)\n", fScreenPos[0], fScreenPos[1], fScreenPos[2], fSize_px,
         fSize_py);
  printf("  light_dir(%g, %g, %g) light_color(0x%08x)\n", fSourceDir[0], fSourceDir[1], fSourceDir[2],
         fBkgColor.fColor);
  printf("  zoom_factor(%g) rt_model(%d) rt_view(%d)\n", fZoom, (int)fModel, (int)fView);
  printf("  viewpoint_state: ");
  fVPstate.Print();
}
namespace Raytracer {

void InitializeModel(vecgeom::VPlacedVolume const *world, RaytracerData_t &rtdata)
{
  using namespace vecCore::math;

  if (!world) return;
  rtdata.fWorld = (vecgeom::VPlacedVolume const *)world;

  // adjust up vector, image scaling
  vecgeom::Vector3D<double> aMin, aMax, vcenter, vsize;
  world->Extent(aMin, aMax);
  vcenter = 0.5 * (aMin + aMax);
  vsize   = 0.5 * (aMax - aMin);

  double imgRadius = vsize.Mag();
  // std::cout << *(fWorld->GetLogicalVolume()->GetUnplacedVolume()) << std::endl;
  // std::cout << "vcenter =  " << vcenter << "   vsize = " << vsize << "  ingRadius = " << imgRadius << std::endl;
  assert(rtdata.fSize_px * rtdata.fSize_py > 0 && "SetWorld: image size not set");

  // Make sure the image fits the parrallel world view, leaving 20% margin
  constexpr double d0 = 1.;
  double dd           = (vcenter - rtdata.fScreenPos).Mag();
  rtdata.fScale       = 2.1 * imgRadius / Min(rtdata.fSize_px, rtdata.fSize_py) / rtdata.fZoom;
  if (rtdata.fView == kRTVperspective) rtdata.fScale *= d0 / (dd + d0);

  // Project up vector on the source plane
  rtdata.fDir = vcenter - rtdata.fScreenPos;
  rtdata.fDir.Normalize();
  rtdata.fStart = rtdata.fScreenPos - d0 * rtdata.fDir;
  rtdata.fRight = vecgeom::Vector3D<double>::Cross(rtdata.fDir, rtdata.fUp);
  rtdata.fRight.Normalize();
  rtdata.fUp = vecgeom::Vector3D<double>::Cross(rtdata.fRight, rtdata.fDir);
  rtdata.fUp.Normalize();
  rtdata.fLeftC =
      rtdata.fScreenPos - 0.5 * rtdata.fScale * (rtdata.fSize_px * rtdata.fRight + rtdata.fSize_py * rtdata.fUp);

  // Light position on top-left
  rtdata.fSourceDir = rtdata.fDir + rtdata.fUp + rtdata.fRight;
  rtdata.fSourceDir.Normalize();

  // Create navigators (only for CPU case)
  // CreateNavigators();

  // Allocate rays
  rtdata.fNrays = rtdata.fSize_px * rtdata.fSize_py;
}

void InitRay(RaytracerData_t const &rtdata, Ray_t &ray)
{
  // This is called for all rays for the first generation. We don't call this for subsequent generations.
  int px = ray.index % rtdata.fSize_px;
  int py = ray.index / rtdata.fSize_px;

  vecgeom::Vector3D<double> pos_onscreen = rtdata.fLeftC + rtdata.fScale * (px * rtdata.fRight + py * rtdata.fUp);
  vecgeom::Vector3D<double> start        = (rtdata.fView == kRTVperspective) ? rtdata.fStart : pos_onscreen;
  ray.fPos                               = start;
  ray.fDir = (rtdata.fView == kRTVperspective) ? pos_onscreen - rtdata.fStart : rtdata.fDir;
  ray.fDir.Normalize();
  ray.fColor = 0;

  if (rtdata.fView == kRTVperspective) {
    ray.fCrtState = rtdata.fVPstate;
    ray.fVolume   = (Ray_t::VPlacedVolumePtr_t)rtdata.fVPstate.Top();
  } else {
    ray.fVolume = LoopNavigator::LocatePointIn(rtdata.fWorld, ray.fPos, ray.fCrtState, true);
  }
}

adept::Color_t RaytraceOne(RaytracerData_t const &rtdata, Ray_t &ray, int generation)
{
  constexpr int kMaxTries = 10;
  constexpr double kPush  = 1.e-8;

  int itry = 0;

  while (!ray.fVolume && itry < kMaxTries) {
    auto snext = rtdata.fWorld->DistanceToIn(ray.fPos, ray.fDir);
    ray.fDone  = snext == vecgeom::kInfLength;
    if (ray.fDone) return ray.fColor;
    // Propagate to the world volume (but do not increment the boundary count)
    ray.fPos += (snext + kPush) * ray.fDir;
    ray.fVolume = LoopNavigator::LocatePointIn(rtdata.fWorld, ray.fPos, ray.fCrtState, true);

    if (ray.fVolume) {
      ray.fNextState = ray.fCrtState;
      Raytracer::ApplyRTmodel(ray, snext, rtdata, generation);
    }
  }
  ray.fDone = ray.fVolume == nullptr;
  if (ray.fDone) return ray.fColor;

  // Now propagate ray
  while (!ray.fDone) {

    auto nextvol = ray.fVolume;
    double snext = vecgeom::kInfLength;
    int nsmall   = 0;

    // if (generation == 2) {
    //   printf("1-fCrtState: ");
    //   ray.fCrtState.Print();
    //   printf("1-fNextState: ");
    //   ray.fNextState.Print();
    //   ray.fVolume->Print();
    // }

    while (nextvol == ray.fVolume && nsmall < kMaxTries) {
      snext   = LoopNavigator::ComputeStepAndPropagatedState(ray.fPos, ray.fDir, vecgeom::kInfLength, ray.fCrtState,
                                                           ray.fNextState);
      nextvol = (Ray_t::VPlacedVolumePtr_t)ray.fNextState.Top();
      ray.fPos += (snext + kPush) * ray.fDir;
      nsmall++;
    }

    if (nsmall == kMaxTries) {
      // std::cout << "error for ray (" << px << ", " << py << ")\n";
      ray.fDone  = true;
      ray.fColor = 0; // black rays for errors
      return ray.fColor;
    }

    // Apply the selected RT model
    ray.fNcrossed++;
    // Current volume should still reflect the state before boundary crossing
    ray.fVolume = nextvol;

    if (nextvol)
      Raytracer::ApplyRTmodel(ray, snext, rtdata, generation);
    else {
      // Rays exiting directly must get the color contribution from the background
      auto bkg_contrib = rtdata.fBkgColor;
      float x          = ray.intensity.load();
      bkg_contrib *= x;
      ray.fColor += bkg_contrib;
      ray.fDone = true;
    }
  }

  return ray.fColor;
}

void ApplyRTmodel(Ray_t &ray, double step, RaytracerData_t const &rtdata, int generation)
{

  auto lastvol = (Ray_t::VPlacedVolumePtr_t)ray.fCrtState.Top();
  auto nextvol = (Ray_t::VPlacedVolumePtr_t)ray.fNextState.Top();

  // Get material structure for last and next volumes
  auto medium_prop_last = (MyMediumProp *)lastvol->GetLogicalVolume()->GetBasketManagerPtr();
  auto medium_prop_next = (MyMediumProp *)nextvol->GetLogicalVolume()->GetBasketManagerPtr();

  bool exit_transparent        = medium_prop_last->material == kRTtransparent;
  bool transparent_transparent = medium_prop_next->material == kRTtransparent &&
                                 medium_prop_last->material == kRTtransparent &&
                                 (medium_prop_next->refr_index * medium_prop_last->refr_index) > 1.;

  if (exit_transparent) {
    // Reduce ray intensity due to absorbtion in the transparent volume
    if (medium_prop_last->transparency_per_cm < 1.)
      ray.TraverseTransparentLayer(medium_prop_last->transparency_per_cm, step, medium_prop_last->fObjColor);
  }

  if (transparent_transparent && rtdata.fReflection) {
    // Apply reflection/refraction only at the surface between 2 transparent volumes
    // We need to check first if the ray is entering or exiting, to determine the volume to use for the normal vector
    bool entering    = ray.fNextState.GetLevel() >= ray.fCrtState.GetLevel();
    auto norm_vol    = (entering) ? nextvol : lastvol;
    auto &norm_state = (entering) ? ray.fNextState : ray.fCrtState;

    vecgeom::Transformation3D m;
    norm_state.TopMatrix(m); // global tramsformation matrix of the touchable of interest
    auto localpoint = m.Transform(ray.fPos);
    vecgeom::Vector3D<double> norm, lnorm;
    norm_vol->GetLogicalVolume()->GetUnplacedVolume()->Normal(localpoint, lnorm);
    m.InverseTransformDirection(lnorm, norm);

    // Compute fraction of reflected light
    float kr = 0;

    ray.Fresnel(norm, medium_prop_last->refr_index, medium_prop_next->refr_index, kr);

    vecgeom::Vector3D<double> reflected, refracted;

    auto initial_int = ray.intensity.load();

    adept::Color_t col_refracted = 0, col_reflected = 0;

    // Compute reflected and refracted directions, check for total reflection
    bool totalreflect = false;
    reflected         = ray.Reflect(norm);
    refracted         = ray.Refract(norm, medium_prop_last->refr_index, medium_prop_next->refr_index, totalreflect);
    reflected.Normalize();
    refracted.Normalize();

    // Generate reflected ray, or keep the current alive in case of total reflection
    ray.generation++; // increase generation before making a copy of the ray, so daughters will inherit the new value

    if (kr * initial_int > 0.1) {
      Ray_t *reflected_ray = (totalreflect) ? &ray : rtdata.sparse_rays[ray.generation % 10]->next_free(ray);
      if (!reflected_ray) COPCORE_EXCEPTION("No available rays left.");

      // Update reflected ray direction and state
      reflected_ray->fDir = reflected;
      reflected_ray->intensity.store(kr * initial_int);
      reflected_ray->fDone = false;
      // Reflected ray stays in the same volume, no need to update to next
      reflected_ray->fVolume = lastvol;
    }

    if (!totalreflect) {
      // update current ray to be the refracted one
      float x = (1 - kr) * ray.intensity.load();
      ray.intensity.store(x);
      if (ray.intensity.load() > 0.1) {
        ray.fDir = refracted;
        ray.UpdateToNextVolume();
      } else {
        auto bkg_contrib = rtdata.fBkgColor;
        float x          = ray.intensity.load();
        bkg_contrib *= x;
        ray.fColor = bkg_contrib;
        ray.fDone  = true;
      }
    }
  }

  else if (medium_prop_next->material == kRTxray) {
    ray.UpdateToNextVolume();
  }

  else if (medium_prop_next->material == kRTspecular) { // specular reflection
    // Calculate normal at the hit point
    vecgeom::Transformation3D m;
    ray.fNextState.TopMatrix(m);
    auto localpoint = m.Transform(ray.fPos);
    vecgeom::Vector3D<double> norm, lnorm;
    nextvol->GetLogicalVolume()->GetUnplacedVolume()->Normal(localpoint, lnorm);
    m.InverseTransformDirection(lnorm, norm);
    vecgeom::Vector3D<double> refl = ray.Reflect(norm);
    refl.Normalize();
    double calf = -rtdata.fSourceDir.Dot(refl);
    // if (calf < 0) calf = 0;
    // calf                   = vecCore::math::Pow(calf, fShininess);
    auto specular_color = rtdata.fBkgColor;
    specular_color.MultiplyLightChannel(1. + 0.5 * calf);
    auto object_color = medium_prop_next->fObjColor;
    object_color.MultiplyLightChannel(1. + 0.5 * calf);

    ray.fColor += specular_color + object_color;
    float x = ray.intensity.load();
    ray.fColor *= x;
    ray.fDone = true;
  }

  else {
    ray.UpdateToNextVolume();
  }
}

/*
void PropagateRays(int id, adept::BlockData<Ray_t> *rays, const RaytracerData_t &rtdata, unsigned char *input_buffer,
                   unsigned char *output_buffer)
{
  // Propagate all rays and write out the image on the CPU
  size_t n10 = 0.1 * rtdata.fNrays;

  int ray_index = id;

  int px = 0;
  int py = 0;

  if (ray_index) {
    px = ray_index % rtdata.fSize_px;
    py = ray_index / rtdata.fSize_px;
  }

  if ((px >= rtdata.fSize_px) || (py >= rtdata.fSize_py)) return;

  // fprintf(stderr, "P3\n%d %d\n255\n", fSize_px, fSize_py);

  if ((ray_index % n10) == 0) printf("%lu %%\n", 10 * ray_index / n10);
  Ray_t *ray = (Ray_t *)(input_buffer + ray_index * sizeof(Ray_t));
  ray->index = ray_index;

  (*rays)[ray_index] = *ray;

  // auto pixel_color = RaytraceOne(rtdata, rays, px, py, ray->index);

  // int pixel_index                = 4 * ray_index;
  // output_buffer[pixel_index + 0] = pixel_color.fComp.red;
  // output_buffer[pixel_index + 1] = pixel_color.fComp.green;
  // output_buffer[pixel_index + 2] = pixel_color.fComp.blue;
  // output_buffer[pixel_index + 3] = 255;
}
*/

/*
void Raytracer::CreateNavigators()
{
  // Create all navigators.
  for (auto &lvol : vecgeom::GeoManager::Instance().GetLogicalVolumesMap()) {
    if (lvol.second->GetDaughtersp()->size() < 4) {
      lvol.second->SetNavigator(vecgeom::NewSimpleNavigator<>::Instance());
    }
    if (lvol.second->GetDaughtersp()->size() >= 5) {
      lvol.second->SetNavigator(vecgeom::SimpleABBoxNavigator<>::Instance());
    }
    if (lvol.second->GetDaughtersp()->size() >= 10) {
      lvol.second->SetNavigator(vecgeom::HybridNavigator<>::Instance());
      vecgeom::HybridManager2::Instance().InitStructure((lvol.second));
    }
    lvol.second->SetLevelLocator(vecgeom::SimpleABBoxLevelLocator::GetInstance());
  }
}
*/

} // End namespace Raytracer
} // End namespace COPCORE_IMPL

#ifndef VECGEOM_CUDA_INTERFACE
void write_ppm(std::string filename, NavIndex_t *buffer, int px, int py)
{
  std::ofstream image(filename);

  image << "P3\n" << px << " " << py << "\n255\n";

  for (int j = py - 1; j >= 0; j--) {
    for (int i = 0; i < px; i++) {
      int idx = 4 * (j * px + i);
      image << (int)buffer[idx + 0] << " " << (int)buffer[idx + 1] << " " << (int)buffer[idx + 2] << "\n";
    }
  }
}
#endif
