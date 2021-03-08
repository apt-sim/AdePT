// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE8_H
#define EXAMPLE8_H

#include <VecGeom/base/Config.h>
#include <VecGeom/volumes/PlacedVolume.h>

#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif

void example8(const vecgeom::cxx::VPlacedVolume *world, int particles, double energy);

#endif
