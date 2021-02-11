// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE4_H
#define EXAMPLE4_H

#include <VecGeom/base/Config.h>
#include <VecGeom/volumes/PlacedVolume.h>

#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif

void example6(const vecgeom::cxx::VPlacedVolume *world);

#endif
