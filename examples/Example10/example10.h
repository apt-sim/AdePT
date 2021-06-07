// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE10_H
#define EXAMPLE10_H

#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif

// Interface between C++ and CUDA.
void example10(const vecgeom::cxx::VPlacedVolume *world, int numParticles, double energy);

#endif
