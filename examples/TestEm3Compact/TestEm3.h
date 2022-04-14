// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTEM3_H
#define TESTEM3_H

#include <VecGeom/base/Config.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif

// Data structures for scoring. The accessors must make sure to use atomic operations if needed.
struct GlobalScoring {
  double energyDeposit;
  // Not int to avoid overflows for more than 100,000 events; unsigned long long
  // is the only other data type available for atomicAdd().
  unsigned long long chargedSteps;
  unsigned long long neutralSteps;
  unsigned long long hits;
  unsigned long long numGammas;
  unsigned long long numElectrons;
  unsigned long long numPositrons;
  // Not used on the device, filled in by the host.
  unsigned long long numKilled;
};

struct ScoringPerVolume {
  double *energyDeposit;
  double *chargedTrackLength;
};

// Interface between C++ and CUDA.
void TestEm3(const vecgeom::cxx::VPlacedVolume *world, int numParticles, double energy, int batch, double startX,
             const int *MCIndex, ScoringPerVolume *scoringPerVolume, int numVolumes, GlobalScoring *globalScoring);

void InitBVH();

#endif
