// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTEM3_CUH
#define TESTEM3_CUH

#include "TestEm3.h"

#include <AdePT/MParray.h>
#include <AdePT/TrackManager.cuh>
#include <CopCore/SystemOfUnits.h>
#include <CopCore/Ranluxpp.h>

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmRandomEngine.hh>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavStateIndex.h>

// A data structure to represent a particle track. The particle type is implicit
// by the queue and not stored in memory.
struct Track {
  using Precision = vecgeom::Precision;
  RanluxppDouble rngState;
  double energy;
  double numIALeft[3];
  double initialRange;
  double dynamicRangeFactor;
  double tlimitMin;

  vecgeom::Vector3D<Precision> pos;
  vecgeom::Vector3D<Precision> dir;
  vecgeom::NavStateIndex navState;

  __host__ __device__ double Uniform() { return rngState.Rndm(); }

  __host__ __device__ void InitAsSecondary(const Track &parent)
  {
    // The caller is responsible to branch a new RNG state and to set the energy.
    this->numIALeft[0] = -1.0;
    this->numIALeft[1] = -1.0;
    this->numIALeft[2] = -1.0;

    this->initialRange       = -1.0;
    this->dynamicRangeFactor = -1.0;
    this->tlimitMin          = -1.0;

    // A secondary inherits the position of its parent; the caller is responsible
    // to update the directions.
    this->pos      = parent.pos;
    this->navState = parent.navState;
  }
};

#ifdef __CUDA_ARCH__
// Define inline implementations of the RNG methods for the device.
// (nvcc ignores the __device__ attribute in definitions, so this is only to
// communicate the intent.)
inline __device__ double G4HepEmRandomEngine::flat()
{
  return ((RanluxppDouble *)fObject)->Rndm();
}

inline __device__ void G4HepEmRandomEngine::flatArray(const int size, double *vect)
{
  for (int i = 0; i < size; i++) {
    vect[i] = ((RanluxppDouble *)fObject)->Rndm();
  }
}
#endif

// A bundle of generators for the three particle types.
struct Secondaries {
  adept::TrackManager<Track> *electrons;
  adept::TrackManager<Track> *positrons;
  adept::TrackManager<Track> *gammas;
};

// Kernels in different TUs.
__global__ void TransportElectrons(adept::TrackManager<Track> *electrons, Secondaries secondaries,
                                   GlobalScoring *globalScoring, ScoringPerVolume *scoringPerVolume);
__global__ void TransportPositrons(adept::TrackManager<Track> *positrons, Secondaries secondaries,
                                   GlobalScoring *globalScoring, ScoringPerVolume *scoringPerVolume);

__global__ void TransportGammas(adept::TrackManager<Track> *gammas, Secondaries secondaries,
                                GlobalScoring *globalScoring, ScoringPerVolume *scoringPerVolume);

// Constant data structures from G4HepEm accessed by the kernels.
// (defined in TestEm3.cu)
extern __constant__ __device__ struct G4HepEmParameters g4HepEmPars;
extern __constant__ __device__ struct G4HepEmData g4HepEmData;

extern __constant__ __device__ int *MCIndex;

// constexpr vecgeom::Precision BzFieldValue = 1 * copcore::units::tesla;
constexpr vecgeom::Precision BzFieldValue = 0;

#endif
