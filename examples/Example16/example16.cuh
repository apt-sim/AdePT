// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE16_CUH
#define EXAMPLE16_CUH

#include "example16.h"

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

// Defined in example16.cu
extern __constant__ __device__ int Zero;

class RanluxppDoubleEngine : public G4HepEmRandomEngine {
  // Wrapper functions to call into RanluxppDouble.
  static __host__ __device__ __attribute__((noinline)) double FlatWrapper(void *object)
  {
    return ((RanluxppDouble *)object)->Rndm();
  }
  static __host__ __device__ __attribute__((noinline)) void FlatArrayWrapper(void *object, const int size, double *vect)
  {
    for (int i = 0; i < size; i++) {
      vect[i] = ((RanluxppDouble *)object)->Rndm();
    }
  }

public:
  __host__ __device__ RanluxppDoubleEngine(RanluxppDouble *engine)
      : G4HepEmRandomEngine(/*object=*/engine, &FlatWrapper, &FlatArrayWrapper)
  {
#ifdef __CUDA_ARCH__
    // This is a hack: The compiler cannot see that we're going to call the
    // functions through their pointers, so it underestimates the number of
    // required registers. By including calls to the (non-inlinable) functions
    // we force the compiler to account for the register usage, even if this
    // particular set of calls are not executed at runtime.
    if (Zero) {
      FlatWrapper(engine);
      FlatArrayWrapper(engine, 0, nullptr);
    }
#endif
  }
};

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

// constexpr vecgeom::Precision BzFieldValue = 0.1 * copcore::units::tesla;
constexpr vecgeom::Precision BzFieldValue = 0;

#endif
