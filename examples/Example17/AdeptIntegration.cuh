// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_INTEGRATION_CUH
#define ADEPT_INTEGRATION_CUH

#include "AdeptIntegration.h"

#include "Track.cuh"
#include <AdePT/TrackManager.cuh>

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmRandomEngine.hh>

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

// A bundle of track managers for the three particle types.
struct Secondaries {
  adept::TrackManager<Track> *electrons;
  adept::TrackManager<Track> *positrons;
  adept::TrackManager<Track> *gammas;
};

struct LeakedTracks {
  MParrayTracks *leakedElectrons;
  MParrayTracks *leakedPositrons;
  MParrayTracks *leakedGammas;
};

struct ParticleType {
  adept::TrackManager<Track> *trackmgr;
  MParrayTracks *leakedTracks;
  cudaStream_t stream;
  cudaEvent_t event;

  enum {
    Electron = 0,
    Positron = 1,
    Gamma    = 2,

    NumParticleTypes,
  };
};

// Track managers for the three particle types.
struct AllTrackManagers {
  adept::TrackManager<Track> *trackmgr[ParticleType::NumParticleTypes];
  MParrayTracks *leakedTracks[ParticleType::NumParticleTypes];
};

// A data structure to transfer statistics after each iteration.
struct Stats {
  adept::TrackManager<Track>::Stats mgr_stats[ParticleType::NumParticleTypes];
  int leakedTracks[ParticleType::NumParticleTypes];
};

struct GPUstate {
  using TrackData = adeptint::TrackData;

  ParticleType particles[ParticleType::NumParticleTypes];
  AllTrackManagers allmgr_h;          ///< Host pointers for track managers
  AllTrackManagers allmgr_d;          ///< Device pointers for track managers
  // Create a stream to synchronize kernels of all particle types.
  cudaStream_t stream;                ///< all-particle sync stream
  TrackData *toDevice_dev{nullptr};   ///< toDevice buffer of tracks
  TrackData *fromDevice_dev{nullptr}; ///< fromDevice buffer of tracks
  Stats *stats_dev{nullptr};          ///< statistics object pointer on device
  Stats *stats{nullptr};              ///< statistics object pointer on host
};

// Constant data structures from G4HepEm accessed by the kernels.
// (defined in TestEm3.cu)
extern __constant__ __device__ struct G4HepEmParameters g4HepEmPars;
extern __constant__ __device__ struct G4HepEmData g4HepEmData;

// Pointer for array of volume auxiliary data on device
extern __constant__ __device__ adeptint::VolAuxData *gVolAuxData;

// constexpr float BzFieldValue = 0.1 * copcore::units::tesla;
extern __constant__ __device__ double BzFieldValue;
constexpr double kPush = 1.e-8 * copcore::units::cm;

#endif
