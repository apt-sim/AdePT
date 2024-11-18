// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRANSPORT_STRUCT_CUH
#define ADEPT_TRANSPORT_STRUCT_CUH

#include <AdePT/core/CommonStruct.h>
#include <AdePT/core/HostScoringStruct.cuh>
#include <AdePT/magneticfield/GeneralMagneticField.h>

#include "Track.cuh"
#include <AdePT/base/TrackManager.cuh>

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>

#ifdef USE_SPLIT_KERNELS
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmGammaTrack.hh>
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

#ifdef USE_SPLIT_KERNELS
struct HepEmBuffers {
  G4HepEmElectronTrack *electronsHepEm;
  G4HepEmElectronTrack *positronsHepEm;
  G4HepEmGammaTrack *gammasHepEm;
};
#endif

// A data structure to transfer statistics after each iteration.
struct Stats {
  adept::TrackManager<Track>::Stats mgr_stats[ParticleType::NumParticleTypes];
  AdeptScoring::Stats scoring_stats;
  int leakedTracks[ParticleType::NumParticleTypes];
};

struct GPUstate {
  using TrackData = adeptint::TrackData;

  ParticleType particles[ParticleType::NumParticleTypes];
  AllTrackManagers allmgr_h; ///< Host pointers for track managers
  AllTrackManagers allmgr_d; ///< Device pointers for track managers
#ifdef USE_SPLIT_KERNELS
  HepEmBuffers hepEMBuffers_d;
#endif
  // Create a stream to synchronize kernels of all particle types.
  cudaStream_t stream;                ///< all-particle sync stream
  TrackData *toDevice_dev{nullptr};   ///< toDevice buffer of tracks
  TrackData *fromDevice_dev{nullptr}; ///< fromDevice buffer of tracks
  Stats *stats_dev{nullptr};          ///< statistics object pointer on device
  Stats *stats{nullptr};              ///< statistics object pointer on host
};

namespace adept_impl {
constexpr double kPush = 1.e-8 * copcore::units::cm;
extern __constant__ struct G4HepEmParameters g4HepEmPars;
extern __constant__ struct G4HepEmData g4HepEmData;

extern __constant__ __device__ adeptint::VolAuxData *gVolAuxData;
extern __constant__ __device__ double BzFieldValue;
extern __constant__ __device__ bool ApplyCuts;
#ifdef ADEPT_USE_EXT_BFIELD
__constant__ __device__ typename cuda_field_t::view_t *MagneticFieldView = nullptr;
#endif 
} // namespace adept_impl
#endif
