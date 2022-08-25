// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE9_CUH
#define EXAMPLE9_CUH

#include <AdePT/MParray.h>
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

    // A secondary inherits the position of its parent; the caller is responsible
    // to update the directions.
    this->pos      = parent.pos;
    this->navState = parent.navState;
  }
};

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

// A data structure for some global scoring. The accessors must make sure to use
// atomic operations if needed.
struct GlobalScoring {
  int hits;
  int secondaries;
  double energyDeposit;
};

// A data structure to manage slots in the track storage.
class SlotManager {
  adept::Atomic_t<int> fNextSlot;
  const int fMaxSlot;

public:
  __host__ __device__ SlotManager(int maxSlot) : fMaxSlot(maxSlot) { fNextSlot = 0; }

  __host__ __device__ int NextSlot()
  {
    int next = fNextSlot.fetch_add(1);
    if (next >= fMaxSlot) return -1;
    return next;
  }
};

// A bundle of pointers to generate particles of an implicit type.
class ParticleGenerator {
  Track *fTracks;
  SlotManager *fSlotManager;
  adept::MParray *fActiveQueue;

public:
  __host__ __device__ ParticleGenerator(Track *tracks, SlotManager *slotManager, adept::MParray *activeQueue)
    : fTracks(tracks), fSlotManager(slotManager), fActiveQueue(activeQueue) {}

  __host__ __device__ Track &NextTrack()
  {
    int slot = fSlotManager->NextSlot();
    if (slot == -1) {
      COPCORE_EXCEPTION("No slot available in ParticleGenerator::NextTrack");
    }
    fActiveQueue->push_back(slot);
    return fTracks[slot];
  }
};

// A bundle of generators for the three particle types.
struct Secondaries {
  ParticleGenerator electrons;
  ParticleGenerator positrons;
  ParticleGenerator gammas;
};


// Kernels in different TUs.
__global__ void RelocateToNextVolume(Track *allTracks, const adept::MParray *relocateQueue);

__global__ void TransportElectrons(
    Track *electrons, const adept::MParray *active, Secondaries secondaries, adept::MParray *activeQueue,
    adept::MParray *relocateQueue, GlobalScoring *scoring);
__global__ void TransportPositrons(
    Track *positrons, const adept::MParray *active, Secondaries secondaries, adept::MParray *activeQueue,
    adept::MParray *relocateQueue, GlobalScoring *scoring);

__global__ void TransportGammas(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, adept::MParray *relocateQueue, GlobalScoring *scoring);

// Constant data structures from G4HepEm accessed by the kernels.
// (defined in example9.cu)
extern __constant__ __device__ struct G4HepEmParameters g4HepEmPars;
extern __constant__ __device__ struct G4HepEmData g4HepEmData;

constexpr vecgeom::Precision BzFieldValue = 0.1 * copcore::units::tesla;

#endif
