// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TRACKH
#define TRACKH

#include <CopCore/Ranluxpp.h>
#include <CopCore/PhysicalConstants.h>

#include <VecGeom/base/Vector3D.h>

#include <VecGeom/navigation/NavStateIndex.h>

#include <cfloat> // for FLT_MAX

enum TrackStatus { alive, dead };

struct track {
  RanluxppDouble rng_state;
  unsigned int index{0};
  unsigned int mother_index{0};
  unsigned int eventId{0};

  double energy{10};
  double energy_loss{0}; // current step's eloss -- used by primitive version of scoring
  double numIALeft[3];
  double interaction_length{DBL_MAX};
  double total_length{0.0}; // length since track start

  vecgeom::Vector3D<double> pos;
  vecgeom::Vector3D<double> dir;
  vecgeom::NavStateIndex current_state;
  vecgeom::NavStateIndex next_state;

  TrackStatus status{alive};

  int number_of_secondaries{0}; // total since track start -- used as counter for RNG

  uint16_t num_step{0}; // number of Steps -- for killing after MaX

  char current_process{0}; // -1 for Navigation / -2 for Field Propagation
  char pdg{0};             // Large enough for e/g. Then uint16_t for p/n

  __device__ __host__ double uniform() { return rng_state.Rndm(); }

  __device__ __host__ int charge() const // charge for e- / e+ / gamma only
  {
    int chrg = (pdg == pdgPositron ? 1 : 0) + (pdg == pdgElectron ? -1 : 0);
    return chrg;
  }

  __device__ __host__ double mass() const // Rest mass for e- / e+ / gamma only
  {
    return (pdg == pdgGamma) ? 0.0 : copcore::units::kElectronMassC2;
  }

  __host__ __device__ void print(int id = -1, bool verbose = false) const;

  __device__ __host__ void SwapStates()
  {
    auto state          = this->current_state;
    this->current_state = this->next_state;
    this->next_state    = state;
  }

  static constexpr char pdgElectron = 11;
  static constexpr char pdgPositron = -11;
  static constexpr char pdgGamma    = 22;
};

void track::print(int extId, bool verbose) const
{
  using copcore::units::MeV;
  using copcore::units::mm;
  static const char *particleName[3] = {"e-", "g", "e+"};
  printf(" %2s / id= %6d  m= %6d  ev=%3d  step# %4d  Pos[mm]: %11.6f, %11.6f, %11.6f  intl[mm]= %8.6g  len[mm]= %8.6f "
         "kE[MeV]= %12.10f  loss[MeV]= %10.6f  "
         "Dir: %11.6f, %11.6f, %11.6f %8s  proc= %1d",
         particleName[charge() + 1], index, mother_index, eventId, (int)num_step, pos[0] / mm, pos[1] / mm, pos[2] / mm,
         interaction_length / mm, total_length / mm, energy / MeV, energy_loss / MeV, dir[0], dir[1], dir[2],
         (status == alive ? "Alive" : "Dead "), (int)current_process);

  if (verbose) {
#ifdef COPCORE_DEVICE_COMPILATION
    auto currentLevel = current_state.GetLevel();
    auto currentIndex = current_state.GetNavIndex(currentLevel);
    auto nextLevel    = next_state.GetLevel();
    auto nextIndex    = next_state.GetNavIndex(nextLevel);
    printf("  current: (lv= %3d  ind= %8u  bnd= %1d)  next: (lv= %3d  ind= %8u  bnd= %1d)", currentLevel, currentIndex,
           (int)current_state.IsOnBoundary(), nextLevel, nextIndex, (int)next_state.IsOnBoundary());
#endif
  }
  printf("\n");
}

#endif
