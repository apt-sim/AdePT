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
  int index{0};
  double energy{10};
  vecgeom::Vector3D<double> pos;
  vecgeom::Vector3D<double> dir;
  vecgeom::NavStateIndex current_state;
  vecgeom::NavStateIndex next_state;
  int mother_index{0};
  TrackStatus status{alive};

  float interaction_length{FLT_MAX};
  float energy_loss{0};         // primitive version of scoring
  int number_of_secondaries{0}; // primitive version of scoring

  short pdg{0};               // Enough for e/g, even p/n/pi/K
  short current_process{0};   // 
   
  __device__ double uniform() { return rng_state.Rndm(); }

  __device__ __host__ int    charge()
  { int chrg= (pdg== -11)-(pdg==11); return chrg; }

  __device__ __host__ double mass()  // Rest mass
  { return  (pdg== 22) ? 0 : copcore::units::kElectronMassC2; } 
   
  __device__ void SwapStates()
  {
    auto state          = this->current_state;
    this->current_state = this->next_state;
    this->next_state    = state;
  }
};

#endif
