// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TRACKH
#define TRACKH

#include <curand_kernel.h>

#include <cfloat> // for FLT_MAX

enum TrackStatus { alive, dead };

struct track {
  curandState_t curand_state;
  int index{0};
  int pdg{0};
  double energy{10};
  double pos[3]{0};
  double dir[3]{1};
  int mother_index{0};
  TrackStatus status{alive};
  int current_process{0};
  float interaction_length{FLT_MAX};
  float energy_loss{0};         // primitive version of scoring
  int number_of_secondaries{0}; // primitive version of scoring

  __device__ float uniform() { return curand_uniform(&curand_state); }
};

#endif
