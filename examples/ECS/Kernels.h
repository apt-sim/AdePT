// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLES_ECS_KERNELS_H_
#define EXAMPLES_ECS_KERNELS_H_

#include "ecs.h"


template<typename TrackRef_t>
__host__ __device__ inline void enumerate_particles(const TrackRef_t& track, unsigned int number)
{
  track.id = number;
}

template<typename TrackBlock_t>
__host__ __device__ inline void enumerate_particles(TrackBlock_t& track, unsigned int tIdx, unsigned int number)
{
  track.id[tIdx] = number;
}

template<typename TrackRef_t>
__host__ __device__ inline void check_enumerate_particles(const TrackRef_t& track, unsigned int number)
{
  assert(track.id == number);
}

template<typename Slab>
void host_enumerate_particles(Slab* slab) {
  unsigned int number = 0;
  for (auto& task : slab->tracks)
    for (auto& block : task)      
      for (unsigned int i = 0; i < block.nSlot; ++i) {
        enumerate_particles(block[i], number++);
      }
}

template<typename Slab>
__global__ void run_enumerate_particles(Slab* slab, unsigned int loadFactor = 1)
{
  for (unsigned int extraRuns = 0; extraRuns < loadFactor; ++extraRuns) {
    for (unsigned int taskIdx = 0; taskIdx < slab->tasks_per_slab; ++taskIdx) {

      for (unsigned int globalTrackIdx_inTask = blockIdx.x * blockDim.x + threadIdx.x;
          globalTrackIdx_inTask < Slab::tracks_per_block * Slab::blocks_per_task;
          globalTrackIdx_inTask += gridDim.x * blockDim.x) {

        const auto tIdx = globalTrackIdx_inTask % Slab::tracks_per_block;
        const auto bIdx = globalTrackIdx_inTask / Slab::tracks_per_block;
        assert(tIdx < Slab::tracks_per_block);
        assert(bIdx < Slab::blocks_per_task);

        auto& block = slab->tracks[taskIdx][bIdx];
        const unsigned int particleID = taskIdx * slab->blocks_per_task * slab->tracks_per_block
            + globalTrackIdx_inTask;

        enumerate_particles(block, tIdx, particleID);
      }
    }
  }
}

template<typename Slab>
__global__

void checkEnumeration(Slab* slab) {
  unsigned int number = 0;
  for (auto& task : slab->tracks)
    for (auto& block : task)
      for (unsigned int i = 0; i < block.nSlot; ++i) {
        check_enumerate_particles(block[i], number++);
      }
}




template<typename TrackRef_t>
__host__ __device__ inline void init_pos_mom(const TrackRef_t& track)
{
  const float initValue = track.id % (SlabSoA::tracks_per_block * SlabSoA::blocks_per_task);
  track.x = initValue;
  track.y = initValue + 0.1;
  track.z = initValue + 0.2;

  track.vx = 1.;
  track.vy = 2.;
  track.vz = 3.;
}

template<typename Track_t>
__host__ __device__ inline void init_pos_mom(Track_t& track, unsigned int tIdx)
{
  assert(tIdx < track.nSlot);
  const float initValue = track.id[tIdx] % (SlabSoA::tracks_per_block * SlabSoA::blocks_per_task);
  track.x[tIdx] = initValue;
  track.y[tIdx] = initValue + 0.1;
  track.z[tIdx] = initValue + 0.2;

  track.vx[tIdx] = 1.;
  track.vy[tIdx] = 2.;
  track.vz[tIdx] = 3.;
}




template<typename TrackRef_t>
__host__ __device__ inline void compute_energy(const TrackRef_t& track)
{
  track.E = std::sqrt( track.vx*track.vx + track.vy*track.vy + track.vz*track.vz );
}

template<typename Track_t>
__host__ __device__ inline void compute_energy(Track_t& track, unsigned int tIdx)
{
  track.E[tIdx] = std::sqrt( track.vx[tIdx]*track.vx[tIdx] + track.vy[tIdx]*track.vy[tIdx] + track.vz[tIdx]*track.vz[tIdx] );
}

template<typename Slab>
void host_compute_energy(Slab* slab) {
  for (auto& task : slab->tracks)
    for (unsigned int blockIndex = 0; blockIndex < Slab::blocks_per_task; ++blockIndex) {
      auto& block = task[blockIndex];
      for (unsigned int trkIdx = 0; trkIdx < block.nSlot; ++trkIdx) {
        init_pos_mom(block[trkIdx]);
        compute_energy(block[trkIdx]);
      }
    }
}

template<typename Slab>
__global__ void run_compute_energy(Slab* slab, unsigned int loadFactor = 1)
{
  for (unsigned int extraRuns = 0; extraRuns < loadFactor; ++extraRuns) {
    for (unsigned int taskIdx = 0; taskIdx < slab->tasks_per_slab; ++taskIdx) {

      for (unsigned int globalTrackIdx_inTask = blockIdx.x * blockDim.x + threadIdx.x;
          globalTrackIdx_inTask < Slab::tracks_per_block * Slab::blocks_per_task;
          globalTrackIdx_inTask += gridDim.x * blockDim.x) {

        const auto tIdx = globalTrackIdx_inTask % Slab::tracks_per_block;
        const auto bIdx = globalTrackIdx_inTask / Slab::tracks_per_block;
        assert(tIdx < Slab::tracks_per_block);
        assert(bIdx < Slab::blocks_per_task);

        init_pos_mom(slab->tracks[taskIdx][bIdx], tIdx);
        compute_energy(slab->tracks[taskIdx][bIdx], tIdx);
      }
    }
  }
}



template<typename TrackRef_t>
__host__ __device__ inline void seed_rng(TrackRef_t track)
{
  track.rng_state.SetSeed(track.id);
}



template<typename TrackRef_t>
__host__ __device__ inline void advance_by_random_distance(TrackRef_t track) {
  if (track.id < 0)
    return;

  const auto rand = track.rng_state.Rndm() + track.id / 1.e9f;
  track.x += track.vx * rand;
  track.y += track.vy * rand;
  track.z += track.vz * rand;
}


template<typename TrackRef_t>
__host__ __device__ inline void kill_random_particles(TrackRef_t track, const float survivalProbability) {
  if (track.id >= 0 && track.rng_state.Rndm() > survivalProbability)
    track.id *= -1;
}


template<typename Slab>
void run_advance_by_random_distance(Slab* slab) {
  for (auto& task : slab->tracks)
    for (auto& block : task)
      for (unsigned int i = 0; i < block.nSlot; ++i) {
        advance_by_random_distance(block[i]);
      }
}

template<typename Slab>
__global__ void run_advance_by_random_distance_and_kill(Slab* slab, unsigned int loadFactor = 1, const float survivalProbability = 1.f)
{
  for (unsigned int extraRuns = 0; extraRuns < loadFactor; ++extraRuns) {
    for (unsigned int taskIdx = 0; taskIdx < slab->tasks_per_slab; ++taskIdx) {

      for (unsigned int globalTrackIdx_inTask = blockIdx.x * blockDim.x + threadIdx.x;
          globalTrackIdx_inTask < Slab::tracks_per_block * Slab::blocks_per_task;
          globalTrackIdx_inTask += gridDim.x * blockDim.x) {

        const auto tIdx = globalTrackIdx_inTask % Slab::tracks_per_block;
        const auto bIdx = globalTrackIdx_inTask / Slab::tracks_per_block;
        assert(globalTrackIdx_inTask < Slab::tracks_per_block * Slab::blocks_per_task);
        assert(tIdx < Slab::tracks_per_block);
        assert(bIdx < Slab::blocks_per_task);

        auto& block = slab->tracks[taskIdx][bIdx];

        advance_by_random_distance(block[tIdx]);

        if (survivalProbability < 1.f) {
          kill_random_particles(block[tIdx], survivalProbability);
        }
      }
    }
  }
}


#endif /* EXAMPLES_ECS_KERNELS_H_ */
