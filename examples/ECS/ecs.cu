#include "ecs.h"
#include "Kernels.h"
#include "mergeAndSortBlocks.cuh"
#include "CudaHelpers.h"

#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <typeinfo>
#include <chrono>
#include <memory>
#include <iostream>
#include <iomanip>


/// Generic launcher for use with e.g. lambdas.
/// It passes to the function:
/// - task index
/// - block index
/// - track index
/// - the number of the track in each task
template<typename Slab, typename Func_t>
__global__ void launcher(Slab* slab, Func_t func, unsigned int loadFactor = 1)
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

        func(slab->tracks[taskIdx][bIdx][tIdx], taskIdx, bIdx, tIdx, globalTrackIdx_inTask);
      }
    }
  }
}


/// A timing helper that invokes a kernel and synchronises the device afterwards.
void time(std::string what, std::string after, std::function<void()> func) {
  std::cout << std::setw(30) << std::left << what << std::flush;
  const auto start = std::chrono::high_resolution_clock::now();

  func();
  checkCuda( cudaDeviceSynchronize() );

  const auto stop = std::chrono::high_resolution_clock::now();

  std::cout << "\t" << std::right << std::fixed << std::setprecision(3) << std::setw(12)
      << std::chrono::duration<double, std::milli>(stop - start).count() << " ms."
      << after << std::flush;
}





template<typename Slab>
__global__
void checkGlobalOccupancy(Slab* slab, unsigned int* storeOccupancy, unsigned int* referenceOccupancy = nullptr) {
  unsigned int occup = 0;
  for (auto& task : slab->tracks) {
    for (auto& block : task) {
      occup += occupancy<Slab::tracks_per_block, 1>(&block);
    }
  }

  if (referenceOccupancy) {
    assert(*referenceOccupancy == occup);
    if (*referenceOccupancy != occup) {
      printf("Error: Occupancy is wrong: last=%d now=%d\n", *referenceOccupancy, occup);
    }
  }

  *storeOccupancy = occup;
}



int main() {
  constexpr unsigned int loadMultiplier = 100;
#define MEASURE_COMPACTING_TIME false
#ifdef NDEBUG
  constexpr unsigned int nBlock = SlabSoA::blocks_per_task;
  constexpr unsigned int nThread = SlabSoA::tracks_per_block;
#else
  constexpr unsigned int nBlock  =  2;
  constexpr unsigned int nThread = 64;
#endif

  // First initialisation, so it doesn't affect timings
  cudaDeviceSynchronize();

  std::unique_ptr<SlabSoA> slabECS;
  std::unique_ptr<SlabAoS> slabAoS;

  time("ECS construct slabs", "\n", [&](){
    slabECS = std::make_unique<SlabSoA>();
  });
  time("AoS construct slabs", "\n", [&](){
    slabAoS = std::make_unique<SlabAoS>();
  });


  std::unique_ptr<SlabSoA, CudaDeleter> GPUMemECS;
  std::unique_ptr<SlabSoA, CudaDeleter> GPUMemECS2;
  std::unique_ptr<SlabAoS, CudaDeleter> GPUMemAoS;
  std::unique_ptr<SlabAoS, CudaDeleter> GPUMemAoS2;

  time("cuda malloc+memcpy", "\n", [&](){

    GPUMemECS  = make_unique_cuda<SlabSoA>();
    GPUMemECS2 = make_unique_cuda<SlabSoA>();
    checkCuda( cudaMemcpy(GPUMemECS.get(), slabECS.get(), sizeof(SlabSoA), cudaMemcpyDefault) );

    GPUMemAoS  = make_unique_cuda<SlabAoS>();
    GPUMemAoS2 = make_unique_cuda<SlabAoS>();
    checkCuda( cudaMemcpy(GPUMemAoS.get(), slabAoS.get(), sizeof(SlabAoS), cudaMemcpyDefault) );
  });
  


  

  // Give all particles an id
  // ------------------------------------

  time("ECS enumerate_particles", "\t", [&](){
    host_enumerate_particles(slabECS.get());
  });
  time(" GPU", "\t", [&](){
    run_enumerate_particles<<<nBlock, nThread>>>(GPUMemECS.get(), 1);
  });
  checkEnumeration<<<1, 1>>>(GPUMemECS.get());
  checkCuda( cudaDeviceSynchronize() );

  for (auto nRuns : std::initializer_list<unsigned int>{1, loadMultiplier}) {
    time(" GPU (lambda, run " + std::to_string(nRuns) + "x)", "\t", [&](){
      launcher<<<nBlock, nThread>>>(GPUMemECS.get(), []__device__(decltype(GPUMemECS->tracks[0][0][0]) track,
                                                            unsigned int taskIdx, unsigned int bIdx, unsigned int tIdx, unsigned int globalTIdx){
          const unsigned int particleID = taskIdx * SlabSoA::blocks_per_task * SlabSoA::tracks_per_block
            + globalTIdx;
          enumerate_particles(track, particleID);
        },
        nRuns);
    });
  }
  std::cout << std::endl;
  checkEnumeration<<<1, 1>>>(GPUMemECS.get());
  checkCuda( cudaDeviceSynchronize() );


  time("AoS enumerate_particles", "\t", [&](){
    host_enumerate_particles(slabAoS.get());
  });
  time(" GPU", "\t", [&](){
    run_enumerate_particles<<<nBlock, nThread>>>(GPUMemAoS.get());
  });
  checkEnumeration<<<1, 1>>>(GPUMemAoS.get());
  checkCuda( cudaDeviceSynchronize() );

  for (auto nRuns : std::initializer_list<unsigned int>{1, loadMultiplier}) {
    time(" GPU (lambda, run " + std::to_string(nRuns) + "x)", "\t", [&](){
      launcher<<<nBlock, nThread>>>(GPUMemAoS.get(), []__device__(decltype(GPUMemAoS->tracks[0][0][0]) track,
                                                            unsigned int taskIdx, unsigned int bIdx, unsigned int tIdx, unsigned int globalTIdx){
          const unsigned int particleID = taskIdx * SlabSoA::blocks_per_task * SlabSoA::tracks_per_block
            + globalTIdx;
          enumerate_particles(track, particleID);
        },
        nRuns);
    });
  }
  std::cout << std::endl;

  checkEnumeration<<<1, 1>>>(GPUMemAoS.get());
  checkCuda( cudaDeviceSynchronize() );



  // Seed all rngs on GPU
  // --------------------

  time("ECS seed rng", "\n", [&](){
    launcher<<<nBlock, nThread>>>(GPUMemECS.get(), []__device__(decltype(GPUMemECS->tracks[0][0][0]) track,
                                                          unsigned int taskIdx, unsigned int bIdx, unsigned int tIdx, unsigned int globalTIdx){
      seed_rng(track);
    });
  });
  time("AoS seed rng", "\n", [&](){
    launcher<<<nBlock, nThread>>>(GPUMemAoS.get(), []__device__(decltype(GPUMemAoS->tracks[0][0][0]) track,
                                                          unsigned int taskIdx, unsigned int bIdx, unsigned int tIdx, unsigned int globalTIdx){
      seed_rng(track);
    });
  });



  // Initialise position, momentum and energy
  // ----------------------------------------

  time("ECS compute_energy", "\t", [&](){
    host_compute_energy(slabECS.get());
  });
  time(" GPU (run 1x, invoke as kernel)", "\t", [&](){
    run_compute_energy<<<nBlock, nThread>>>(GPUMemECS.get());
  });
  for (auto nRuns : std::initializer_list<unsigned int>{1, loadMultiplier}) {
    time(" GPU (run " + std::to_string(nRuns) + "x, invoke as lambda)", "\t", [&](){
    launcher<<<nBlock, nThread>>>(GPUMemECS.get(), []__device__(decltype(GPUMemECS->tracks[0][0][0]) track,
                                                          unsigned int taskIdx, unsigned int bIdx, unsigned int tIdx, unsigned int globalTIdx){
        init_pos_mom(track);
        compute_energy(track);
      },
      nRuns);
    });
  }
  std::cout << std::endl;

  time("AoS compute_energy", "\t", [&](){
    host_compute_energy(slabAoS.get());
  });
  time(" GPU (run 1x, invoke as kernel)", "\t", [&](){
    run_compute_energy<<<nBlock, nThread>>>(GPUMemAoS.get());
  });
  for (auto nRuns : std::initializer_list<unsigned int>{1, loadMultiplier}) {
    time(" GPU (run " + std::to_string(nRuns) + "x, invoke as lambda)", "\t", [&](){
    launcher<<<nBlock, nThread>>>(GPUMemAoS.get(), [=]__device__(decltype(GPUMemAoS->tracks[0][0][0]) track,
                                                          unsigned int taskIdx, unsigned int bIdx, unsigned int tIdx, unsigned int globalTIdx){
        init_pos_mom(track);
        compute_energy(track);
      },
      nRuns);
    });
  }
  std::cout << std::endl;

  // Print some particles
  // ------------------------
  std::cout << "Particles from ECS, CPU:\n";
  slabECS->tracks[0][0].dump(0);
  slabECS->tracks[0][0].dump(1);
  slabECS->tracks[0][1].dump(2);
  slabECS->tracks[1][1].dump(3);
  
  std::cout << "Particles from AoS, CPU:\n";
  slabAoS->tracks[0][0*SlabSoA::tracks_per_block + 0].dump(0);
  slabAoS->tracks[0][0*SlabSoA::tracks_per_block + 1].dump(0);
  slabAoS->tracks[0][1*SlabSoA::tracks_per_block + 2].dump(0);
  slabAoS->tracks[1][1*SlabSoA::tracks_per_block + 3].dump(0);

  time("Memcpy back", "\n", [&](){
    checkCuda( cudaMemcpy(slabECS.get(), GPUMemECS.get(), sizeof(SlabSoA), cudaMemcpyDefault) );
    checkCuda( cudaMemcpy(slabAoS.get(), GPUMemAoS.get(), sizeof(SlabAoS), cudaMemcpyDefault) );
  });
  
  
  std::cout << "Particles from ECS, GPU:\n";
  slabECS->tracks[0][0].dump(0);
  slabECS->tracks[0][0].dump(1);
  slabECS->tracks[0][1].dump(2);
  slabECS->tracks[1][1].dump(3);
  
  std::cout << "Particles from AoS, GPU:\n";
  slabAoS->tracks[0][0*SlabSoA::tracks_per_block + 0].dump(0);
  slabAoS->tracks[0][0*SlabSoA::tracks_per_block + 1].dump(0);
  slabAoS->tracks[0][1*SlabSoA::tracks_per_block + 2].dump(0);
  slabAoS->tracks[1][1*SlabSoA::tracks_per_block + 3].dump(0);


  // Save current state, so we can run compactification multiple times
  time("Memcpy GPU to GPU", "\n", [&](){
    checkCuda( cudaMemcpy(GPUMemECS2.get(), GPUMemECS.get(), sizeof(SlabSoA), cudaMemcpyDefault) );
    checkCuda( cudaMemcpy(GPUMemAoS2.get(), GPUMemAoS.get(), sizeof(SlabAoS), cudaMemcpyDefault) );
  });



  // Advance particle by momentum vector with random magnitude
  // ---------------------------------------------------------
  // As a second step, advance the particles and kill them randomly

  constexpr float survivalProbability = 0.97f;
  unsigned int* occup;
  cudaMallocManaged(&occup, sizeof(unsigned int));

  time("\nECS advance_by_random_distance", "\t", [&](){
    run_advance_by_random_distance(slabECS.get());
  });
  for (auto nRuns : std::initializer_list<unsigned int>{1, loadMultiplier}) {
    time(" GPU (run " + std::to_string(nRuns) + "x)", "\t", [&](){
      run_advance_by_random_distance_and_kill<<<nBlock, nThread>>>(GPUMemECS.get(), nRuns);
    });
  }
  time(" GPU (run " + std::to_string(loadMultiplier) + "x, create holes)", "\n", [&](){
    run_advance_by_random_distance_and_kill<<<nBlock, nThread>>>(GPUMemECS.get(), loadMultiplier, survivalProbability);
  });


  time("AoS advance_by_random_distance", "\t", [&](){
    run_advance_by_random_distance(slabAoS.get());
  });
  for (auto nRuns : std::initializer_list<unsigned int>{1, loadMultiplier}) {
    time(" GPU (run " + std::to_string(nRuns) + "x)", "\t", [&](){
      run_advance_by_random_distance_and_kill<<<nBlock, nThread>>>(GPUMemAoS.get(), nRuns);
    });
  }
  time(" GPU (run " + std::to_string(loadMultiplier) + "x, create holes)", "\n", [&](){
    run_advance_by_random_distance_and_kill<<<nBlock, nThread>>>(GPUMemAoS.get(), loadMultiplier, survivalProbability);
  });


  
  checkCuda( cudaMemcpy(GPUMemECS.get(), GPUMemECS2.get(), sizeof(SlabSoA), cudaMemcpyDefault) );
  time(" GPU (run separate launches " + std::to_string(loadMultiplier) + "x, create holes, compact)", "\n", [&](){
    for (unsigned int run = 0; run < loadMultiplier; ++run) {
      launcher<<<nBlock, nThread>>>(GPUMemECS.get(), []__device__(decltype(GPUMemECS->tracks[0][0][0]) track,
                                                                  unsigned int taskIdx, unsigned int bIdx, unsigned int tIdx, unsigned int globalTIdx) {
        advance_by_random_distance(track);
        kill_random_particles(track, survivalProbability);
      });
      if (run % 5 == 0) {
        if constexpr (MEASURE_COMPACTING_TIME) {
          cudaDeviceSynchronize();
          checkGlobalOccupancy<<<1, 1>>>(GPUMemECS.get(), occup);
          cudaDeviceSynchronize();
          time("ECS Compactification run", "", [&](){
            run_merge_blocks<<<nBlock, nThread>>>(GPUMemECS.get());
          });
          checkGlobalOccupancy<<<1, 1>>>(GPUMemECS.get(), occup, occup);
          cudaDeviceSynchronize();
          std::cout << " Occupancy= " << *occup << std::endl; 
        } else {
          run_merge_blocks<<<nBlock, nThread>>>(GPUMemECS.get());
        }
      }
    }
  });
  checkCuda( cudaMemcpy(GPUMemAoS.get(), GPUMemAoS2.get(), sizeof(SlabAoS), cudaMemcpyDefault) );
  time(" GPU (run separate launches " + std::to_string(loadMultiplier) + "x, create holes, compact)", "\n", [&](){
    for (unsigned int run = 0; run < loadMultiplier; ++run) {
      launcher<<<nBlock, nThread>>>(GPUMemAoS.get(), []__device__(decltype(GPUMemAoS->tracks[0][0][0]) track,
                                                                  unsigned int taskIdx, unsigned int bIdx, unsigned int tIdx, unsigned int globalTIdx) {
        advance_by_random_distance(track);
        kill_random_particles(track, survivalProbability);
      });
      if (run % 5 == 0) {
        if constexpr (MEASURE_COMPACTING_TIME) {
          cudaDeviceSynchronize();
          checkGlobalOccupancy<<<1, 1>>>(GPUMemAoS.get(), occup);
          cudaDeviceSynchronize();
          time("AoS Compactification run", "", [&](){
            run_merge_blocks<<<nBlock, nThread>>>(GPUMemAoS.get());
          });
          checkGlobalOccupancy<<<1, 1>>>(GPUMemAoS.get(), occup, occup);
          cudaDeviceSynchronize();
          std::cout << " Occupancy= " << *occup << std::endl; 
        } else {
          run_merge_blocks<<<nBlock, nThread>>>(GPUMemAoS.get());
        }
      }
    }
  });



  
  time("Memcpy back", "\n", [&](){
    checkCuda( cudaMemcpy(slabECS.get(), GPUMemECS.get(), sizeof(SlabSoA), cudaMemcpyDefault) );
    checkCuda( cudaMemcpy(slabAoS.get(), GPUMemAoS.get(), sizeof(SlabAoS), cudaMemcpyDefault) );
  });
  
  
  std::cout << "Particles from ECS, GPU:\n";
  slabECS->tracks[0][0].dump(0);
  slabECS->tracks[0][0].dump(1);
  slabECS->tracks[0][1].dump(2);
  slabECS->tracks[1][1].dump(3);
  
  std::cout << "Particles from AoS, GPU:\n";
  slabAoS->tracks[0][0*SlabSoA::tracks_per_block + 0].dump(0);
  slabAoS->tracks[0][0*SlabSoA::tracks_per_block + 1].dump(0);
  slabAoS->tracks[0][1*SlabSoA::tracks_per_block + 2].dump(0);
  slabAoS->tracks[1][1*SlabSoA::tracks_per_block + 3].dump(0);
  
}
