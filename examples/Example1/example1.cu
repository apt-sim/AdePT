// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "process.h"
#include "process_list.h"
#include "pair_production.h"
#include "energy_loss.h"

#include "track.h"

#include <AdePT/BlockData.h>
#include <AdePT/MParray.h>

#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <iomanip>
#include <stdio.h>

// some simple scoring
struct Scoring {
  adept::Atomic_t<int> secondaries;
  adept::Atomic_t<float> totalEnergyLoss;

  __host__ __device__
  Scoring() {}

  __host__ __device__
  static Scoring *MakeInstanceAt(void *addr)
  {
    Scoring *obj = new (addr) Scoring();
    return obj;
  }
};

// kernel select processes based on interaction lenght and put particles in the appropriate queues
__global__ void DefinePhysicalStepLength(adept::BlockData<track> *block, process_list **proclist,
                                         adept::MParray **queues)
{
  int n = block->GetNused() + block->GetNholes();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {

    // skip particles that are already dead
    if ((*block)[i].status == dead) continue;

    // return value (if step limited by physics or geometry) not used for the moment
    // now, I know which process wins, so I add the particle to the appropriate queue
    (*proclist)->GetPhysicsInteractionLength(i, block);
    queues[(*block)[i].current_process]->push_back(i);
  }
}

// kernel to call Along Step function for particles in the queues
__global__ void CallAlongStepProcesses(adept::BlockData<track> *block, process_list **proclist, adept::MParray **queues,
                                       Scoring *scor)
{
  int particle_index;

  // loop over all processes
  for (int process_id=0 ; process_id < (*proclist)->list_size; process_id++)
    {
      // for each process [process_id] consume the associated queue of particles
      int queue_size = queues[process_id]->size();

      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < queue_size; i += blockDim.x * gridDim.x)
        {
          // get particles index from the queue
          particle_index = (*(queues[process_id]))[i];
          // and call the process for it
          ((*proclist)->list)[process_id]->GenerateInteraction(particle_index, block);

          // a simple version of scoring
          scor->totalEnergyLoss.fetch_add((*block)[particle_index].energy_loss);
          scor->secondaries.fetch_add((*block)[particle_index].number_of_secondaries);

          // if particles returns with 'dead' status, release the element from the block
          if ((*block)[particle_index].status == dead) block->ReleaseElement(particle_index);
        }
    }
}

// kernel function to initialize a track, most importantly the random state
__global__ void init_track(track *mytrack)
{
  /* we have to initialize the state */
  curand_init(0, 0, 0, &mytrack->curand_state);
}

// kernel to create the processes and process list
__global__ void create_processes(process_list **proclist, process **processes)
{
  // instantiate the existing processes
  *(processes) = new energy_loss();
  *(processes+1) = new pair_production();

  // add them to process_list (process manager)
  *proclist = new process_list(processes, 2);
}

//
int main()
{
  // call the kernel to create the processes to be run on the device
  process_list **proclist;
  process **processes;
  cudaMalloc((void**)&proclist, sizeof(process_list*));
  cudaMalloc((void**)&processes, 2*sizeof(process*));
  create_processes<<<1,1>>>(proclist, processes);
  cudaDeviceSynchronize();

  // Capacity of the different containers
  constexpr int capacity = 1 << 20;

  // setting the number of existing processes
  constexpr int numberOfProcesses = 2;
  char *buffer1[numberOfProcesses];

  // reserving queues for each of the processes
  adept::MParray **queues = nullptr;
  cudaMallocManaged(&queues, numberOfProcesses * sizeof(adept::MParray *));
  size_t buffersize = adept::MParray::SizeOfInstance(capacity);

  for (int i = 0; i < numberOfProcesses; i++) {
    buffer1[i] = nullptr;
    cudaMallocManaged(&buffer1[i], buffersize);
    queues[i] = adept::MParray::MakeInstanceAt(capacity, buffer1[i]);
  }

  // Allocate the content of Scoring in a buffer
  char *buffer_scor = nullptr;
  cudaMallocManaged(&buffer_scor, sizeof(Scoring));
  Scoring *scor = Scoring::MakeInstanceAt(buffer_scor);
  // Initialize scoring
  scor->secondaries     = 0;
  scor->totalEnergyLoss = 0;

  // Allocate a block of tracks with capacity larger than the total number of spawned threads
  size_t blocksize = adept::BlockData<track>::SizeOfInstance(capacity);
  char *buffer2    = nullptr;
  cudaMallocManaged(&buffer2, blocksize);
  auto block = adept::BlockData<track>::MakeInstanceAt(capacity, buffer2);

  // initializing one track in the block
  auto track    = block->NextElement();
  track->energy = 100.0f;
  track->energy_loss = 0.0f;
  //  track->index = 1; // this is not use for the moment, but it should be a unique track index
  init_track<<<1, 1>>>(track);
  cudaDeviceSynchronize();

  // initializing second track in the block
  auto track2    = block->NextElement();
  track2->energy = 30.0f;
  track2->energy_loss = 0.0f;
  //  track2->index = 2; // this is not use for the moment, but it should be a unique track index
  init_track<<<1, 1>>>(track2);
  cudaDeviceSynchronize();

  // simple version of scoring
  float* energy_deposition = nullptr;
  cudaMalloc((void **)&energy_deposition, sizeof(float));

  constexpr dim3 nthreads(32);
  constexpr dim3 maxBlocks(10);
  dim3 numBlocks;

  while (block->GetNused()>0)
  {
    numBlocks.x = (block->GetNused() + block->GetNholes() + nthreads.x - 1) / nthreads.x;
    numBlocks.x = std::min(numBlocks.x, maxBlocks.x);

    // call the kernel to do check the step lenght and select process
    DefinePhysicalStepLength<<<numBlocks, nthreads>>>(block, proclist, queues);

    // call the kernel for Along Step Processes
    CallAlongStepProcesses<<<numBlocks, nthreads>>>(block, proclist, queues, scor);

    cudaDeviceSynchronize();
    // clear all the queues before next step
    for (int i = 0; i < numberOfProcesses; i++) queues[i]->clear();
    cudaDeviceSynchronize();

    std::cout << "Number of tracks in flight: " << std::setw(8) << block->GetNused() << " total energy depostion: " << std::setw(10) << scor->totalEnergyLoss.load()
    << " total number of secondaries: " << scor->secondaries.load() << std::endl;
  }
}
