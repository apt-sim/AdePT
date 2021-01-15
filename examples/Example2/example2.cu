// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example2.h"

#include "process.h"
#include "process_list.h"
#include "pair_production.h"
#include "energy_loss.h"

#include "track.h"
#include "trackBlock.h"

#include <AdePT/Atomic.h>
#include <AdePT/BlockData.h>
#include <AdePT/LoopNavigator.h>
#include <AdePT/MParray.h>

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Stopwatch.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

#include <iostream>
#include <iomanip>
#include <stdio.h>

// #include <cfloat> // for FLT_MIN

#include <CopCore/PhysicalConstants.h>

// some simple scoring
struct Scoring {
  adept::Atomic_t<int> hits;
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

#include "IterationStats.h"
#include "initTracks.h"

// Statistics for propagation chords
IterationStats      *chordIterStats     = nullptr;
__device__
IterationStats_impl *chordIterStats_dev = nullptr;

__host__ void ReportStatistics( IterationStats & iterStats );
// For temporary use in PrepareSta

__host__ void PrepareStatistics()
{
  chordIterStats = new IterationStats();
  // chordIterStats_dev = chordIterStats->GetDevicePtr();
  IterationStats_impl* ptrDev= chordIterStats->GetDevicePtr();
  assert( ptrDev != nullptr );
  cudaMemcpy(&chordIterStats_dev, &ptrDev,
             sizeof(IterationStats_impl*), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(chordIterStats_dev, &ptrDev,
                     sizeof(IterationStats_impl*));
  // Add assert(chordIterStats_dev != nullptr); in first use !?
}

__host__ void ReportStatistics( IterationStats & iterStats )
{
  // int  maxChordItersGPU;
  // cudaMemcpy(&maxChordItersGPU, maxItersDone_dev, sizeof(int), cudaMemcpyDeviceToHost);   
   std::cout << "-  Chord iterations: max (dev) = " << iterStats.GetMax() // GetMaxFromDevice()
             << "  total iters = " <<  iterStats.GetTotal() /*GetTotalFromDevice() */  ;
   std::cout << "  addr = " << iterStats.GetDevicePtr() << " "; 
   // printf(" -- host = %4d \n", GetMaxIterationsDone_host() );
   // std::cout << std::endl; // printf(" \n");   
}

#include "ConstBzFieldStepper.h"
#include "fieldPropagatorConstBz.h"

constexpr double kPush = 1.e-8;

constexpr bool  BfieldOn = true;
constexpr float BzFieldValue = 1.0e-30 * copcore::units::tesla;  // 0.1 * copcore::units::tesla;  // 30. * FLT_MIN; // 

// kernel select processes based on interaction lenght and put particles in the appropriate queues
__global__ void DefinePhysicalStepLength(adept::BlockData<track> *block, process_list *proclist,
                                         adept::MParray **queues, Scoring *scor)
{
  int n = block->GetNused() + block->GetNholes();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    // skip particles that are already dead
    track &mytrack = (*block)[i];

    // Experimental limit on number of steps - to avoid 'forever' particles
    constexpr uint16_t maxNumSteps = 250;  // Configurable in real simulation -- 1000 ?
    ++  mytrack.num_step;
    if( mytrack.num_step > maxNumSteps ) {
       mytrack.status = dead;
    }
    
    if (mytrack.status == dead) continue;

    // return value (if step limited by physics or geometry) not used for the moment
    // now, I know which process wins, so I add the particle to the appropriate queue
    float physics_step = proclist->GetPhysicsInteractionLength(i, block);
    float step= 0.0;

    if( ! BfieldOn ) {
       step= LoopNavigator::ComputeStepAndPropagatedState(mytrack.pos, mytrack.dir, physics_step,
                                                          mytrack.current_state, mytrack.next_state);
       mytrack.pos += (step + kPush) * mytrack.dir;
       if( step < physics_step ) mytrack.current_process = -1;
    }    
    else {
       fieldPropagatorConstBz fieldPropagator;
       step= fieldPropagator.ComputeStepAndPropagatedState(mytrack, physics_step, BzFieldValue);
       // updates state of 'mytrack'
       if( step < physics_step ) {
          // assert( ! mytrack.next_state.IsOnBoundary() && "Field Propagator returned step<phys -- yet boundary!" );
          mytrack.current_process = -2;
       }
    }
    mytrack.total_length += step;

    if ( mytrack.next_state.IsOnBoundary()) { // ( step == physics_step ) {  // JA-TEMP 
      // For now, just count that we hit something.
      scor->hits++;
      mytrack.SwapStates();      
    } else {
      // Enqueue track index for later processing.
      // assert(mytrack.current_process >= 0);
      if(mytrack.current_process >= 0)  // JA-TOFIX --- take this out !!! 
         queues[mytrack.current_process]->push_back(i);

      if(mytrack.current_state.IsOnBoundary()) {
         mytrack.SwapStates();
         // Just to clear the boundary flag from the current step ... 
      }
    }

    // mytrack.SwapStates();
  }
}




// kernel to call Along Step function for particles in the queues
__global__ void CallAlongStepProcesses(adept::BlockData<track> *block, process_list *proclist, adept::MParray **queues,
                                       Scoring *scor)
{
  int particle_index;

  // loop over all processes
  for (int process_id = 0; process_id < proclist->list_size; process_id++) {
    // for each process [process_id] consume the associated queue of particles
    int queue_size = queues[process_id]->size();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < queue_size; i += blockDim.x * gridDim.x) {
      // get particles index from the queue
      particle_index = (*(queues[process_id]))[i];
      // and call the process for it
      proclist->list[process_id]->GenerateInteraction(particle_index, block);

      // a simple version of scoring
      scor->totalEnergyLoss.fetch_add((*block)[particle_index].energy_loss);
      scor->secondaries.fetch_add((*block)[particle_index].number_of_secondaries);

      // if particles returns with 'dead' status, release the element from the block
      if ((*block)[particle_index].status == dead) block->ReleaseElement(particle_index);
    }
  }
}

// kernel function to initialize a track, most importantly the random state
__global__ void init_track(track *mytrack, const vecgeom::VPlacedVolume *world)
{
  /* we have to initialize the state */
  mytrack->rng_state.SetSeed(314159265);
  LoopNavigator::LocatePointIn(world, mytrack->pos, mytrack->current_state, true);
  mytrack->next_state= mytrack->current_state;
}

// kernel to create the processes and process list
__global__ void create_processes(process_list **proclist, process **processes)
{
  // instantiate the existing processes
  processes[0] = new energy_loss();
  processes[1] = new pair_production();

  // add them to process_list (process manager)
  *proclist = new process_list(processes, 2);
}

//
void example2(const vecgeom::cxx::VPlacedVolume *world)
{
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  cudaManager.LoadGeometry(world);
  cudaManager.Synchronize();

  const vecgeom::cuda::VPlacedVolume *gpu_world = cudaManager.world_gpu();

  // call the kernel to create the processes to be run on the device
  process_list **proclist_dev;
  process **processes;
  COPCORE_CUDA_CHECK(cudaMalloc((void **)&proclist_dev, sizeof(process_list *)));
  COPCORE_CUDA_CHECK(cudaMalloc((void **)&processes, 2 * sizeof(process *)));
  create_processes<<<1, 1>>>(proclist_dev, processes);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  process_list *proclist;
  COPCORE_CUDA_CHECK(cudaMemcpy(&proclist, proclist_dev, sizeof(process_list *), cudaMemcpyDeviceToHost));
  COPCORE_CUDA_CHECK(cudaFree(proclist_dev));

  PrepareStatistics();
  
  // Capacity of the different containers
  constexpr int capacity = 4 * 65536; // 1 << 20;

  std::cout << "INFO: capacity of containers (incl. BlockData<track>) set at "
            << capacity << std::endl;
  
  // setting the number of existing processes
  constexpr int numberOfProcesses = 2;
  char *buffer1[numberOfProcesses];

  // reserving queues for each of the processes
  adept::MParray **queues = nullptr;
  COPCORE_CUDA_CHECK(cudaMallocManaged(&queues, numberOfProcesses * sizeof(adept::MParray *)));
  size_t buffersize = adept::MParray::SizeOfInstance(capacity);

  for (int i = 0; i < numberOfProcesses; i++) {
    buffer1[i] = nullptr;
    COPCORE_CUDA_CHECK(cudaMallocManaged(&buffer1[i], buffersize));
    queues[i] = adept::MParray::MakeInstanceAt(capacity, buffer1[i]);
  }

  // Allocate the content of Scoring in a buffer
  char *buffer_scor = nullptr;
  COPCORE_CUDA_CHECK(cudaMallocManaged(&buffer_scor, sizeof(Scoring)));
  Scoring *scor = Scoring::MakeInstanceAt(buffer_scor);
  // Initialize scoring
  scor->hits            = 0;
  scor->secondaries     = 0;
  scor->totalEnergyLoss = 0;

    // Allocate a block of tracks with capacity larger than the total number of spawned threads
  size_t blocksize = adept::BlockData<track>::SizeOfInstance(capacity);
  char *buffer2    = nullptr;
  COPCORE_CUDA_CHECK(cudaMallocManaged(&buffer2, blocksize));
  auto block = adept::BlockData<track>::MakeInstanceAt(capacity, buffer2);

  // initializing one track in the block
  auto track1        = block->NextElement();
  track1->energy      = 100.0f;
  track1->energy_loss = 0.0f;
  //  track->index = 1; // this is not use for the moment, but it should be a unique track index
  track1->pos = {0, 0, 0};
  track1->dir = {1.0, 0, 0}; // {1.0/sqrt(2.), 0, 1.0/sqrt(2.)};
  track1->pdg= 11;  // e-
  track1->index= 1;
  track1->status= alive;  
  track1->interaction_length= 20.0;  
  init_track<<<1, 1>>>(track1, gpu_world);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());  

  // initializing second track in the block
  auto track2         = block->NextElement();
  track2->energy      = 30.0f;
  track2->energy_loss = 0.0f;
  //  track2->index = 2; // this is not use for the moment, but it should be a unique track index
  track2->pos = {0, 0.05, 0};
  track2->dir = {1.0, 0.0, 0.0}; // {1.0/sqrt(2.), 0, 1.0/sqrt(2.)};
  track2->pdg= -11;  // e+
  track2->index= 2;
  track2->status= alive;    
  track2->interaction_length= 10.0;
  init_track<<<1, 1>>>(track2, gpu_world);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  std::cout << "INFO: running with field " << ( BfieldOn ?  "ON" : "OFF" ) ;
  if( BfieldOn ) std::cout << " field value: Bz = " << BzFieldValue / copcore::units::tesla << " T ";
  std::cout << std::endl;

  // simple version of scoring
  float *energy_deposition = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc((void **)&energy_deposition, sizeof(float)));

  constexpr dim3 nthreads(32);
  constexpr dim3 maxBlocks(10);

  dim3 numBlocks;

  vecgeom::Stopwatch timer;
  timer.Start();

  int iterNo=0;
  int maxPrint = 32;

  bool verbose= false;
  std::cout << " Track1 with nav index: " << std::endl;
  track1->print(1, verbose );
  
  std::cout << " Track2 with nav index: " << std::endl;
  track2->print(2, verbose);
  
  std::cout << " Tracks at simulation start " << std::endl;
  printTracks(block, verbose, maxPrint);    
  
  while (block->GetNused() > 0 && iterNo < 1000 ) {
     
    numBlocks.x = (block->GetNused() + block->GetNholes() + nthreads.x - 1) / nthreads.x;
    numBlocks.x = std::min(numBlocks.x, maxBlocks.x);

    // call the kernel to do check the step lenght and select process
    DefinePhysicalStepLength<<<numBlocks, nthreads>>>(block, proclist, queues, scor);

    // call the kernel for Along Step Processes
    CallAlongStepProcesses<<<numBlocks, nthreads>>>(block, proclist, queues, scor);

    cudaDeviceSynchronize();      // Sync to print ...     
    std::cout << " Tracks after CallsAlongStepProccesses " << std::endl;
    printTracks(block, true, maxPrint);

    COPCORE_CUDA_CHECK(cudaDeviceSynchronize());    
    // clear all the queues before next step
    for (int i = 0; i < numberOfProcesses; i++)
      queues[i]->clear();
    COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "iter " << std::setw(4) << iterNo << " -- tracks in flight: " << std::setw(5) << block->GetNused() << " energy deposition: " << std::setw(8)
              << scor->totalEnergyLoss.load() << " number of secondaries: " << std::setw(5) << scor->secondaries.load()
              << " number of hits: " << std::setw(4) << scor->hits.load();
    ReportStatistics(*chordIterStats); // Chord statistics
    std::cout << std::endl;
    
    iterNo++;
  }

  auto time_cpu = timer.Stop();
  std::cout << "Run time: " << time_cpu << "\n";
}
