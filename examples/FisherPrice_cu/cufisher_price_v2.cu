// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

#include <AdePT/BlockData.h>

using Queue_t = adept::mpmc_bounded_queue<int>;

struct MyTrack {
  int index{0};
  int pdg{0};
  double energy{10};
  double pos[3]{0};
  double dir[3]{1};
  bool flag1;
  bool flag2;
};

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

// kernel function that does transportation
__global__ void transport(int n, adept::BlockData<MyTrack> *block, curandState_t *states, Queue_t *queues)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    // transport particles
    for (int xyz = 0; xyz < 3; xyz++) {
      (*block)[i].pos[xyz] = (*block)[i].pos[xyz] + (*block)[i].energy * (*block)[i].dir[xyz];
    }
  }
}

// kernel function that assigns next process to the particle
__global__ void select_process(adept::BlockData<MyTrack> *block, Scoring *scor, curandState_t *states,
                               Queue_t *queues[])
{
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  // check if you are not outside the used block
  if (particle_index > block->GetNused() + block->GetNholes()) return;

  // check if the particle is still alive (E>0)
  if ((*block)[particle_index].energy == 0) return;

  // generate random number
  float r = curand_uniform(states);

  if (r > 0.5f) {
    queues[0]->enqueue(particle_index);
  } else {
    queues[1]->enqueue(particle_index);
  }
}

// kernel function that does energy loss
__global__ void process_eloss(int n, adept::BlockData<MyTrack> *block, Scoring *scor, curandState_t *states,
                              Queue_t *queue)
{
  int particle_index;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    if (!queue->dequeue(particle_index)) return;

    // check if the particle is still alive (E>0)
    if ((*block)[particle_index].energy == 0) return;

    // call the 'process'
    // energy loss
    float eloss = 0.2f * (*block)[particle_index].energy;
    scor->totalEnergyLoss.fetch_add(eloss < 0.001f ? (*block)[particle_index].energy : eloss);
    (*block)[particle_index].energy = (eloss < 0.001f ? 0.0f : ((*block)[particle_index].energy - eloss));

    // if particle dies (E=0) release the slot
    if ((*block)[particle_index].energy < 0.001f) block->ReleaseElement(particle_index);
  }
}

// kernel function that does pair production
__global__ void process_pairprod(int n, adept::BlockData<MyTrack> *block, Scoring *scor, curandState_t *states,
                                 Queue_t *queue)
{
  int particle_index;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    if (!queue->dequeue(particle_index)) return;

    // check if the particle is still alive (E>0)
    if ((*block)[particle_index].energy == 0) return;

    // pair production
    auto secondary_track = block->NextElement();
    assert(secondary_track != nullptr && "No slot available for secondary track");

    float eloss = 0.5f * (*block)[particle_index].energy;
    (*block)[particle_index].energy -= eloss;

    secondary_track->energy = eloss;

    // increase the counter of secondaries
    scor->secondaries.fetch_add(1);
  }
}

/* this GPU kernel function is used to initialize the random states */
__global__ void init(curandState_t *states)
{
  /* we have to initialize the state */
  curand_init(0, 0, 0, states);
}

//

int main()
{

  curandState_t *state;
  cudaMalloc((void **)&state, sizeof(curandState_t));
  init<<<1, 1>>>(state);
  cudaDeviceSynchronize();

  // Capacity of the different containers
  constexpr int capacity = 1 << 20;

  using Queue_t = adept::mpmc_bounded_queue<int>;

  constexpr int numberOfProcesses = 3;
  char *buffer[numberOfProcesses];

  Queue_t **queues = nullptr;
  cudaMallocManaged(&queues, numberOfProcesses * sizeof(Queue_t *));

  size_t buffersize = Queue_t::SizeOfInstance(capacity);

  for (int i = 0; i < numberOfProcesses; i++) {
    buffer[i] = nullptr;
    cudaMallocManaged(&buffer[i], buffersize);

    queues[i] = Queue_t::MakeInstanceAt(capacity, buffer[i]);
  }

  // Allocate the content of Scoring in a buffer
  char *buffer_scor = nullptr;
  cudaMallocManaged(&buffer_scor, sizeof(Scoring));
  Scoring *scor = Scoring::MakeInstanceAt(buffer_scor);
  // Initialize scoring
  scor->secondaries     = 0;
  scor->totalEnergyLoss = 0;

  // Allocate a block of tracks with capacity larger than the total number of spawned threads
  // Note that if we want to allocate several consecutive block in a buffer, we have to use
  // Block_t::SizeOfAlignAware rather than SizeOfInstance to get the space needed per block

  using Block_t    = adept::BlockData<MyTrack>;
  size_t blocksize = Block_t::SizeOfInstance(capacity);
  char *buffer2    = nullptr;
  cudaMallocManaged(&buffer2, blocksize);
  auto block = Block_t::MakeInstanceAt(capacity, buffer2);

  // initializing one track in the block
  auto track    = block->NextElement();
  track->energy = 100.0f;

  // initializing second track in the block
  auto track2    = block->NextElement();
  track2->energy = 30.0f;

  constexpr dim3 nthreads(32);
  constexpr dim3 maxBlocks(10);
  dim3 numBlocks, numBlocks_eloss, numBlocks_pairprod, numBlocks_transport;

  while (block->GetNused()) {

    numBlocks.x = (block->GetNused() + block->GetNholes() + nthreads.x - 1) / nthreads.x;

    // here I set the maximum number of blocks

    numBlocks_transport.x = std::min(numBlocks.x, maxBlocks.x);

    transport<<<numBlocks_transport, nthreads>>>(queues[2]->size(), block, state, queues[2]);

    // call the kernel to select the process
    select_process<<<numBlocks, nthreads>>>(block, scor, state, queues);

    cudaDeviceSynchronize();

    // call the process kernels

    numBlocks_eloss.x    = std::min((queues[0]->size() + nthreads.x - 1) / nthreads.x, maxBlocks.x);
    numBlocks_pairprod.x = std::min((queues[1]->size() + nthreads.x - 1) / nthreads.x, maxBlocks.x);

    process_eloss<<<numBlocks_eloss, nthreads>>>(queues[0]->size(), block, scor, state, queues[0]);

    process_pairprod<<<numBlocks_pairprod, nthreads>>>(queues[1]->size(), block, scor, state, queues[1]);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    std::cout << "Total energy loss " << scor->totalEnergyLoss.load() << " number of secondaries "
              << scor->secondaries.load() << " blocks used " << block->GetNused() << std::endl;
  }
}
