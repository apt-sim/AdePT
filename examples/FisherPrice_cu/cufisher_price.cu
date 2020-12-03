// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

#include <AdePT/BlockData.h>

struct MyTrack {
  int index{0};
  int pdg{0};
  double energy{0};
  double pos[3]{0};
  double dir[3]{0};
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

// kernel function that does energy loss or pair production
__global__ void process(adept::BlockData<MyTrack> *block, Scoring *scor, curandState_t *states, int max_index)
{
  int particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  // check if you are not outside the used block
  if (particle_index > max_index) return;

  double energy = (*block)[particle_index].energy;

  // check if the particle is still alive (E>0)
  if (energy == 0) return;

  // generate random number
  float r = curand_uniform(states);

  // call the 'process'
  if (r < 0.5f) {
    // energy loss
    float eloss = 0.2f * energy;
    scor->totalEnergyLoss.fetch_add(eloss < 0.001f ? energy : eloss);
    energy = eloss < 0.001f ? 0.0f : (energy - eloss);
    (*block)[particle_index].energy = energy;

    // if particle dies (E=0) release the slot
    if (energy < 0.001f) block->ReleaseElement(particle_index);
  } else {
    // pair production
    float eloss = 0.5f * energy;
    (*block)[particle_index].energy = energy - eloss;

    // here I need to create a new particle
    auto secondary_track = block->NextElement();
    assert(secondary_track != nullptr && "No slot available for secondary track");
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

  // Track capacity of the block
  constexpr int capacity = 1 << 20;

  // Allocate the content of Scoring in a buffer
  char *buffer1 = nullptr;
  cudaMallocManaged(&buffer1, sizeof(Scoring));
  Scoring *scor = Scoring::MakeInstanceAt(buffer1);
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

  cudaDeviceSynchronize();
  //
  constexpr dim3 nthreads(32);
  dim3 numBlocks;

  while (block->GetNused()) {
    int max_index = block->GetNused() + block->GetNholes();
    numBlocks.x = (max_index + nthreads.x - 1) / nthreads.x;
    // call the kernels
    process<<<numBlocks, nthreads>>>(block, scor, state, max_index);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    std::cout << "Total energy loss " << scor->totalEnergyLoss.load() << " number of secondaries "
              << scor->secondaries.load() << " blocks used " << block->GetNused() << std::endl;
  }
}
