// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// This is the main include file to give us all of the alpaka classes, functions etc.
#include <alpaka/alpaka.hpp>

#include <iostream>

#include "particleProcessor.h"

// This processes one event on the device. A list of particles is passed through
// For each particle we call process(*partList, iTh)
// process will then work the same was as in the C++ version
struct processEvent {
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc, particle *partList, int *steps) const
  {

    using namespace alpaka;
    // iTh is the thread number we use this throughout
    uint32_t iTh = getIdx<Grid, Threads>(acc)[0];
    // dummy sensitive, not used yet
    sensitive SD(400., 500.);

    // Create an alpaka random generator using a seed of 1984
    auto generator = rand::engine::createDefault(acc, 1984, iTh);

    // Here we split threads that start immediatly, from threads that will have to wait until the shower is under way
    particleProcessor particleProcessor;
    int ns     = particleProcessor.processParticle(acc, partList, iTh, generator, SD);
    steps[iTh] = ns;
  }
};

int main()
{
  using namespace alpaka; // alpaka functionality all lives in this namespace

  using Dim = DimInt<1>;
  using Idx = uint32_t;

  // Define the alpaka accelerator to be Nvidia GPU
  using Acc = AccGpuCudaRt<Dim, Idx>;

  // Get the first device available of type GPU (i.e should be our sole GPU)/device
  auto const device = getDevByIdx<Acc>(0u);
  // Create a device for host for memory allocation, using the first CPU available
  auto devHost = getDevByIdx<DevCpu>(0u);

  // Allocate memory on both the host and device for part objects and numbers of steps
  uint32_t NPART = 200;
  Vec<Dim, Idx> bufferExtent{NPART};
  auto device_event = allocBuf<particle, Idx>(device, bufferExtent);
  auto host_event   = allocBuf<particle, Idx>(devHost, bufferExtent);
  auto device_steps = allocBuf<int, Idx>(device, bufferExtent);
  auto host_steps   = allocBuf<int, Idx>(devHost, bufferExtent);

  int size = NPART * sizeof(particle); // memory size to use
  std::cout << "memory allocated " << size << std::endl;

  // All photons have momentum 20 GeV along z
  vec3 mom = vec3(0., 0., 20000.);

  // create NPART particles:
  for (int ii = 0; ii < NPART; ii++) {
    vec3 pos                                = vec3(0., 0., (float)ii);
    getPtrNative(host_event)[ii]            = particle(pos, mom, 22);
  }

  // Copy particles to the GPU
  auto queue = Queue<Acc, Blocking>{device};
  memcpy(queue, device_event, host_event, bufferExtent);

  // Check that this makes sense on CPU
  for (int ii = 0; ii < NPART; ii++) {
    std::cout << __CUDACC__ << "  " << ii << ", " << getPtrNative(host_event)[ii].getPos().z() << std::endl;
  }

  // Launch one thread per particle for NPART particles
  uint32_t blocksPerGrid     = NPART;
  uint32_t threadsPerBlock   = 1;
  uint32_t elementsPerThread = 1;

  auto workDiv = WorkDivMembers<Dim, Idx>{blocksPerGrid, threadsPerBlock, elementsPerThread};

  // Create a task for processEvent, that we can run and then run it via a queue
  processEvent processEvent;

  auto taskRunProcessEvent =
      createTaskKernel<Acc>(workDiv, processEvent, getPtrNative(device_event), getPtrNative(device_steps));

  enqueue(queue, taskRunProcessEvent);

  memcpy(queue, host_steps, device_steps, bufferExtent);
  std::cout << "Thread, nsteps" << std::endl;
  for (int ii = 0; ii < NPART; ii++)
    std::cout << ii << ", " << getPtrNative(host_steps)[ii] << std::endl;

  return 0;
}
