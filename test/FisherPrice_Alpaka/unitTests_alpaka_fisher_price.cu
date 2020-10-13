// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// This is the main include file to give us all of the alpaka classes, functions etc.
#include <alpaka/alpaka.hpp>

#include "part.h"
#include "particleProcessor.h"

#include <iostream>

using namespace alpaka;

struct testEnergyLoss {
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc, part *partList, float *eLoss) const
  {

    // iTh is the thread number we use this throughout
    uint32_t iTh = idx::getIdx<Grid, Threads>(acc)[0];

    // Create an alpaka random generator using a seed of 1984
    auto generator = rand::generator::createDefault(acc, 1984, iTh);

    // Create a particleProcessor class, from which we can test its member functions.
    particleProcessor particleProcessor;

    eLoss[iTh] = particleProcessor.energyLoss(acc, partList[iTh], generator);
  }
};

struct testSplitParticle {
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc, part *partList, part *newPart) const
  {

    // iTh is the thread number we use this throughout
    uint32_t iTh = idx::getIdx<Grid, Threads>(acc)[0];

    // Create an alpaka random generator using a seed of 1984
    auto generator = rand::generator::createDefault(acc, 1984, iTh);

    // Create a particleProcessor class, from which we can test its member functions.
    particleProcessor particleProcessor;

    part *partAfterSplit = (particleProcessor.splitParticle(acc, partList[iTh], generator));
    if (partAfterSplit) newPart[iTh] = *partAfterSplit;
  }
};

int main()
{
  using Dim = dim::DimInt<1>;
  using Idx = uint32_t;

  // Define the alpaka accelerator to be Nvidia GPU
  using Acc = acc::AccGpuCudaRt<Dim, Idx>;

  // Get the first device available of type GPU (i.e should be our sole GPU)/device
  auto const device = pltf::getDevByIdx<Acc>(0u);
  // Create a device for host for memory allocation, using the first CPU available
  auto devHost = pltf::getDevByIdx<dev::DevCpu>(0u);

  // Allocate memory on both the host and device for part objects and numbers of steps
  uint32_t NPART = 200;
  vec::Vec<Dim, Idx> bufferExtent{NPART};
  auto d_event = mem::buf::alloc<part, Idx>(device, bufferExtent);
  auto h_event = mem::buf::alloc<part, Idx>(devHost, bufferExtent);

  // All photons have momentum 20 GeV along z
  float initialMomentum = 20000.;
  vec3 mom              = vec3(0., 0., initialMomentum);

  // create NPART particles:
  for (int ii = 0; ii < NPART; ii++) {
    vec3 pos                             = vec3(0., 0., (float)ii);
    mem::view::getPtrNative(h_event)[ii] = part(pos, mom, 22);
  }

  // Copy particles to the GPU
  auto queue = queue::Queue<Acc, queue::Blocking>{device};
  mem::view::copy(queue, d_event, h_event, bufferExtent);

  // Launch one thread per particle for NPART particles
  uint32_t blocksPerGrid     = NPART;
  uint32_t threadsPerBlock   = 1;
  uint32_t elementsPerThread = 1;

  auto workDiv = workdiv::WorkDivMembers<Dim, Idx>{blocksPerGrid, threadsPerBlock, elementsPerThread};

  // Create data for testing particleProcessor.energyLoss
  auto d_ELoss = mem::buf::alloc<float, Idx>(device, bufferExtent);
  auto h_ELoss = mem::buf::alloc<float, Idx>(devHost, bufferExtent);

  // Create a task for testEnergyLoss, that we can run and then run it via a queue
  testEnergyLoss testEnergyLoss;
  auto taskRunTestEnergyLoss = kernel::createTaskKernel<Acc>(workDiv, testEnergyLoss, mem::view::getPtrNative(d_event),
                                                             mem::view::getPtrNative(d_ELoss));

  queue::enqueue(queue, taskRunTestEnergyLoss);
  //copy the calculated energy losses back to the host.
  mem::view::copy(queue, h_ELoss, d_ELoss, bufferExtent);

  //Then we test the output of energyLoss for each thread.
  //By construction the energy loss should not be more than 0.2 MeV.
  bool testOK = true;
  for (unsigned int counter = 0; counter < NPART; counter++) {

    float ELoss = mem::view::getPtrNative(h_ELoss)[counter];
    if (ELoss > 0.2) testOK = false;
  }

  //Print the test result.
  std::cout << "Status of testEnergyLoss is " << testOK << std::endl;

  // Create data for testing particleProcessor.splitParticle
  auto d_newPart = mem::buf::alloc<part, Idx>(device, bufferExtent);
  auto h_newPart = mem::buf::alloc<part, Idx>(devHost, bufferExtent);

  //Create a task for testSplitParticle, that we can run and then run it via a queue
  testSplitParticle testSplitParticle;
  auto taskRunTestSplitParticle = kernel::createTaskKernel<Acc>(
      workDiv, testSplitParticle, mem::view::getPtrNative(d_event), mem::view::getPtrNative(d_newPart));

  queue::enqueue(queue, taskRunTestSplitParticle);
  //copy the part objects back to the host
  mem::view::copy(queue, h_newPart, d_newPart, bufferExtent);

  //Then we test the output part for each thread.
  //By construction the particle momentum should not be more than the initial momentum.
  testOK = true;
  for (unsigned int counter = 0; counter < NPART; counter++) {
    part newPart = mem::view::getPtrNative(h_newPart)[counter];
    if ( (newPart.getMom().length()) > initialMomentum) testOK = false;
  }

  std::cout << "Status of testSplitParticle is " << testOK << std::endl;

  return 0;
}