// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// This is the main include file to give us all of the alpaka classes, functions etc.
#include <alpaka/alpaka.hpp>

#include "particle.h"
#include "particleProcessor.h"

#include <iostream>

using namespace alpaka;

struct testEnergyLoss {
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc, particle *partList, float *eLoss) const
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
  ALPAKA_FN_ACC void operator()(Acc const &acc, particle *partList, particle *newPart) const
  {

    // iTh is the thread number we use this throughout
    uint32_t iTh = idx::getIdx<Grid, Threads>(acc)[0];

    // Create an alpaka random generator using a seed of 1984
    auto generator = rand::generator::createDefault(acc, 1984, iTh);

    // Create a particleProcessor class, from which we can test its member functions.
    particleProcessor particleProcessor;
    particleProcessor.lowerThreshold();

    // Now we call splitParticle. This could return either a valid pointer, if we
    // split the particle or a null pointer if we don't.
    particle *partAfterSplit = (particleProcessor.splitParticle(acc, partList[iTh], generator));
    if (partAfterSplit) newPart[iTh] = *partAfterSplit;
  }
};

struct testStep {
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc, particle *partList, particle *newPart) const
  {

    // iTh is the thread number we use this throughout
    uint32_t iTh = idx::getIdx<Grid, Threads>(acc)[0];

    // Create an alpaka random generator using a seed of 1984
    auto generator = rand::generator::createDefault(acc, 1984, iTh);

    // Create a particleProcessor class, from which we can test its member functions.
    particleProcessor particleProcessor;
    particleProcessor.lowerThreshold();

    // dummy sensitive, not used yet
    sensitive SD(400., 500.);

    // Now we call step. This could return either a valid pointer, if we
    // split the particle via the call internally to splitParticle or a null pointer if we don't.
    particle *partAfterStep = (particleProcessor.step(acc, partList[iTh], generator, SD));
    if (partAfterStep) newPart[iTh] = *partAfterStep;
  }
};

struct testProcessParticle {
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc, particle *partList, int *steps) const
  {

    using namespace alpaka;
    // iTh is the thread number we use this throughout
    uint32_t iTh = idx::getIdx<Grid, Threads>(acc)[0];
    // dummy sensitive, not used yet
    sensitive SD(400., 500.);

    // Create an alpaka random generator using a seed of 1984
    auto generator = rand::generator::createDefault(acc, 1984, iTh);

    // Here we split threads that start immediatly, from threads that will have to wait until the shower is under way
    particleProcessor particleProcessor;
    int ns     = particleProcessor.processParticle(acc, partList, iTh, generator, SD);
    steps[iTh] = ns;
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
  auto h_event = mem::buf::alloc<particle, Idx>(devHost, bufferExtent);

  // All photons have momentum 20 GeV along z
  float initialMomentum = 20000.;
  vec3 mom              = vec3(0., 0., initialMomentum);

  // create NPART particles:
  for (int ii = 0; ii < NPART; ii++) {
    vec3 pos                             = vec3(0., 0., (float)ii);
    mem::view::getPtrNative(h_event)[ii] = particle(pos, mom, 22);   
  }

  // Copy particles to the GPU
  auto queue = queue::Queue<Acc, queue::Blocking>{device};

  // Launch one thread per particle for NPART particles
  uint32_t blocksPerGrid     = NPART;
  uint32_t threadsPerBlock   = 1;
  uint32_t elementsPerThread = 1;

  auto workDiv = workdiv::WorkDivMembers<Dim, Idx>{blocksPerGrid, threadsPerBlock, elementsPerThread};

  // Create data for testing particleProcessor.energyLoss
  auto d_ELoss = mem::buf::alloc<float, Idx>(device, bufferExtent);
  auto h_ELoss = mem::buf::alloc<float, Idx>(devHost, bufferExtent);
  auto d_eventELoss = mem::buf::alloc<particle, Idx>(device, bufferExtent);
  mem::view::copy(queue, d_eventELoss, h_event, bufferExtent);

  // Create a task for testEnergyLoss, that we can run and then run it via a queue

  testEnergyLoss testEnergyLoss;
  auto taskRunTestEnergyLoss = kernel::createTaskKernel<Acc>(workDiv, testEnergyLoss, mem::view::getPtrNative(d_eventELoss),
                                                             mem::view::getPtrNative(d_ELoss));

  queue::enqueue(queue, taskRunTestEnergyLoss);
  // copy the calculated energy losses back to the host.
  mem::view::copy(queue, h_ELoss, d_ELoss, bufferExtent);

  // Then we test the output of energyLoss for each thread.
  // By construction the energy loss should not be more than 0.2 MeV.
  bool testOK = true;
  for (unsigned int counter = 0; counter < NPART; counter++) {

    float ELoss = mem::view::getPtrNative(h_ELoss)[counter];
    if (ELoss > 0.2) testOK = false;
  }

  // Print the test result.
  std::cout << "Status of testEnergyLoss is " << testOK << std::endl;

  // Create data for testing particleProcessor.splitParticle
  auto d_newPartSplit = mem::buf::alloc<particle, Idx>(device, bufferExtent);
  auto h_newPartSplit = mem::buf::alloc<particle, Idx>(devHost, bufferExtent);
  auto d_eventSplit = mem::buf::alloc<particle, Idx>(device, bufferExtent);
  mem::view::copy(queue, d_eventSplit, h_event, bufferExtent);

  // Create a task for testSplitParticle, that we can run and then run it via a queue
  testSplitParticle testSplitParticle;
  auto taskRunTestSplitParticle = kernel::createTaskKernel<Acc>(
      workDiv, testSplitParticle, mem::view::getPtrNative(d_eventSplit), mem::view::getPtrNative(d_newPartSplit));

  queue::enqueue(queue, taskRunTestSplitParticle);
  // copy the part objects back to the host
  mem::view::copy(queue, h_newPartSplit, d_newPartSplit, bufferExtent);

  // Then we test the output part for each thread.
  // By construction the particle momentum should not be more than the initial momentum.
  testOK = true;
  for (unsigned int counter = 0; counter < NPART; counter++) {
    particle newPart = mem::view::getPtrNative(h_newPartSplit)[counter];
    if ((newPart.getMom().length()) > initialMomentum) testOK = false;
  }

  std::cout << "Status of testSplitParticle is " << testOK << std::endl;

  // Create data for testing particleProcessor.step
  auto d_newPartStep = mem::buf::alloc<particle, Idx>(device, bufferExtent);
  auto h_newPartStep = mem::buf::alloc<particle, Idx>(devHost, bufferExtent);
  auto d_eventStep = mem::buf::alloc<particle, Idx>(device, bufferExtent);
  mem::view::copy(queue, d_eventStep, h_event, bufferExtent);

  // Create a task for testStep, that we can run and then run it via a queue
  testStep testStep;
  auto taskRunTestStep = kernel::createTaskKernel<Acc>(
      workDiv, testStep, mem::view::getPtrNative(d_eventStep), mem::view::getPtrNative(d_newPartStep));

  queue::enqueue(queue, taskRunTestStep);
  // copy the new part objects back to the host
  mem::view::copy(queue, h_newPartStep, d_newPartStep, bufferExtent);

  // Then we test the output part for each thread.
  // By construction the z-position should be the thread number + 0.1
  // As is the case in the test for SplitParticle by construction the particle momentum
  // should not be more than the initial momentum (because step calls splitParticle internally)
  testOK = true;
  for (unsigned int counter = 0; counter < NPART; counter++) {
    particle newPart = mem::view::getPtrNative(h_newPartStep)[counter];
    // skip cases where no splitting occurred, because then there is no particle to check.
    if (!newPart.getMom().length() > 0) continue;
    if ((newPart.getMom().length()) > initialMomentum) {
      std::cout << " Error in testStep: New momentum and initial momentum are " << newPart.getMom().length() << " and "
                << initialMomentum << std::endl;
      testOK = false;
    }
    float expectedZPosition = counter + 0.1;
    if (fabs(newPart.getPos().z() - expectedZPosition) > 0.0001) {
      std::cout << " Error in testStep: New position and expected Z position are " << newPart.getPos().z() << " and "
                << expectedZPosition << std::endl;
      testOK = false;
    }
  }

  std::cout << "Status of testStep is " << testOK << std::endl;

  // Create data for testing particleProcessor.processParticle
  auto d_stepsProcess = mem::buf::alloc<int, Idx>(device, bufferExtent);
  auto h_stepsProcess = mem::buf::alloc<int, Idx>(devHost, bufferExtent);
  auto d_eventProcess = mem::buf::alloc<particle, Idx>(device, bufferExtent);
  mem::view::copy(queue, d_eventProcess, h_event, bufferExtent);  

  //Create a task for testProcessParticle, that we can run and then run it via a queue
  testProcessParticle testProcessParticle;
  auto taskRunTestProcessParticle = kernel::createTaskKernel<Acc>(
    workDiv, testProcessParticle, mem::view::getPtrNative(d_eventProcess), mem::view::getPtrNative(d_stepsProcess));

  queue::enqueue(queue, taskRunTestProcessParticle);
  //Copy steps back to the host
  mem::view::copy(queue, h_stepsProcess, d_stepsProcess, bufferExtent);

  //Then we test the output - the number of steps should be at least 1, but there is no maximum value.
  //Thus we simply test that the nSteps incremented beyond the initial zero value.
  testOK = true;
  for (unsigned int counter = 0; counter < NPART; counter++) {
    if ( !(mem::view::getPtrNative(h_stepsProcess)[counter] > 0) ) testOK = false;
  }

  std::cout << "Status of testProcessParticle is " << testOK << std::endl;

  return 0;
}