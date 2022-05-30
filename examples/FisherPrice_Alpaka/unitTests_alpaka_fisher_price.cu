// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// This is the main include file to give us all of the alpaka classes, functions etc.
#include <alpaka/alpaka.hpp>

#include "particle.h"
#include "particleProcessor.h"

#include <iostream>

using namespace alpaka;

/**
 * @brief kernel that tests @sa particleProcessor.energyLoss
 * The output is returned to the host, which checks that output.
 */
struct testEnergyLoss {
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc, particle *partList, float *eLoss) const
  {

    // iTh is the thread number we use this throughout
    uint32_t iTh = getIdx<Grid, Threads>(acc)[0];

    // Create an alpaka random generator using a seed of 1984
    auto generator = rand::engine::createDefault(acc, 1984, iTh);

    // Create a particleProcessor class, from which we can test its member functions.
    particleProcessor particleProcessor;

    eLoss[iTh] = particleProcessor.energyLoss(acc, partList[iTh], generator);
  }
};

/**
 * @brief kernel that tests @sa particleProcessor.splitParticle
 * The output is returned to the host, which checks that output.
 */
struct testSplitParticle {
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc, particle *partList, particle *newPart) const
  {

    // iTh is the thread number we use this throughout
    uint32_t iTh = getIdx<Grid, Threads>(acc)[0];

    // Create an alpaka random generator using a seed of 1984
    auto generator = rand::engine::createDefault(acc, 1984, iTh);

    // Create a particleProcessor class, from which we can test its member functions.
    particleProcessor particleProcessor;
    particleProcessor.lowerThreshold();

    // Now we call splitParticle. This could return either a valid pointer, if we
    // split the particle or a null pointer if we don't.
    particle *partAfterSplit = (particleProcessor.splitParticle(acc, partList[iTh], generator));
    if (partAfterSplit) newPart[iTh] = *partAfterSplit;
  }
};

/**
 * @brief kernel that tests @sa particleProcessor.step
 * The output is returned to the host, which checks that output.
 */
struct testStep {
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc, particle *partList, particle *newPart) const
  {

    // iTh is the thread number we use this throughout
    uint32_t iTh = getIdx<Grid, Threads>(acc)[0];

    // Create an alpaka random generator using a seed of 1984
    auto generator = rand::engine::createDefault(acc, 1984, iTh);

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

/**
 * @brief kernel that tests @sa particleProcessor.processParticle
 * The output is returned to the host, which checks that output.
 */
struct testProcessParticle {
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
  auto host_event = allocBuf<particle, Idx>(devHost, bufferExtent);

  // All photons have momentum 20 GeV along z
  float initialMomentum = 20000.;
  vec3 mom              = vec3(0., 0., initialMomentum);

  // create NPART particles:
  for (int ii = 0; ii < NPART; ii++) {
    vec3 pos                     = vec3(0., 0., (float)ii);
    getPtrNative(host_event)[ii] = particle(pos, mom, 22);
  }

  // Copy particles to the GPU
  auto queue = Queue<Acc, Blocking>{device};

  // Launch one thread per particle for NPART particles
  uint32_t blocksPerGrid     = NPART;
  uint32_t threadsPerBlock   = 1;
  uint32_t elementsPerThread = 1;

  auto workDiv = WorkDivMembers<Dim, Idx>{blocksPerGrid, threadsPerBlock, elementsPerThread};

  // Create data for testing particleProcessor.energyLoss
  auto device_ELoss      = allocBuf<float, Idx>(device, bufferExtent);
  auto host_ELoss        = allocBuf<float, Idx>(devHost, bufferExtent);
  auto device_eventELoss = allocBuf<particle, Idx>(device, bufferExtent);
  memcpy(queue, device_eventELoss, host_event, bufferExtent);

  // Create a task for testEnergyLoss, that we can run and then run it via a queue

  testEnergyLoss testEnergyLoss;
  auto taskRunTestEnergyLoss =
      createTaskKernel<Acc>(workDiv, testEnergyLoss, getPtrNative(device_eventELoss), getPtrNative(device_ELoss));

  enqueue(queue, taskRunTestEnergyLoss);
  // copy the calculated energy losses back to the host.
  memcpy(queue, host_ELoss, device_ELoss, bufferExtent);

  // Then we test the output of energyLoss for each thread.
  // By construction the energy loss should not be more than 0.2 MeV.
  bool testOK = true;
  for (unsigned int counter = 0; counter < NPART; counter++) {

    float ELoss = getPtrNative(host_ELoss)[counter];
    if (ELoss > 0.2) testOK = false;
  }

  // Print the test result.
  std::cout << "Status of testEnergyLoss is " << testOK << std::endl;

  // Create data for testing particleProcessor.splitParticle
  auto device_newPartSplit = allocBuf<particle, Idx>(device, bufferExtent);
  auto host_newPartSplit   = allocBuf<particle, Idx>(devHost, bufferExtent);
  auto device_eventSplit   = allocBuf<particle, Idx>(device, bufferExtent);
  memcpy(queue, device_eventSplit, host_event, bufferExtent);

  // Create a task for testSplitParticle, that we can run and then run it via a queue
  testSplitParticle testSplitParticle;
  auto taskRunTestSplitParticle = createTaskKernel<Acc>(workDiv, testSplitParticle, getPtrNative(device_eventSplit),
                                                        getPtrNative(device_newPartSplit));

  enqueue(queue, taskRunTestSplitParticle);
  // copy the part objects back to the host
  memcpy(queue, host_newPartSplit, device_newPartSplit, bufferExtent);

  // Then we test the output part for each thread.
  // By construction the particle momentum should not be more than the initial momentum.
  testOK = true;
  for (unsigned int counter = 0; counter < NPART; counter++) {
    particle newPart = getPtrNative(host_newPartSplit)[counter];
    if ((newPart.getMom().length()) > initialMomentum) testOK = false;
  }

  std::cout << "Status of testSplitParticle is " << testOK << std::endl;

  // Create data for testing particleProcessor.step
  auto device_newPartStep = allocBuf<particle, Idx>(device, bufferExtent);
  auto host_newPartStep   = allocBuf<particle, Idx>(devHost, bufferExtent);
  auto device_eventStep   = allocBuf<particle, Idx>(device, bufferExtent);
  memcpy(queue, device_eventStep, host_event, bufferExtent);

  // Create a task for testStep, that we can run and then run it via a queue
  testStep testStep;
  auto taskRunTestStep =
      createTaskKernel<Acc>(workDiv, testStep, getPtrNative(device_eventStep), getPtrNative(device_newPartStep));

  enqueue(queue, taskRunTestStep);
  // copy the new part objects back to the host
  memcpy(queue, host_newPartStep, device_newPartStep, bufferExtent);

  // Then we test the output part for each thread.
  // By construction the z-position should be the thread number + 0.1
  // As is the case in the test for SplitParticle by construction the particle momentum
  // should not be more than the initial momentum (because step calls splitParticle internally)
  testOK = true;
  for (unsigned int counter = 0; counter < NPART; counter++) {
    particle newPart = getPtrNative(host_newPartStep)[counter];
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
  auto device_stepsProcess = allocBuf<int, Idx>(device, bufferExtent);
  auto host_stepsProcess   = allocBuf<int, Idx>(devHost, bufferExtent);
  auto device_eventProcess = allocBuf<particle, Idx>(device, bufferExtent);
  memcpy(queue, device_eventProcess, host_event, bufferExtent);

  // Create a task for testProcessParticle, that we can run and then run it via a queue
  testProcessParticle testProcessParticle;
  auto taskRunTestProcessParticle = createTaskKernel<Acc>(
      workDiv, testProcessParticle, getPtrNative(device_eventProcess), getPtrNative(device_stepsProcess));

  enqueue(queue, taskRunTestProcessParticle);
  // Copy steps back to the host
  memcpy(queue, host_stepsProcess, device_stepsProcess, bufferExtent);

  // Then we test the output - the number of steps should be at least 1, but there is no maximum value.
  // Thus we simply test that the nSteps incremented beyond the initial zero value.
  testOK = true;
  for (unsigned int counter = 0; counter < NPART; counter++) {
    if (!(getPtrNative(host_stepsProcess)[counter] > 0)) testOK = false;
  }

  std::cout << "Status of testProcessParticle is " << testOK << std::endl;

  return 0;
}