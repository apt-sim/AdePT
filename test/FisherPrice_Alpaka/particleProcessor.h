// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file particleProcessor.h
 * @brief simulates electrons and photons - it can calculate ionisaton energy loss, make a particle undergo
 * brehmstrahhlung or pair production and move a particle along the x-direction. All class functions are specified to
 * run the accelerator (device) only.
 * @author Davide Costanzo (d.costanzo@sheffield.ac.uk) and Mark Hodgkinson (d.costanzo@sheffield.ac.uk)
 * */

#ifndef PARTICLEPROCESSOR_H
#define PARTICLEPROCESSOR_H

#include "part.h"
#include "particleStack.h"
#include "sensitive.h"
#include <alpaka/alpaka.hpp>

class particleProcessor {

public:
  /** @brief A minimal constructor */
  ALPAKA_FN_ACC particleProcessor() { m_encourageSplit = false;}

  /** @brief A minimal destructor */
  ALPAKA_FN_ACC ~particleProcessor() {}

  /**
   * @brief calculates the energy loss of a particle, mypart, in the range 0 to 0.2 MeV in the direction of the
   * momentum. Acc is the Alpaka acceleraror type (e.g. CPU or GPU), mypart is the part to calculate the energy loss for
   * and generator is a random number generator provided by alpaka. Assumes the generator has already been initialised
   * with a seed by the client. Returns a float value representing the energy lost, which is calculated as a randum
   * number (between zero and one) multiplied by 0.2 MeV.
   * */
  template <typename Acc>
  ALPAKA_FN_ACC float energyLoss(Acc const &acc, part &mypart,
                                 alpaka::rand::generator::uniform_cuda_hip::Xor &generator);

  /**
   * @brief Takes a list of particles on a specific thread and processes them. Processing means that the particles moves
   * along the x-direction in finite steps. In each step a new particle(s) may or may not be generated - if so it is
   * added to an internal. to this function, array of particles on the stack. The function continues to step until the
   * initial particles momentum drops below 1 MeV. Acc is the Alpaka acceleraror type (e.g. CPU or GPU), partList is a
   * list of part, iTh is the thread being used, generator is a random number generator provided by alpaka and SD is an
   * instance of a c++ class, of type sensitive, representing a sensitive detector. Returns an unsigned integer
   * representing the number of steps undertaken.
   */
  template <typename Acc>
  ALPAKA_FN_ACC unsigned int processParticle(Acc const &acc, part *partList, const int &iTh,
                                             alpaka::rand::generator::uniform_cuda_hip::Xor &generator, sensitive &SD);

  /**
   * @brief throws a random number to determine whether to generate secondary particles. An electron will always
   * generate a photon and a photon will always generate an electron-positron pair. Momentum is conserved such that the
   * initial particle reduces in momentum by the momentum of the additional particles. Acc is the Alpaka acceleraror
   * type (e.g. CPU or GPU), mypart is the particle to consider and generator is a random number generator provided by
   * alpaka. Returns a new particle representing either the photon (in e -> gamma brehmstrahhlung case) or positron (in
   * the photon -> e+e- case). If we don't produce any additional particles 0 is returned.
   */
  template <typename Acc>
  ALPAKA_FN_ACC part *splitParticle(Acc const &acc, part &mypart,
                                    alpaka::rand::generator::uniform_cuda_hip::Xor &generator);

  /**
   * @brief steps the particle 0.1 mm in the x-direction. Additionally determines how much energy it loses via @sa
   * particleProcessor::energyLoss and whether it produces secondary particles via @sa particleProcessor::splitParticle.
   * Acc is the Alpaka acceleraror type (e.g. CPU or GPU), processPart is the part to process, generator is a random
   * number generator provided by alpaka and SD is an instance of a c++ class, of type sensitive, representing a
   * sensitive detector. Returns the output of @sa particleProcessor::splitParticle, which is a pointer to a c++ class
   * of type part.
   */
  template <typename Acc>
  ALPAKA_FN_ACC part *step(Acc const &acc, part &processPart, alpaka::rand::generator::uniform_cuda_hip::Xor &generator,
                           sensitive &SD);

  /**
   * @brief If called this lowers the threshold used in @sa splitParticle from 0.99 to 0.5.
   * This is only to be done in unit tests in order to ensure all code paths are executed.
   */
  ALPAKA_FN_ACC void lowerThreshold() { m_encourageSplit = true;}

  private:
    /** Toggle to increase probablity of splitting when running unit tests, such that we ensure
     * all code execution paths are covered.
     */
    bool m_encourageSplit;


};

#endif

template <typename Acc>
ALPAKA_FN_ACC float particleProcessor::energyLoss(Acc const &acc, part &mypart,
                                                  alpaka::rand::generator::uniform_cuda_hip::Xor &generator)
{
  // take off a random 0 to 0.2 MeV in the direction of momentum
  auto func(alpaka::rand::distribution::createUniformReal<float>(acc));
  float enLoss = 0.02 * func(generator);
  vec3 theMom  = mypart.getMom();
  theMom.energyLoss(enLoss);
  mypart.setMom(theMom);
  return enLoss;
}

template <typename Acc>
ALPAKA_FN_ACC unsigned int particleProcessor::processParticle(Acc const &acc, part *partList, const int &iTh,
                                                              alpaka::rand::generator::uniform_cuda_hip::Xor &generator,
                                                              sensitive &SD)
{
  // Create a particleStack for this thread:
  particleStack<100> PS;
  // And push in the particle corresponding to this thread:
  PS.push(&partList[iTh]);

  part *mypart = PS.top();
  unsigned int nsteps = 0;
  int maxsize         = 0;
  bool firstParticle = true;
  while (!PS.empty()) {
    part *processPart = PS.top();
    PS.pop();
    do {
      nsteps++;
      part *newPart = this->step(acc, *processPart, generator, SD);
      if (newPart) PS.push(newPart); // if a new particle is generated we add it to the stack
      if (PS.size() > maxsize) maxsize = PS.size();
    } while (processPart->momentum() > 1.);
    //We only want to delete the particles created with "new" inside the call to the step function above.
    //The first particle in partList[iTh] is memory allocated outside this function, and deleting it is a "double free".
    if (firstParticle) firstParticle = false;
    else delete processPart;
  }
  return nsteps;
}

template <typename Acc>
ALPAKA_FN_ACC part *particleProcessor::splitParticle(Acc const &acc, part &mypart,
                                                     alpaka::rand::generator::uniform_cuda_hip::Xor &generator)
{
  // throw a random number if >0.99 do a splitting (e.g. a Brem or pair production)
  auto func(alpaka::rand::distribution::createUniformReal<float>(acc));

  float threshold = 0.99;
  if (m_encourageSplit) threshold = 0.5;

  if (func(generator) > threshold) {
    // This is a dummy splitter. We do a process a->b+c
    // b will take fracb of momentum with fracb betweek 0.5 and 0.9
    // c will take fracc of the momentum with fracc between 0. and 1-fracb
    // fracb+fracc has to be < 1 !!
    float fracb = func(generator) * 0.4 + 0.5; // number between 0.5 and 0.9
    float fracc = 1 - fracb;
    if (fracb + fracc > 1.0) printf("ARGHHH I MADE A MISTAKE \n");
    // copy momentum and scale it
    vec3 mom = mypart.getMom();
    mom.scaleLength(fracc);
    // Create a new particle in this position with the scaled down momentum
    part *newpart = new part(mypart.getPos(), mom, 22);
    // scale the momentum of the original particle
    vec3 origMom = mypart.getMom();
    origMom.scaleLength(fracb);
    mypart.setMom(origMom);
    // To make it look like an EM shower, if the particle is a photon we split into e+ e-
    if (mypart.getPType() == 22) {
      mypart.setPType(13);
      newpart->setPType(-13);
    }
    return newpart;
  }
  return 0;
}

template <typename Acc>
ALPAKA_FN_ACC part *particleProcessor::step(Acc const &acc, part &processPart,
                                            alpaka::rand::generator::uniform_cuda_hip::Xor &generator, sensitive &SD)
{
  // Move the particle by a deltaX=0.1mm in the momentum direction
  float deltaX = 0.1;
  float mom    = processPart.momentum();
  processPart.setPos(processPart.getPos() + (deltaX / mom) * processPart.getMom());
  float enLoss = this->energyLoss(acc, processPart, generator);

  return this->splitParticle(acc, processPart, generator); // randomly see if we split the particle
}
