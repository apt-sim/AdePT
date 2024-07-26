// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#include "VecGeom/navigation/NavigationState.h"

#ifndef ADEPT_SCORING_H
#define ADEPT_SCORING_H

// Templates for the AdePTScoring CUDA methods

// These methods are meant to be templated on a struct containing the necessary information
namespace adept_scoring
{
  template <typename Scoring>
  Scoring* InitializeOnGPU(Scoring *scoring){}

  template <typename Scoring>
  void FreeGPU(Scoring *scoring, Scoring *scoring_dev){}

  template <typename Scoring>
  __device__ void RecordHit(Scoring *scoring_dev, char aParticleType, double aStepLength, double aTotalEnergyDeposit,
                          vecgeom::NavigationState const *aPreState, vecgeom::Vector3D<Precision> *aPrePosition,
                          vecgeom::Vector3D<Precision> *aPreMomentumDirection,
                          vecgeom::Vector3D<Precision> *aPrePolarization, double aPreEKin, double aPreCharge,
                          vecgeom::NavigationState const *aPostState, vecgeom::Vector3D<Precision> *aPostPosition,
                          vecgeom::Vector3D<Precision> *aPostMomentumDirection,
                          vecgeom::Vector3D<Precision> *aPostPolarization, double aPostEKin, double aPostCharge){}

template <typename Scoring>
__device__ void AccountProduced(Scoring *scoring_dev, int num_ele, int num_pos, int num_gam);

template <typename Scoring>
__device__ __forceinline__ void EndOfIterationGPU(Scoring *scoring_dev);

template <typename Scoring, typename IntegrationLayer>
inline void EndOfIteration(Scoring &scoring, Scoring *scoring_dev, cudaStream_t &stream, IntegrationLayer &integration);

template <typename Scoring, typename IntegrationLayer>
inline void EndOfTransport(Scoring &scoring, Scoring *scoring_dev, cudaStream_t &stream, IntegrationLayer &integration);
}

#endif