// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_G4HEPEM_RANDOM_ENGINE_DEVICE_IMPL_HH
#define ADEPT_G4HEPEM_RANDOM_ENGINE_DEVICE_IMPL_HH

#include <AdePT/copcore/Ranluxpp.h>

#include <G4HepEmRandomEngine.hh>

#ifdef __CUDA_ARCH__
// G4HepEm expects the consumer to supply device-side implementations for
// flat/flatArray. AdePT binds that interface to RanluxppDouble here.
inline __device__ double G4HepEmRandomEngine::flat()
{
  return ((RanluxppDouble *)fObject)->Rndm();
}

inline __device__ void G4HepEmRandomEngine::flatArray(const int size, double *vect)
{
  for (int i = 0; i < size; i++) {
    vect[i] = ((RanluxppDouble *)fObject)->Rndm();
  }
}
#endif

#endif
