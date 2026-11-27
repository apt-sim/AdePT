// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <AdePT/transport/random/Ranluxpp.h>

#include <G4HepEmRandomEngine.hh>

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
// G4HepEm expects the consumer to supply device-side implementations for
// flat/flatArray. AdePT binds that interface to RanluxppDouble here.
// Device pass only: the host definitions come from the G4HepEm library. clang (HIP) matches a definition to its
// declaration by execution space, so there it carries the host-device attribute of the declaration.
#if defined(__HIP__)
#define ADEPT_HEPEM_RNG_ATT __host__ __device__
#else
#define ADEPT_HEPEM_RNG_ATT __device__
#endif
inline ADEPT_HEPEM_RNG_ATT double G4HepEmRandomEngine::flat()
{
  return ((RanluxppDouble *)fObject)->Rndm();
}

inline ADEPT_HEPEM_RNG_ATT void G4HepEmRandomEngine::flatArray(const int size, double *vect)
{
  for (int i = 0; i < size; i++) {
    vect[i] = ((RanluxppDouble *)fObject)->Rndm();
  }
}
#undef ADEPT_HEPEM_RNG_ATT
#endif
