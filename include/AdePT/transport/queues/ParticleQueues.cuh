// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRANSPORT_QUEUES_PARTICLE_QUEUES_CUH
#define ADEPT_TRANSPORT_QUEUES_PARTICLE_QUEUES_CUH

#include <AdePT/transport/containers/MParray.h>
#include <AdePT/transport/tracks/ParticleTypes.hh>

#include <utility>

namespace AsyncAdePT {

// A bundle of queues per particle type:
//  * Two for active particles, one for the current iteration and the second for the next.
struct ParticleQueues {
  /*
Gamma interactions:
0 - Conversion
1 - Compton
2 - Photoelectric
3 - Unused
4 - Relocation

Electron interactions:
0 - Ionization
1 - Bremsstrahlung
2 - Unused
3 - Unused
4 - Relocation

Positron interactions:
0 - Ionization
1 - Bremsstrahlung
2 - In flight annihilation
3 - Stopped annihilation
4 - Relocation

In-flight and stopped annihilation use different codes but may be merged to save space
in unused queues or if launching one kernel is faster than two smaller ones

It is not straightforward to allocate just the needed queues per particle type because
ParticleQueues needs to be passed by copy to the kernels, which means that we can't do
dynamic allocations
*/
  static constexpr char numInteractions = 5;
  adept::MParray *nextActive;
  adept::MParray *initiallyActive;
#ifdef ADEPT_USE_SPLIT_KERNELS
  adept::MParray *propagation;
  adept::MParray *interactionQueues[numInteractions];
#endif

  void SwapActive() { std::swap(initiallyActive, nextActive); }
};

/// @brief Named array-index enum for the per-species GPU state arrays.
///
/// Carries the three physical species plus Woodcock-tracking sentinels (GammaWDT,
/// NumParticleQueues). For physics data (steps, tracks) use the free @ref ParticleType
/// enum class instead; a static_assert below guarantees the numeric values stay in sync.
enum GPUQueueIndex {
  Electron = 0,
  Positron = 1,
  Gamma    = 2,

  NumSpecies,
  // alias for Woodcock tracking gammas:
  // as there is no explicit Woodcock tracking species, but NumSpecies is used to loop over
  // AllParticleQueues (which contain the Woodcock tracking gammas), an alias is used here to mark their access
  GammaWDT          = NumSpecies,
  NumParticleQueues = NumSpecies + 1
};

static_assert(GPUQueueIndex::Electron == static_cast<int>(ParticleType::Electron),
              "GPUQueueIndex and ParticleType electron values must match");
static_assert(GPUQueueIndex::Positron == static_cast<int>(ParticleType::Positron),
              "GPUQueueIndex and ParticleType positron values must match");
static_assert(GPUQueueIndex::Gamma == static_cast<int>(ParticleType::Gamma),
              "GPUQueueIndex and ParticleType gamma values must match");

static constexpr double kRelativeQueueSize[GPUQueueIndex::NumSpecies] = {0.35, 0.15, 0.5};

// A bundle of queues for the three particle types.
struct AllParticleQueues {
  // AllParticleQueues has queues for each particle type + one for Woodcock tracking
  ParticleQueues queues[GPUQueueIndex::NumParticleQueues];
};

#ifdef ADEPT_USE_SPLIT_KERNELS
// A bundle of queues per interaction type
struct AllInteractionQueues {
  adept::MParray *queues[ParticleQueues::numInteractions];
};
#endif

struct QueueIndexPair {
  unsigned int slot;
  short queue;
};

} // namespace AsyncAdePT

#endif
