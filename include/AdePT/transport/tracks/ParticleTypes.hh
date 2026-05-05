// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_PARTICLE_TYPES_HH
#define ADEPT_PARTICLE_TYPES_HH

/// @brief Strongly-typed enum for the three EM particle species tracked by AdePT.
///
/// Used in GPUStep, SecondaryInitData, HostTrackData, and anywhere a step or track
/// needs to carry its particle species as data.
///
/// Note: The numeric values (0, 1, 2) intentionally match `GPUQueueIndex`
/// (`transport/queues/ParticleQueues.cuh`), which is used as an array index for
/// GPU queue/state arrays. Both enums represent the same three species, but
/// `GPUQueueIndex` also carries `NumSpecies` / `GammaWDT` /
/// `NumParticleQueues` sentinels for loop bounds and array sizing.
/// Use `ParticleType` for physics data (steps, tracks); use
/// `GPUQueueIndex::{Electron,Positron,Gamma}` for GPU state array indexing.
/// A static_assert in `ParticleQueues.cuh` guarantees the values
/// stay in sync.
enum class ParticleType : char {
  Electron = 0,
  Positron = 1,
  Gamma    = 2,
};

#endif // ADEPT_PARTICLE_TYPES_HH
