// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_THREAD_ID_HH
#define ADEPT_THREAD_ID_HH

#include "G4Threading.hh"

namespace adept_integration {

/// @brief Return the non-negative Geant4 worker slot used by AdePT per-thread host arrays.
/// @details
/// In sequential Geant4, `G4GetThreadId()` returns `MASTER_ID` (-1) or
/// `SEQUENTIAL_ID` (-2). AdePT uses non-negative slot indices 0..N-1, so map
/// any negative Geant4 thread id to slot 0.
inline int GetThreadId()
{
  const auto tid = G4Threading::G4GetThreadId();
  return (tid < 0) ? 0 : tid;
}

} // namespace adept_integration

#endif
