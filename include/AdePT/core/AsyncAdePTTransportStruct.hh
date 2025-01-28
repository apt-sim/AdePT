// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNC_ADEPT_TRANSPORT_STRUCT_HH
#define ASYNC_ADEPT_TRANSPORT_STRUCT_HH

namespace AsyncAdePT {

struct GPUstate;
static constexpr int kMaxThreads = 256;

// We need a deleter for the unique_ptr to the GPUstate
// This deleter is implemented in AsyncAdePTTransportStruct.cuh
struct GPUstateDeleter {
  void operator()(GPUstate *ptr);
};

enum class EventState : unsigned char {
  NewTracksFromG4,
  G4RequestsFlush,
  Inject,
  InjectionCompleted,
  Transporting,
  WaitingForTransportToFinish,
  RequestHitFlush,
  FlushingHits,
  HitsFlushed,
  FlushingTracks,
  DeviceFlushed,
  LeakedTracksRetrieved,
  ScoringRetrieved
};

} // namespace AsyncAdePT

#endif
