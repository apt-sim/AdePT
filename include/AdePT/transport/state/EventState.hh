// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRANSPORT_STATE_EVENT_STATE_HH
#define ADEPT_TRANSPORT_STATE_EVENT_STATE_HH

namespace AsyncAdePT {

struct GPUstate;

static constexpr int kMaxThreads = 256;

// Keep the public transport header independent of the full CUDA-side GPUstate
// definition while still allowing unique_ptr ownership.
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
  RequestStepFlush,
  SwappingStepBuffers,
  FlushingSteps,
  StepsFlushed,
  DeviceFlushed
};

} // namespace AsyncAdePT

#endif
