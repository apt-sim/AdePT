// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNC_ADEPT_TRANSPORT_STRUCT_HH
#define ASYNC_ADEPT_TRANSPORT_STRUCT_HH

namespace AsyncAdePT {

static constexpr int kMaxThreads = 256;

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

}

#endif