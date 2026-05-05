// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRANSPORT_QUEUES_PARTICLE_MANAGER_CUH
#define ADEPT_TRANSPORT_QUEUES_PARTICLE_MANAGER_CUH

#include <AdePT/transport/containers/SlotManager.cuh>
#include <AdePT/transport/queues/ParticleQueues.cuh>
#include <AdePT/transport/tracks/Track.cuh>

#include <utility>

namespace AsyncAdePT {

// A bundle of pointers to generate particles of an implicit type.
template <typename TrackT>
struct SpeciesParticleManager {
  TrackT *fTracks;
  SlotManager *fSlotManager;
  adept::MParray *fActiveQueue;
  adept::MParray *fNextActiveQueue;

public:
  __host__ __device__ SpeciesParticleManager(TrackT *tracks, SlotManager *slotManager, adept::MParray *activeQueue,
                                             adept::MParray *nextActiveQueue)
      : fTracks(tracks), fSlotManager(slotManager), fActiveQueue(activeQueue), fNextActiveQueue(nextActiveQueue)
  {
  }

  /// Obtain track at given slot position
  __device__ __forceinline__ TrackT &TrackAt(SlotManager::value_type slot) { return fTracks[slot]; }

  /// Obtain a slot for a track, but don't enqueue.
  __device__ auto NextSlot() { return fSlotManager->NextSlot(); }

  // enqueue into next-active queue
  __device__ __forceinline__ bool EnqueueNext(SlotManager::value_type slot)
  {
    return fNextActiveQueue->push_back(slot);
  }

  // size of the active queue
  __device__ __forceinline__ int ActiveSize() const { return fActiveQueue->size(); }

  // read slot from active queue by index
  __device__ __forceinline__ SlotManager::value_type ActiveAt(int i) const { return (*fActiveQueue)[i]; }

  /// Construct a track at the given location, forwarding all arguments to the constructor.
  template <typename... Ts>
  __device__ TrackT &InitTrack(SlotManager::value_type slot, Ts &&...args)
  {
    return *new (fTracks + slot) TrackT{std::forward<Ts>(args)...};
  }

  /// Obtain a slot and construct a track, forwarding args to the track constructor.
  template <typename... Ts>
  __device__ TrackT &NextTrack(Ts &&...args)
  {
    const auto slot = NextSlot();
    // next track is only visible in next GPU iteration, therefore pushed in the NextActiveQueue
    fNextActiveQueue->push_back(slot);
    auto &track = InitTrack(slot, std::forward<Ts>(args)...);
    return track;
  }
};

// A bundle of generators for the three particle types.
struct ParticleManager {
  SpeciesParticleManager<ChargedTrack> electrons;
  SpeciesParticleManager<ChargedTrack> positrons;
  SpeciesParticleManager<NeutralTrack> gammas;
  SpeciesParticleManager<NeutralTrack> gammasWDT;
};

} // namespace AsyncAdePT

#endif
