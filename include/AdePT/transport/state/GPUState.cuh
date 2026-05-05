// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRANSPORT_STATE_GPU_STATE_CUH
#define ADEPT_TRANSPORT_STATE_GPU_STATE_CUH

#include <AdePT/transport/containers/MParrayT.h>
#include <AdePT/transport/containers/SlotManager.cuh>
#include <AdePT/transport/magneticfield/GeneralMagneticField.cuh>
#include <AdePT/transport/queues/ParticleManager.cuh>
#include <AdePT/transport/state/EventState.hh>
#include <AdePT/transport/state/TransportStats.hh>
#include <AdePT/transport/steps/GPUStepTransferManager.cuh>
#include <AdePT/transport/tracks/Track.cuh>
#include <AdePT/transport/support/ResourceManagement.cuh>

#ifdef ADEPT_USE_SPLIT_KERNELS
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmGammaTrack.hh>
#endif

#include <atomic>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

namespace AsyncAdePT {

/// @brief Holds all GPU resources needed to manage in-flight tracks of one particle species:
///        track buffer, slot manager, interaction queues, CUDA stream and event.
template <typename TrackT>
struct SpeciesState {
  TrackT *tracks{nullptr};
  SlotManager *slotManager{nullptr};
  ParticleQueues queues{};
  ADEPT_DEVICE_API_SYMBOL(Stream_t) stream {};
  ADEPT_DEVICE_API_SYMBOL(Event_t) event {};
};

#ifdef ADEPT_USE_SPLIT_KERNELS
struct HepEmBuffers {
  G4HepEmElectronTrack *electronsHepEm;
  G4HepEmElectronTrack *positronsHepEm;
  G4HepEmGammaTrack *gammasHepEm;
};
#endif

// Pointers to track storage for each particle type
struct TracksAndSlots {
  ChargedTrack *electrons;
  ChargedTrack *positrons;
  NeutralTrack *gammas;
  SlotManager *const slotManagers[GPUQueueIndex::NumSpecies];

  __device__ __forceinline__ short ThreadIdAt(unsigned int particleType, SlotManager::value_type slot) const
  {
    if (particleType == GPUQueueIndex::Electron) return electrons[slot].threadId;
    if (particleType == GPUQueueIndex::Positron) return positrons[slot].threadId;
    return gammas[slot].threadId;
  }
};

struct AllSlotManagers {
  SlotManager slotManagers[GPUQueueIndex::NumSpecies];
};

struct GPUstate {

#ifdef ADEPT_USE_EXT_BFIELD
  // If using a general magnetic field, GPUstate will store the host-side instance
  GeneralMagneticField magneticField;
#endif

  SpeciesState<ChargedTrack> electrons;
  SpeciesState<ChargedTrack> positrons;
  SpeciesState<NeutralTrack> gammas;

  // particle queues for gammas doing woodcock tracking. Only the `initiallyActive` and `nextActive` queues are
  // allocated.
  ParticleQueues woodcockQueues;

  std::vector<void *> allCudaPointers;
  // Create a stream to synchronize kernels of all particle types.
  ADEPT_DEVICE_API_SYMBOL(Stream_t) stream; ///< all-particle sync stream

  static constexpr unsigned int nSlotManager_dev = 3;

  AllSlotManagers allmgr_h;              // All host slot managers, statically allocated
  SlotManager *slotManager_dev{nullptr}; // All device slot managers

#ifdef ADEPT_USE_SPLIT_KERNELS
  HepEmBuffers hepEmBuffers_d; // All device buffers of hepem tracks
#endif

  Stats *stats_dev{nullptr}; ///< statistics object pointer on device
  Stats *stats{nullptr};     ///< statistics object pointer on host

  std::unique_ptr<GPUStepTransferManager> fGPUStepTransferManager;

  adept::MParrayT<QueueIndexPair> *injectionQueue;

  enum class InjectState { Idle, CreatingSlots, ReadyToEnqueue, Enqueueing };
  std::atomic<InjectState> injectState;
  std::atomic_bool runTransport{true}; ///< Keep transport thread running

  ~GPUstate()
  {
    try {
      if (stats) ADEPT_DEVICE_API_CALL(FreeHost(stats));
      if (stream) ADEPT_DEVICE_API_CALL(StreamDestroy(stream));

      auto destroySpeciesSync = [](auto &particleType) {
        if (particleType.stream) ADEPT_DEVICE_API_CALL(StreamDestroy(particleType.stream));
        if (particleType.event) ADEPT_DEVICE_API_CALL(EventDestroy(particleType.event));
      };
      destroySpeciesSync(electrons);
      destroySpeciesSync(positrons);
      destroySpeciesSync(gammas);
      for (void *ptr : allCudaPointers) {
        ADEPT_DEVICE_API_CALL(Free(ptr));
      }
    } catch (const std::exception &e) {
      std::cerr << "\033[31m" << "GPUstate::~GPUstate : Error during device API call" << "\033[0m" << std::endl;
      std::cerr << "\033[31m" << e.what() << "\033[0m" << std::endl;
    }
    allCudaPointers.clear();
  }
};

} // namespace AsyncAdePT

#endif
