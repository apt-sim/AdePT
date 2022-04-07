// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/Atomic.h>
#include <AdePT/LoopNavigator.h>
#include <AdePT/MParray.h>

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Stopwatch.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

#include <G4SystemOfUnits.hh>

#include <G4HepEmData.hh>
#include <G4HepEmElectronInit.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmMaterialInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmParametersInit.hh>

#define NOMSC
#define NOFLUCTUATION

#include <G4HepEmElectronManager.hh>
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmElectronInteractionBrem.hh>
#include <G4HepEmElectronInteractionIoni.hh>
#include <G4HepEmPositronInteractionAnnihilation.hh>
// Pull in implementation.
#include <G4HepEmRunUtils.icc>
#include <G4HepEmInteractionUtils.icc>
#include <G4HepEmElectronManager.icc>
#include <G4HepEmElectronInteractionBrem.icc>
#include <G4HepEmElectronInteractionIoni.icc>
#include <G4HepEmPositronInteractionAnnihilation.icc>

#include <CopCore/Global.h>
#include <CopCore/PhysicalConstants.h>
#include <CopCore/Ranluxpp.h>

#include <ConstBzFieldStepper.h>
#include <fieldPropagatorConstBz.h>

#include "example8.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>

// A data structure to represent a particle.
struct Track {
  using Precision = vecgeom::Precision;
  RanluxppDouble rng_state;
  double energy;
  double numIALeft[3];

  vecgeom::Vector3D<Precision> pos;
  vecgeom::Vector3D<Precision> dir;
  vecgeom::NavStateIndex current_state;
  vecgeom::NavStateIndex next_state;

  char pdg; // Large enough for e/g. Then uint16_t for p/n

  __device__ __host__ double uniform() { return rng_state.Rndm(); }

  __device__ __host__ int charge() const // charge for e- / e+ / gamma only
  {
    int chrg = (pdg == pdgPositron ? 1 : 0) + (pdg == pdgElectron ? -1 : 0);
    return chrg;
  }

  __device__ __host__ double mass() const // Rest mass for e- / e+ / gamma only
  {
    return (pdg == pdgGamma) ? 0.0 : copcore::units::kElectronMassC2;
  }

  __device__ __host__ void SwapStates()
  {
    auto state          = this->current_state;
    this->current_state = this->next_state;
    this->next_state    = state;
  }

  static constexpr char pdgElectron = 11;
  static constexpr char pdgPositron = -11;
  static constexpr char pdgGamma    = 22;
};

// A data structure for some simple scoring.
struct Scoring {
  adept::Atomic_t<int> hits;
  adept::Atomic_t<int> secondaries;
  adept::Atomic_t<float> totalEnergyDeposit;
};

// A data structure to manage slots in the track storage.
class SlotManager {
  adept::Atomic_t<int> fNextSlot;
  const int fMaxSlot;

public:
  __host__ __device__ SlotManager(int maxSlot) : fMaxSlot(maxSlot) { fNextSlot = 0; }

  __host__ __device__ int nextSlot()
  {
    int next = fNextSlot.fetch_add(1);
    if (next >= fMaxSlot) return -1;
    return next;
  }
};

constexpr vecgeom::Precision BzFieldValue = 0.1 * copcore::units::tesla;

__constant__ __device__ struct G4HepEmParameters g4HepEmPars;
__constant__ __device__ struct G4HepEmData g4HepEmData;

struct G4HepEmState {
  G4HepEmData data;
  G4HepEmParameters parameters;
};

static G4HepEmState *InitG4HepEm()
{
  G4HepEmState *state = new G4HepEmState;
  InitG4HepEmData(&state->data);
  InitHepEmParameters(&state->parameters);

  InitMaterialAndCoupleData(&state->data, &state->parameters);

  InitElectronData(&state->data, &state->parameters, true);
  InitElectronData(&state->data, &state->parameters, false);

  G4HepEmMatCutData *cutData = state->data.fTheMatCutData;
  std::cout << "fNumG4MatCuts = " << cutData->fNumG4MatCuts << ", fNumMatCutData = " << cutData->fNumMatCutData
            << std::endl;

  // Copy to GPU.
  CopyG4HepEmDataToGPU(&state->data);
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmPars, &state->parameters, sizeof(G4HepEmParameters)));

  // Create G4HepEmData with the device pointers.
  G4HepEmData dataOnDevice;
  dataOnDevice.fTheMatCutData   = state->data.fTheMatCutData_gpu;
  dataOnDevice.fTheMaterialData = state->data.fTheMaterialData_gpu;
  dataOnDevice.fTheElementData  = state->data.fTheElementData_gpu;
  dataOnDevice.fTheElectronData = state->data.fTheElectronData_gpu;
  dataOnDevice.fThePositronData = state->data.fThePositronData_gpu;
  dataOnDevice.fTheSBTableData  = state->data.fTheSBTableData_gpu;
  // The other pointers should never be used.
  dataOnDevice.fTheMatCutData_gpu   = nullptr;
  dataOnDevice.fTheMaterialData_gpu = nullptr;
  dataOnDevice.fTheElementData_gpu  = nullptr;
  dataOnDevice.fTheElectronData_gpu = nullptr;
  dataOnDevice.fThePositronData_gpu = nullptr;
  dataOnDevice.fTheSBTableData_gpu  = nullptr;

  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmData, &dataOnDevice, sizeof(G4HepEmData)));

  return state;
}

static void FreeG4HepEm(G4HepEmState *state)
{
  FreeG4HepEmData(&state->data);
  delete state;
}

class RanluxppDoubleEngine : public G4HepEmRandomEngine {
  // Wrapper functions to call into RanluxppDouble.
  static __host__ __device__ double flatWrapper(void *object) { return ((RanluxppDouble *)object)->Rndm(); }
  static __host__ __device__ void flatArrayWrapper(void *object, const int size, double *vect)
  {
    for (int i = 0; i < size; i++) {
      vect[i] = ((RanluxppDouble *)object)->Rndm();
    }
  }

public:
  __host__ __device__ RanluxppDoubleEngine(RanluxppDouble *engine)
      : G4HepEmRandomEngine(/*object=*/engine, &flatWrapper, &flatArrayWrapper)
  {
  }
};

__host__ __device__ void InitSecondary(Track &secondary, const Track &parent)
{
  // Initialize a new PRNG state.
  secondary.rng_state = parent.rng_state;
  secondary.rng_state.Skip(1 << 15);

  // The caller is responsible to set the energy.
  secondary.numIALeft[0] = -1.0;
  secondary.numIALeft[1] = -1.0;
  secondary.numIALeft[2] = -1.0;

  // The caller is responsible to set the particle type (via pdg).

  // A secondary inherits the position of its parent; the caller is responsible
  // to update the directions.
  secondary.pos           = parent.pos;
  secondary.current_state = parent.current_state;
  secondary.next_state    = parent.current_state;
}

// Create a pair of e-/e+ from the intermediate gamma.
__host__ __device__ void PairProduce(Track *allTracks, SlotManager *manager, adept::MParray *activeQueue,
                                     const Track &currentTrack, double energy, const double *dir)
{
  int electronSlot = manager->nextSlot();
  if (electronSlot == -1) {
    COPCORE_EXCEPTION("No slot available for secondary electron track");
  }
  activeQueue->push_back(electronSlot);
  Track &electron = allTracks[electronSlot];

  int positronSlot = manager->nextSlot();
  if (positronSlot == -1) {
    COPCORE_EXCEPTION("No slot available for secondary positron track");
  }
  activeQueue->push_back(positronSlot);
  Track &positron = allTracks[positronSlot];

  // TODO: Distribute energy and momentum.
  double remainingEnergy = energy - 2 * copcore::units::kElectronMassC2;

  InitSecondary(electron, /*parent=*/currentTrack);
  electron.pdg    = Track::pdgElectron;
  electron.energy = remainingEnergy / 2;
  electron.dir.Set(dir[0], dir[1], dir[2]);

  InitSecondary(positron, /*parent=*/currentTrack);
  positron.pdg    = Track::pdgPositron;
  positron.energy = remainingEnergy / 2;
  positron.dir.Set(dir[0], dir[1], dir[2]);
}

// Compute the physics and geometry step limit, transport the particle while
// applying the continuous effects and possibly select a discrete process that
// could generate secondaries.
__global__ void PerformStep(Track *allTracks, SlotManager *manager, const adept::MParray *currentlyActive,
                            adept::MParray *activeQueue, adept::MParray *relocateQueue, Scoring *scoring)
{
  fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);

  int activeSize = currentlyActive->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*currentlyActive)[i];
    Track &currentTrack = allTracks[slot];

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack elTrack;
    G4HepEmTrack *theTrack = elTrack.GetTrack();
    theTrack->SetEKin(currentTrack.energy);
    // For now, just assume a single material.
    theTrack->SetMCIndex(1);
    // In this kernel, we only have electrons and positrons.
    const bool isElectron = currentTrack.pdg == Track::pdgElectron;
    theTrack->SetCharge(isElectron ? -1.0 : 1.0);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft                  = -std::log(currentTrack.uniform());
        currentTrack.numIALeft[ip] = numIALeft;
      }
      theTrack->SetNumIALeft(numIALeft, ip);
    }

    // Call G4HepEm to compute the physics step limit.
    G4HepEmElectronManager::HowFar(&g4HepEmData, &g4HepEmPars, &elTrack, nullptr);

    // Get result into variables.
    double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();
    // The phyiscal step length is the amount that the particle experiences
    // which might be longer than the geometrical step length due to MSC. As
    // long as we call PerformContinuous in the same kernel we don't need to
    // care, but we need to make this available when splitting the operations.
    // double physicalStepLength = elTrack.GetPStepLength();
    int winnerProcessIndex = theTrack->GetWinnerProcessIndex();
    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    bool propagated = true;
    double geometryStepLength = fieldPropagatorBz.ComputeStepAndNextVolume(
        currentTrack.energy, currentTrack.mass(), currentTrack.charge(), geometricalStepLengthFromPhysics,
        currentTrack.pos, currentTrack.dir, currentTrack.current_state, currentTrack.next_state, propagated);

    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(currentTrack.next_state.IsOnBoundary());

    // Apply continuous effects.
    bool stopped = G4HepEmElectronManager::PerformContinuous(&g4HepEmData, &g4HepEmPars, &elTrack, nullptr);
    // Collect the changes.
    currentTrack.energy = theTrack->GetEKin();
    scoring->totalEnergyDeposit.fetch_add(theTrack->GetEnergyDeposit());

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

    if (stopped) {
      if (!isElectron) {
        // For a stopped positron, we should call annihilation but this produces
        // a gamma which we don't yet have processes for. Deposit the amount of
        // energy that the photon would have from the annihilation at rest with
        // an electron.
        scoring->totalEnergyDeposit.fetch_add(2 * copcore::units::kElectronMassC2);
      }
      // Particles are killed by not enqueuing them into the new activeQueue.
      continue;
    }

    if (currentTrack.next_state.IsOnBoundary()) {
      // For now, just count that we hit something.
      scoring->hits++;

      activeQueue->push_back(slot);
      relocateQueue->push_back(slot);

      // Move to the next boundary.
      currentTrack.SwapStates();
      continue;
    } else if (!propagated) {
      // Did not yet reach the interaction point due to error in the magnetic
      // field propagation. Try again next time.
      activeQueue->push_back(slot);
      continue;
    } else if (winnerProcessIndex < 0) {
      // No discrete process, move on.
      activeQueue->push_back(slot);
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[winnerProcessIndex] = -1.0;

    // Check if a delta interaction happens instead of the real discrete process.
    if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, currentTrack.uniform())) {
      // A delta interaction happened, move on.
      activeQueue->push_back(slot);
      continue;
    }

    // Perform the discrete interaction.
    RanluxppDoubleEngine rnge(&currentTrack.rng_state);

    // For now, just assume a single material.
    int theMCIndex        = 1;
    const double energy   = currentTrack.energy;
    const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[theMCIndex].fSecElProdCutE;

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke ionization (for e-/e+):
      double deltaEkin = (isElectron) ? G4HepEmElectronInteractionIoni::SampleETransferMoller(theElCut, energy, &rnge)
                                      : G4HepEmElectronInteractionIoni::SampleETransferBhabha(theElCut, energy, &rnge);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondary[3];
      G4HepEmElectronInteractionIoni::SampleDirections(energy, deltaEkin, dirSecondary, dirPrimary, &rnge);

      int secondarySlot = manager->nextSlot();
      if (secondarySlot == -1) {
        COPCORE_EXCEPTION("No slot available for secondary track");
      }
      activeQueue->push_back(secondarySlot);
      Track &secondary = allTracks[secondarySlot];
      scoring->secondaries++;

      InitSecondary(secondary, /*parent=*/currentTrack);
      secondary.pdg    = Track::pdgElectron;
      secondary.energy = deltaEkin;
      secondary.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);

      currentTrack.energy = energy - deltaEkin;
      currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      // The current track continues to live.
      activeQueue->push_back(slot);
      break;
    }
    case 1: {
      // Invoke model for Bremsstrahlung: either SB- or Rel-Brem.
      double logEnergy = std::log(energy);
      double deltaEkin = energy < g4HepEmPars.fElectronBremModelLim
                             ? G4HepEmElectronInteractionBrem::SampleETransferSB(&g4HepEmData, energy, logEnergy,
                                                                                 theMCIndex, &rnge, isElectron)
                             : G4HepEmElectronInteractionBrem::SampleETransferRB(&g4HepEmData, energy, logEnergy,
                                                                                 theMCIndex, &rnge, isElectron);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondary[3];
      G4HepEmElectronInteractionBrem::SampleDirections(energy, deltaEkin, dirSecondary, dirPrimary, &rnge);

      // We would need to create a gamma, but only do so if it has enough energy
      // to immediately pair-produce. Otherwise just deposit the energy locally.
      if (deltaEkin > 2 * copcore::units::kElectronMassC2) {
        PairProduce(allTracks, manager, activeQueue, currentTrack, deltaEkin, dirSecondary);
        scoring->secondaries += 2;
      } else {
        scoring->totalEnergyDeposit.fetch_add(deltaEkin);
      }

      currentTrack.energy = energy - deltaEkin;
      currentTrack.dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      // The current track continues to live.
      activeQueue->push_back(slot);
      break;
    }
    case 2: {
      // Invoke annihilation (in-flight) for e+
      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double theGamma1Ekin, theGamma2Ekin;
      double theGamma1Dir[3], theGamma2Dir[3];
      G4HepEmPositronInteractionAnnihilation::SampleEnergyAndDirectionsInFlight(
          energy, dirPrimary, &theGamma1Ekin, theGamma1Dir, &theGamma2Ekin, theGamma2Dir, &rnge);

      // For each of the two gammas, pair-produce if they have enough energy
      // or deposit the energy locally.
      if (theGamma1Ekin > 2 * copcore::units::kElectronMassC2) {
        PairProduce(allTracks, manager, activeQueue, currentTrack, theGamma1Ekin, theGamma1Dir);
        scoring->secondaries += 2;
      } else {
        scoring->totalEnergyDeposit.fetch_add(theGamma1Ekin);
      }
      if (theGamma2Ekin > 2 * copcore::units::kElectronMassC2) {
        PairProduce(allTracks, manager, activeQueue, currentTrack, theGamma2Ekin, theGamma2Dir);
        scoring->secondaries += 2;
      } else {
        scoring->totalEnergyDeposit.fetch_add(theGamma2Ekin);
      }

      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    }
  }
}

__device__ __forceinline__ vecgeom::VPlacedVolume const *ShuffleVolume(unsigned mask, vecgeom::VPlacedVolume const *ptr,
                                                                       int src)
{
  // __shfl_sync only accepts integer and floating point values, so we have to
  // cast into a type of approriate length...
  auto val = reinterpret_cast<unsigned long long>(ptr);
  val      = __shfl_sync(mask, val, src);
  return reinterpret_cast<vecgeom::VPlacedVolume const *>(val);
}

// A parallel version of LoopNavigator::RelocateToNextVolume. This function
// uses the parallelism of a warp to check daughters in parallel.
__global__ void RelocateToNextVolume(Track *allTracks, const adept::MParray *relocateQueue)
{
  // Determine which threads are active in the current warp.
  unsigned mask     = __activemask();
  int threadsInWrap = __popc(mask);
  // Count warps per block, including incomplete ones.
  int warpsPerBlock = (blockDim.x + warpSize - 1) / warpSize;
  // Find this thread's warp and lane.
  int warp = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;

  int queueSize = relocateQueue->size();
  // Note the loop setup: All threads in a block relocate one particle.
  // For comparison, here's the usual grid-strided loop header:
  //   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < queueSize; i += blockDim.x * gridDim.x)
  for (int i = blockIdx.x * warpsPerBlock + warp; i < queueSize; i += warpsPerBlock * gridDim.x) {
    const int slot      = (*relocateQueue)[i];
    Track &currentTrack = allTracks[slot];

    vecgeom::NavStateIndex &state = currentTrack.current_state;

    vecgeom::VPlacedVolume const *currentVolume;
    vecgeom::Precision localCoordinates[3];

    // The first lane removes all volumes from the state that were left, and
    // stores it in the variable currentVolume. During the process, it also
    // transforms the point to local coordinates, eventually stored in the
    // variable localCoordinates.
    if (lane == 0) {
      // Push the point inside the next volume.
      static constexpr double kPush = LoopNavigator::kBoundaryPush;
      vecgeom::Vector3D<vecgeom::Precision> pushed = currentTrack.pos + kPush * currentTrack.dir;

      // Calculate local point from global point.
      vecgeom::Transformation3D m;
      state.TopMatrix(m);
      vecgeom::Vector3D<vecgeom::Precision> localPoint = m.Transform(pushed);

      currentVolume = state.Top();

      // Store the transformed coordinates, to be broadcasted to the other
      // active threads in this warp.
      localCoordinates[0] = localPoint.x();
      localCoordinates[1] = localPoint.y();
      localCoordinates[2] = localPoint.z();
    }

    // Broadcast the values.
    currentVolume = ShuffleVolume(mask, currentVolume, 0);
    for (int dim = 0; dim < 3; dim++) {
      localCoordinates[dim] = __shfl_sync(mask, localCoordinates[dim], 0);
    }

    if (currentVolume) {
      unsigned hasNextVolume;
      do {

        vecgeom::Vector3D<vecgeom::Precision> localPoint(localCoordinates[0], localCoordinates[1], localCoordinates[2]);

        const auto &daughters = currentVolume->GetDaughters();
        auto daughtersSize    = daughters.size();
        vecgeom::Vector3D<vecgeom::Precision> transformedPoint;
        vecgeom::VPlacedVolume const *nextVolume = nullptr;
        // The active threads in the wrap check all daughters in parallel.
        for (int d = lane; d < daughtersSize; d += threadsInWrap) {
          const auto *daughter = daughters[d];
          if (daughter->Contains(localPoint, transformedPoint)) {
            nextVolume = daughter;
            break;
          }
        }

        // All active threads in the warp synchronize and vote which of them
        // found a daughter that is entered. The result has the Nth bit set if
        // the Nth lane has a nextVolume != nullptr.
        hasNextVolume = __ballot_sync(mask, nextVolume != nullptr);
        if (hasNextVolume != 0) {
          // Determine which volume to use if there are multiple: Just pick the
          // first one, corresponding to the position of the first set bit.
          int firstThread = __ffs(hasNextVolume) - 1;
          if (lane == firstThread) {
            localCoordinates[0] = transformedPoint.x();
            localCoordinates[1] = transformedPoint.y();
            localCoordinates[2] = transformedPoint.z();

            currentVolume = nextVolume;
            state.Push(currentVolume);
          }

          // Broadcast the values.
          currentVolume = ShuffleVolume(mask, currentVolume, firstThread);
          for (int dim = 0; dim < 3; dim++) {
            localCoordinates[dim] = __shfl_sync(mask, localCoordinates[dim], firstThread);
          }
        }
        // If hasNextVolume is zero, there is no point in synchronizing since
        // this will exit the loop.
      } while (hasNextVolume != 0);
    }

    // Finally the first lane again leaves all assembly volumes.
    if (lane == 0) {
      if (state.Top() != nullptr) {
        while (state.Top()->IsAssembly()) {
          state.Pop();
        }
        assert(!state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
      }
    }
  }
}

__global__ void FinishStep(adept::MParray *currentlyActive, const adept::MParray *nextActive,
                           adept::MParray *relocateQueue, int *inFlight)
{
  currentlyActive->clear();
  *inFlight = nextActive->size();
  relocateQueue->clear();
}

// Kernel function to initialize a single queue.
__global__ void InitQueue(adept::MParray *queue, size_t Capacity)
{
  adept::MParray::MakeInstanceAt(Capacity, queue);
}

// Kernel function to initialize a set of primary particles.
__global__ void InitPrimaries(Track *allTracks, SlotManager *manager, adept::MParray *activeQueue, int particles,
                              double energy, const vecgeom::VPlacedVolume *world)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particles; i += blockDim.x * gridDim.x) {
    const int slot = manager->nextSlot();
    if (slot == -1) {
      COPCORE_EXCEPTION("No slot available for primary track");
    }
    activeQueue->push_back(slot);
    Track &track = allTracks[slot];

    track.rng_state.SetSeed(314159265 * (i + 1));
    track.energy       = energy;
    track.numIALeft[0] = -1.0;
    track.numIALeft[1] = -1.0;
    track.numIALeft[2] = -1.0;

    track.pos = {0, 0, 0};
    track.dir = {1.0, 0, 0};
    LoopNavigator::LocatePointIn(world, track.pos, track.current_state, true);
    // next_state is initialized as needed.

    track.pdg = Track::pdgElectron;
  }
}

//
void example8(const vecgeom::cxx::VPlacedVolume *world, int particles, double energy)
{
  auto &cudaManager = vecgeom::cxx::CudaManager::Instance();
  cudaManager.LoadGeometry(world);
  cudaManager.Synchronize();

  const vecgeom::cuda::VPlacedVolume *world_dev = cudaManager.world_gpu();

  G4HepEmState *state = InitG4HepEm();

  // Capacity of the different containers aka the maximum number of particles.
  constexpr int Capacity = 256 * 1024;

  std::cout << "INFO: capacity of containers (incl. BlockData<track>) set at " << Capacity << std::endl;

  // Allocate memory to hold all tracks.
  Track *allTracks = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&allTracks, sizeof(Track) * Capacity));

  // Allocate object to manage slots.
  SlotManager *slotManager = nullptr;
  SlotManager slotManager_host(Capacity);
  COPCORE_CUDA_CHECK(cudaMalloc(&slotManager, sizeof(SlotManager)));
  COPCORE_CUDA_CHECK(cudaMemcpy(slotManager, &slotManager_host, sizeof(SlotManager), cudaMemcpyHostToDevice));

  // Allocate queues to remember particles:
  //  * Two for active particles, one for the current iteration and the second for the next.
  //  * One for all particles that need to be relocated to the next volume.
  constexpr int NumQueues = 3;
  const size_t queueSize  = adept::MParray::SizeOfInstance(Capacity);

  adept::MParray *queues[NumQueues];
  for (int i = 0; i < NumQueues; i++) {
    COPCORE_CUDA_CHECK(cudaMalloc(&queues[i], queueSize));
    InitQueue<<<1, 1>>>(queues[i], Capacity);
  }
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  adept::MParray *currentlyActive = queues[0];
  adept::MParray *nextActive      = queues[1];

  adept::MParray *relocateQueue = queues[2];

  // Allocate memory for an integer to transfer the number of particles in flight.
  int *inFlight_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&inFlight_dev, sizeof(int)));

  // Allocate and initialize a primitive version of Scoring.
  Scoring *scoring = nullptr;
  Scoring scoring_host;
  scoring_host.hits               = 0;
  scoring_host.secondaries        = 0;
  scoring_host.totalEnergyDeposit = 0;
  COPCORE_CUDA_CHECK(cudaMalloc(&scoring, sizeof(Scoring)));
  COPCORE_CUDA_CHECK(cudaMemcpy(scoring, &scoring_host, sizeof(Scoring), cudaMemcpyHostToDevice));

  // Initializing primary particles.
  constexpr int initThreads = 32;
  int initBlocks            = (particles + initThreads - 1) / initThreads;
  InitPrimaries<<<initBlocks, initThreads>>>(allTracks, slotManager, currentlyActive, particles, energy, world_dev);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  std::cout << "INFO: running with field Bz = " << BzFieldValue / copcore::units::tesla << " T";
  std::cout << std::endl;

  constexpr int maxBlocks       = 1024;
  constexpr int stepThreads     = 32;
  constexpr int relocateThreads = 32;
  int stepBlocks, relocateBlocks;

  vecgeom::Stopwatch timer;
  timer.Start();

  int inFlight = particles;
  int iterNo   = 0;

  do {
    stepBlocks = (inFlight + stepThreads - 1) / stepThreads;
    stepBlocks = std::min(stepBlocks, maxBlocks);

    relocateBlocks = std::min(inFlight, maxBlocks);

    // Call the kernel to compute the physics and geomtry step limit, transport
    // the particle while applying the continuous effects and maybe a discrete
    // process that could generate secondaries.
    PerformStep<<<stepBlocks, stepThreads>>>(allTracks, slotManager, currentlyActive, nextActive, relocateQueue,
                                             scoring);

    // Call the kernel to relocate particles that left their current volume,
    // either entering a new daughter or moving out of the current volume.
    RelocateToNextVolume<<<relocateBlocks, relocateThreads>>>(allTracks, relocateQueue);

    // Call the kernel to finish the current step, clear the queues, and return the number of particles in flight.
    FinishStep<<<1, 1>>>(currentlyActive, nextActive, relocateQueue, inFlight_dev);

    // Copy number of particles in flight and Scoring for output. Also synchronizes with the device.
    COPCORE_CUDA_CHECK(cudaMemcpy(&scoring_host, scoring, sizeof(Scoring), cudaMemcpyDeviceToHost));
    COPCORE_CUDA_CHECK(cudaMemcpy(&inFlight, inFlight_dev, sizeof(int), cudaMemcpyDeviceToHost));

    std::swap(currentlyActive, nextActive);

    std::cout << std::fixed << std::setprecision(4) << std::setfill(' ');
    std::cout << "iter " << std::setw(4) << iterNo << " -- tracks in flight: " << std::setw(5) << inFlight
              << " energy deposition: " << std::setw(10) << scoring_host.totalEnergyDeposit.load() / copcore::units::GeV
              << " number of secondaries: " << std::setw(5) << scoring_host.secondaries.load()
              << " number of hits: " << std::setw(4) << scoring_host.hits.load();
    std::cout << std::endl;

    iterNo++;
  } while (inFlight > 0 && iterNo < 1000);

  auto time_cpu = timer.Stop();
  std::cout << "Run time: " << time_cpu << "\n";

  FreeG4HepEm(state);
}
