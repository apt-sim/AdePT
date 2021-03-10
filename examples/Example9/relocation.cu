// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example9.cuh"

#include <AdePT/MParray.h>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavStateIndex.h>
#include <VecGeom/volumes/PlacedVolume.h>

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

    vecgeom::NavStateIndex &state = currentTrack.currentState;

    vecgeom::VPlacedVolume const *currentVolume;
    vecgeom::Precision localCoordinates[3];

    // The first lane removes all volumes from the state that were left, and
    // stores it in the variable currentVolume. During the process, it also
    // transforms the point to local coordinates, eventually stored in the
    // variable localCoordinates.
    if (lane == 0) {
      // Push the point inside the next volume.
      vecgeom::Vector3D<vecgeom::Precision> pushed = currentTrack.pos + 1.E-6 * currentTrack.dir;

      // Calculate local point from global point.
      vecgeom::Transformation3D m;
      state.TopMatrix(m);
      vecgeom::Vector3D<vecgeom::Precision> localPoint = m.Transform(pushed);

      currentVolume = state.Top();

      // Remove all volumes that were left.
      while (currentVolume && (currentVolume->IsAssembly() || !currentVolume->UnplacedContains(localPoint))) {
        state.Pop();
        localPoint    = currentVolume->GetTransformation()->InverseTransform(localPoint);
        currentVolume = state.Top();
      }

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
