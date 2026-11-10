// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <AdePT/AdePTBuildFlags.hh>

#include <AdePT/transport/state/DeviceGlobals.cuh>
#include <AdePT/transport/woodcock/WoodcockData.hh>

namespace adept::transport {

/// @brief Resolve a G4 region instance ID to its WDT bucket index, or -1 if WDT is disabled there.
__host__ __device__ __forceinline__ int GetWDTRegionIndex(const adeptint::WDTDeviceView &view, int regionId)
{
  if (regionId < 0 || view.nRegions == 0) return -1;
  return view.regionToWDT[regionId];
}

/// @brief Resolve this navigation state's WDT bucket index without checking envelope containment.
__device__ __forceinline__ int GetWDTRegionIndex(const vecgeom::NavigationState &state)
{
#ifndef ADEPT_ENABLE_WDT
  (void)state;
  return -1;
#else
  if (state.IsOutside()) return -1;
  return GetWDTRegionIndex(gWDTData, gVolAuxData[state.GetLogicalId()].fGPUregionId);
#endif
}

/// @brief Find the WDT envelope containing this navigation state within the resolved region.
/// @returns Global index into view.roots, or -1 when the region has no enclosing WDT envelope.
__host__ __device__ __forceinline__ int FindWDTRoot(const vecgeom::NavigationState &state,
                                                    const adeptint::WDTDeviceView &view, int wdtIdx)
{
  if (state.IsOutside() || wdtIdx < 0) return -1;
  const auto &region = view.regions[wdtIdx];
  for (int i = 0; i < region.count; ++i) {
    const int rootIdx = region.offset + i;
    const auto &root  = view.roots[rootIdx].root;
    if (state.GetState() == root.GetState() || state.IsDescendent(root.GetState())) return rootIdx;
  }
  return -1;
}

// Cheap region/energy eligibility check for the WDT queue. The WDT kernel verifies
// actual envelope containment and returns false positives to ordinary gamma tracking.
__device__ __forceinline__ bool ShouldUseWDT(const vecgeom::NavigationState &state, double eKin)
{
  const int wdtIdx = GetWDTRegionIndex(state);
  if (wdtIdx < 0 || eKin <= gWDTData.regions[wdtIdx].ekinMin) return false;
  return true;
}

} // namespace adept::transport
