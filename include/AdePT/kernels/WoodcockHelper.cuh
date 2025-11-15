// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/CommonStruct.h>
#include <AdePT/core/AsyncAdePTTransportStruct.cuh>

#ifdef ASYNC_MODE
namespace AsyncAdePT {

// Helper function to decide whether a gamma should be processed via Woodcock tracking based on navigation state and
// kinetic energy
__device__ __forceinline__ bool ShouldUseWDT(const vecgeom::NavigationState &state, double eKin)
{
  // 1) region from current logical volume
  const int lvolId   = state.GetLogicalId();
  const auto &aux    = gVolAuxData[lvolId];
  const int regionId = aux.fGPUregionId;
  if (regionId < 0) return false;

  // 2) WDT mapping for this region
  const adeptint::WDTDeviceView &view = gWDTData;
  const int wdtIdx                    = view.regionToWDT[regionId];
  // wdtIdx >= 0 is a Woodcock tracking region
  if (wdtIdx < 0) return false;

  const adeptint::WDTRegion reg = view.regions[wdtIdx];

  // 3) Check for min energy
  if (eKin <= reg.ekinMin) return false;

  // Now it is 1) a GPU region 2) a Woodcock tracking region and 3) the track has enough energy:
  // mark for Woodcock tracking
  return true;
}

} // namespace AsyncAdePT
#endif
