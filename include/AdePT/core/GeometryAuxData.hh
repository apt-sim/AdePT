// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_GEOMETRY_AUX_DATA_HH
#define ADEPT_GEOMETRY_AUX_DATA_HH

#include <VecGeom/navigation/NavigationState.h>

#include <unordered_map>
#include <vector>

namespace adeptint {

/// @brief Auxiliary logical volume data. This stores in the same structure the material-cuts couple index,
/// the sensitive volume handler index and the flag if the region is active for AdePT.
struct VolAuxData {
  int fSensIndex{-1};   ///< index of handler for sensitive volumes (-1 means non-sensitive)
  int fMCIndex{0};      ///< material-cut couple index in G4HepEm
  int fGPUregionId{-1}; ///< GPU region index, corresponds to G4Region.instanceID if tracked on GPU, -1 otherwise
#if defined(ADEPT_STEPACTION_TYPE) && (ADEPT_STEPACTION_TYPE == 3)
  bool fAtlasPhotonRussianRoulette{false}; ///< ATLAS-only flag: apply photon Russian Roulette to gammas born here
#endif
};

/// @brief Structure holding the arrays of auxiliary volume data on host and device.
struct VolAuxArray {
  int fNumVolumes{0};
  VolAuxData *fAuxData{nullptr};     ///< array of auxiliary volume data on host
  VolAuxData *fAuxData_dev{nullptr}; ///< array of auxiliary volume data on device

  static VolAuxArray &GetInstance()
  {
    static VolAuxArray theAuxArray;
    return theAuxArray;
  }
};

/// @brief Root Volume data of a Woodcock tracking region: Navigation index + G4HepEm material cut couple index.
struct WDTRoot {
  vecgeom::NavigationState root; ///< NavState of the root placed volume
  int hepemIMC;                  ///< G4HepEm mat-cut index for this root
};

/// @brief Compact Woodcock region header.
struct WDTRegion {
  int offset;    ///< first index in roots[]
  int count;     ///< number of roots for this region
  float ekinMin; ///< kinetic energy threshold
};

/// @brief Device view pointing to the uploaded Woodcock data.
struct WDTDeviceView {
  const WDTRoot *roots;     ///< [nRoots]
  const WDTRegion *regions; ///< [nRegions] (only WDT-enabled regions)
  const int *regionToWDT;   ///< [regionToWDTLen], regionId -> bucket (index into regions[]) or -1
  int nRoots;
  int nRegions;
  unsigned short maxIter; ///< maximum number of Woodcock iterations
};

/// @brief Sparse host-side Woodcock collection built during geometry traversal.
struct WDTHostRaw {
  std::unordered_map<int, std::vector<int>> regionToRootIndices; ///< regionId -> list of indices into roots
  std::vector<WDTRoot> roots;                                    ///< one entry per root placed volume
  float ekinMin{0.f};                                            ///< global Woodcock minimum kinetic energy
};

/// @brief Compact, upload-ready representation of Woodcock data.
struct WDTHostPacked {
  std::vector<WDTRoot> roots;     ///< packed per-region contiguous
  std::vector<WDTRegion> regions; ///< one per WDT region
  std::vector<int> regionToWDT;   ///< dense by regionId (size = number of G4 regions)
};

/// @brief Owned device buffers to manage lifetime of Woodcock tracking data.
struct WDTDeviceBuffers {
  WDTRoot *d_roots     = nullptr;
  WDTRegion *d_regions = nullptr;
  int *d_map           = nullptr;
};

} // namespace adeptint

#endif
