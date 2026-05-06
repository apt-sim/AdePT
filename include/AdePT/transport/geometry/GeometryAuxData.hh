// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace adeptint {

/// @brief Auxiliary logical volume data. This stores in the same structure the material-cuts couple index,
/// the sensitive volume handler index and the flag if the region is active for AdePT.
struct VolAuxData {
  int fSensIndex{-1};   ///< index of handler for sensitive volumes (-1 means non-sensitive)
  int fMCIndex{0};      ///< material-cut couple index in G4HepEm
  int fGPUregionId{-1}; ///< GPU region index, corresponds to G4Region.instanceID if tracked on GPU, -1 otherwise
#if defined(ADEPT_STEPACTION_TYPE) && (ADEPT_STEPACTION_TYPE == 1)
  bool fCMSDeadRegion{false}; ///< CMS-only flag: tracks entering this volume are killed by the stepping action
#endif
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

} // namespace adeptint
