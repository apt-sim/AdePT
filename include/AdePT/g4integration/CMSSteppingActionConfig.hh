// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <AdePT/transport/config/CMSSteppingActionParams.hh>

#include <string>
#include <utility>
#include <vector>

class G4Track;

/// @brief Configuration of the CMS stepping action (ADEPT_STEPPINGACTION=CMS), set by CMSSW.
/// @details CMSSW applies its stepping action and track filter to the replayed GPU steps and erases (and deletes) the
/// GPU-born secondaries that its filter kills, although they are already transported on the GPU. The GPU therefore
/// applies the same kills and gamma Russian roulette, so this configuration must mirror the CMSSW one.
/// Regions that are missing from the geometry are ignored, except for the dead regions.
struct CMSSteppingActionConfig {
  /// Decides on the host, for each track handed over to the GPU, whether gamma Russian roulette may be applied to the
  /// gammas created by the track and its descendants in the restricted regions.
  using GammaRussianRouletteApplicability = bool (*)(G4Track const &);

  CMSSteppingActionParams params;             ///< cuts copied to the GPU
  std::vector<std::string> deadRegions;       ///< all tracks are killed in these regions
  std::vector<std::string> lowDensityRegions; ///< see CMSSteppingActionParams::criticalEnergyForVacuum
  std::vector<std::pair<std::string, double>> gammaRussianRoulette; ///< survival probability per region
  std::vector<std::string> gammaRussianRouletteRestrictedRegions;   ///< roulette only for tracks accepted below
  GammaRussianRouletteApplicability gammaRussianRouletteApplicability{nullptr};
};
