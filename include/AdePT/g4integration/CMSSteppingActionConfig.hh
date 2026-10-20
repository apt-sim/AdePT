// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <AdePT/transport/config/CMSSteppingActionParams.hh>

#include <string>
#include <vector>

/// @brief Configuration of the CMS stepping action (ADEPT_STEPPINGACTION=CMS), set by CMSSW.
/// @details CMSSW applies its stepping action to the replayed GPU steps, so the GPU applies the same kills and this
/// configuration must mirror the CMSSW one. Regions that are missing from the geometry are rejected.
struct CMSSteppingActionConfig {
  CMSSteppingActionParams params;       ///< cuts copied to the GPU
  std::vector<std::string> deadRegions; ///< all tracks are killed in these regions
};
