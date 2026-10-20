// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <AdePT/transport/support/SystemOfUnits.h>

#include <limits>

/// @brief Cuts of the CMS stepping action (ADEPT_STEPPINGACTION=CMS), copied to the GPU.
/// @details They mirror the CMSSW stepping action and track filter (CMSSW parameter names in brackets) and are set by
/// CMSSW through AdePTConfiguration::SetCMSSteppingActionConfig. The cuts introduced for the track filter default to
/// not applying, the older ones keep their previous defaults.
struct CMSSteppingActionParams {
  /// Time cut for tracks with |z| < maxZCentral and for all new secondaries [MaxTrackTime]
  double maxTrackTime{std::numeric_limits<double>::infinity()};
  /// Time cut for tracks with |z| >= maxZCentral, where no other cut is applied [MaxTrackTimeForward]
  double maxTrackTimeForward{2000.0 * copcore::units::nanosecond};
  double maxZCentral{50.0 * copcore::units::meter}; ///< [MaxZCentralCMS]
  /// e+- below this energy are killed in materials with density <= criticalDensity, and no secondaries up to this
  /// energy are created in low-density regions [CriticalEnergyForVacuum]
  double criticalEnergyForVacuum{2.0 * copcore::units::MeV};
  double criticalDensity{1e-15 * (copcore::units::g / copcore::units::cm3)}; ///< [CriticalDensity]
};
