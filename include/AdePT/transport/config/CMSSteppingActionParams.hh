// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <AdePT/transport/support/SystemOfUnits.h>

/// @brief Parameters of the CMS stepping action (ADEPT_STEPPINGACTION=CMS).
struct CMSSteppingActionParams {
  double tmax    = 2000.0 * copcore::units::nanosecond;               // time cut
  double zmax    = 50.0 * copcore::units::meter;                      // longitudinal envelope
  double ecut    = 2.0 * copcore::units::MeV;                         //  kinetic E for the vacuum cut
  double density = 1e-15 * (copcore::units::g / copcore::units::cm3); // density for the vacuum cut
};
