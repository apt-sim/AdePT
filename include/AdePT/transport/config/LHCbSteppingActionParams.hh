// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <AdePT/transport/support/SystemOfUnits.h>

/// @brief Parameters of the LHCb stepping action (ADEPT_STEPPINGACTION=LHCb).
struct LHCbSteppingActionParams {
  // default parameters for world cut
  double xmin{-10000 * copcore::units::millimeter}, xmax{10000 * copcore::units::millimeter},
      ymin{-10000 * copcore::units::millimeter}, ymax{10000 * copcore::units::millimeter},
      zmin{-5000 * copcore::units::millimeter}, zmax{25000 * copcore::units::millimeter};
  // default parameters for energy tracking cut
  double ecut = 1.0 * copcore::units::MeV;
};
