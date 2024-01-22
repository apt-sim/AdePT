// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author:   J. Apostolakis,  29 Nov 2021

#ifndef FIELD_CONSTANTS_H__
#define FIELD_CONSTANTS_H__

namespace fieldConstants {

  static constexpr double kB2C = -0.299792458 * (copcore::units::GeV / (copcore::units::tesla * copcore::units::meter));
   // 
  static constexpr float  deltaIntersection = 1.0e-4 * copcore::units::millimeter;
   // Accuracy required for intersection of curved trajectory with surface(s)

  static constexpr float  gEpsilonDeflect = 1.E-2 * copcore::units::cm;  // Allowable deflection during an integration step
       // The difference between the endpoint and the projection of the straight-line path after such a step
       // Used to ensure the accuracy of (intersecting with volumes) the curved path is well approximated by chords.
};

#endif
