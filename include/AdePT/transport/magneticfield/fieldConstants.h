// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author:   J. Apostolakis,  29 Nov 2021

#pragma once

namespace fieldConstants {

static constexpr double kB2C = -0.299792458 * (copcore::units::GeV / (copcore::units::tesla * copcore::units::meter));
//
static constexpr float deltaIntersection = 0.001 * copcore::units::millimeter;
// Accuracy required for intersection of curved trajectory with surface(s) CURRENTLY NOT USED IN ADEPT

static constexpr float deltaChord = 0.25 * copcore::units::millimeter;
// Maximum distance between the curved path and the chord that replaces it for navigation outside geometry safety.
// As in G4's G4ChordFinder, a chord is accepted while its (estimated) distance from the curve stays below this
// value, and 0.25 mm is G4's default.
// TODO: refine boundary crossings on the curve to deltaIntersection, as G4 does.

}; // namespace fieldConstants
