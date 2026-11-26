// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author:   J. Apostolakis,  29 Nov 2021

#pragma once

namespace fieldConstants {

static constexpr double kB2C = -0.299792458 * (copcore::units::GeV / (copcore::units::tesla * copcore::units::meter));
//
static constexpr float deltaIntersection = 0.001 * copcore::units::millimeter;
// Accuracy of the intersection of the curved path with a boundary, as G4's deltaIntersection, whose default is
// 0.001 mm. A crossing found on a chord is refined until its estimated distance from the curve is below this value,
// or until a refinement or iteration limit of fieldPropagatorRungeKutta stops it. The crossing on the chord is then
// accepted farther away; this happens mostly at shallow crossings.

static constexpr float deltaChord = 0.25 * copcore::units::millimeter;
// Maximum distance between the curved path and the chord that replaces it for navigation outside geometry safety.
// As in G4's G4ChordFinder, a chord is accepted while its (estimated) distance from the curve stays below this
// value, and 0.25 mm is G4's default. Boundary crossings found on a chord are then refined to deltaIntersection,
// apart from the few that reach a refinement limit (see deltaIntersection).

}; // namespace fieldConstants
