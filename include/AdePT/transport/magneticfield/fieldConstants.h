// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author:   J. Apostolakis,  29 Nov 2021

#ifndef FIELD_CONSTANTS_H__
#define FIELD_CONSTANTS_H__

namespace fieldConstants {

static constexpr double kB2C = -0.299792458 * (copcore::units::GeV / (copcore::units::tesla * copcore::units::meter));
//
static constexpr float deltaIntersection = 0.001 * copcore::units::millimeter;
// Accuracy required for intersection of curved trajectory with surface(s) CURRENTLY NOT USED IN ADEPT

static constexpr float deltaChord = 0.25 * copcore::units::millimeter;
// Allowable deflection during an integration step. The difference between the endpoint and the projection of the
// straight-line path after such a step Used to ensure the accuracy of (intersecting with volumes) the curved path is
// well approximated by chords. Although it is used slightly different, this variable describes the same error as
// deltaChord in G4, so we keep the same name

}; // namespace fieldConstants

#endif
