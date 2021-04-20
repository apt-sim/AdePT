// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE7_GEOMETRY_H
#define EXAMPLE7_GEOMETRY_H

#include <CopCore/SystemOfUnits.h>

// This basically copies the TestEm3 example's way of constructing the test geometry
// Yes, global, but fine for a dumb example where we just need a couple of constants common
// in two programs...
const char *WorldMaterial    = "G4_Galactic";
const char *GapMaterial      = "G4_Pb";
const char *AbsorberMaterial = "G4_lAr";

constexpr double ProductionCut = 0.7 * copcore::units::mm;

constexpr double fCalorSizeYZ       = 40 * copcore::units::cm;
constexpr int fNbOfLayers           = 50;
constexpr int fNbOfAbsorbers        = 2;
constexpr double GapThickness      = 2.3 * copcore::units::mm;
constexpr double AbsorberThickness = 5.7 * copcore::units::mm;
constexpr double fAbsorThickness[fNbOfAbsorbers+1] = {0.0, GapThickness, AbsorberThickness};

constexpr double fLayerThickness = GapThickness + AbsorberThickness;
constexpr double fCalorThickness = fNbOfLayers * fLayerThickness;

constexpr double fWorldSizeX  = 1.2 * fCalorThickness;
constexpr double fWorldSizeYZ = 1.2 * fCalorSizeYZ;

#endif