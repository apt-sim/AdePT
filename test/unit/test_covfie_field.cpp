// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include "CovfieTestField.hh"
#include "MagneticFields.hh"

#include <G4SystemOfUnits.hh>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

std::filesystem::path MakeTempFieldPath()
{
  auto path = std::filesystem::temp_directory_path();
  path /= "adept_covfie_field_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(&path)) + ".cvf";
  return path;
}

} // namespace

int main()
{
  // Use an exactly representable internal field value so this checks the wrapper
  // and clamping behavior without conflating it with float interpolation roundoff.
  const G4ThreeVector expectedField(0.0, 0.0, 1.0 / 1024.0);
  const auto fieldPath = MakeTempFieldPath();
  {
    std::ofstream output(fieldPath, std::ofstream::binary);
    if (!output.good()) {
      std::cerr << "Failed to create temporary field file: " << fieldPath << "\n";
      return 1;
    }
    adept_test::WriteConstantCovfieField(
        output, std::array<float, 3>{{static_cast<float>(expectedField.x()), static_cast<float>(expectedField.y()),
                                      static_cast<float>(expectedField.z())}});
  }

  CovfieField field(fieldPath.string());
  UniformField uniformField(expectedField);

  G4double points[2][4] = {{0.0, 0.0, 0.0, 0.0}, {1.e9, -1.e9, 1.e9, 0.0}};
  for (auto &point : points) {
    G4double covfieField[3]  = {-1.0, -1.0, -1.0};
    G4double uniformValue[3] = {-2.0, -2.0, -2.0};
    field.GetFieldValue(point, covfieField);
    uniformField.GetFieldValue(point, uniformValue);

    for (int component = 0; component < 3; ++component) {
      if (covfieField[component] != uniformValue[component]) {
        std::cerr << "CovfieField and UniformField differ at component " << component
                  << ": Covfie=" << covfieField[component] << ", uniform=" << uniformValue[component] << "\n";
        std::filesystem::remove(fieldPath);
        return 1;
      }
    }
  }

  std::filesystem::remove(fieldPath);

  // Independent component gradients on unequal grid dimensions test each axis and its own clamp limits.
  const float fieldScale = 1.f / 1024.f;
  {
    std::ofstream output(fieldPath, std::ofstream::binary);
    adept_test::WriteGradientCovfieField(output, fieldScale);
  }
  CovfieField gradient(fieldPath.string());
  std::filesystem::remove(fieldPath);
  const double coordinates[] = {-40000., -20000., -10000., 0., 10000., 20000.};
  const double fractions[]   = {0., 0., 0., 0.5, 1., 1.};
  const double upperField[]  = {fieldScale, 2. * fieldScale, 4. * fieldScale};
  for (int axis = 0; axis < 3; ++axis) {
    for (int i = 0; i < 6; ++i) {
      G4double point[4] = {0., 0., 0., 0.};
      point[axis]       = coordinates[i];
      G4double value[3] = {-1., -1., -1.};
      gradient.GetFieldValue(point, value);
      for (int component = 0; component < 3; ++component) {
        const double expected = upperField[component] * (component == axis ? fractions[i] : 0.5);
        if (value[component] != expected) {
          std::cerr << "Gradient CovfieField on axis " << axis << " at " << coordinates[i] << ", component "
                    << component << ": " << value[component] << ", expected " << expected << "\n";
          return 1;
        }
      }
    }
  }
  return 0;
}
