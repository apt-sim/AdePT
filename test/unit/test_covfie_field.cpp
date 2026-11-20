// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include "ConstantCovfieField.hh"
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

  // Field rising along x from 0 to bz over [-10000, 10000]: outside the map the value is the edge value, as on the
  // CUDA texture, not an extrapolation
  const float bz = 1.f / 1024.f;
  {
    std::ofstream output(fieldPath, std::ofstream::binary);
    adept_test::WriteXGradientCovfieField(output, bz);
  }
  CovfieField gradient(fieldPath.string());
  std::filesystem::remove(fieldPath);
  const double xs[]       = {-20000., -10000., 0., 10000., 20000.};
  const double expected[] = {0., 0., 0.5 * bz, bz, bz};
  for (int i = 0; i < 5; ++i) {
    G4double point[4] = {xs[i], 0., 0., 0.};
    G4double value[3] = {-1., -1., -1.};
    gradient.GetFieldValue(point, value);
    if (value[2] != expected[i]) {
      std::cerr << "Gradient CovfieField at x=" << xs[i] << ": Bz=" << value[2] << ", expected " << expected[i] << "\n";
      return 1;
    }
  }
  return 0;
}
