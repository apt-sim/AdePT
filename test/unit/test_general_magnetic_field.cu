// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include "ConstantCovfieField.hh"

#include <AdePT/transport/support/Portability.hh>

#include <G4SystemOfUnits.hh>

#include <fstream>
#include <iostream>
#include <memory>

#include <AdePT/transport/magneticfield/GeneralMagneticField.cuh>
#include <AdePT/transport/magneticfield/UniformMagneticField.cuh>

#include <cstdint>
#include <filesystem>
#include <string>

namespace {

std::filesystem::path MakeTempFieldPath()
{
  auto path = std::filesystem::temp_directory_path();
  path /= "adept_general_magnetic_field_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(&path)) + ".cvf";
  return path;
}

constexpr int kNumTestPoints = 4;
constexpr int kNumComponents = 3;

__global__ void EvaluateFieldKernel(const GeneralMagneticField *generalField, UniformMagneticField uniformField,
                                    float *values)
{
  const vecgeom::Vector3D<float> positions[kNumTestPoints] = {
      {0.f, 0.f, 0.f}, {123.5f, -456.25f, 789.75f}, {9999.f, -9999.f, 100.f}, {25000.f, -25000.f, 25000.f}};

  for (int point = 0; point < kNumTestPoints; ++point) {
    const auto generalValue = generalField->Evaluate(positions[point]);
    const auto uniformValue = uniformField.Evaluate(positions[point]);

    const int offset   = point * kNumComponents * 2;
    values[offset]     = generalValue[0];
    values[offset + 1] = generalValue[1];
    values[offset + 2] = generalValue[2];
    values[offset + 3] = uniformValue[0];
    values[offset + 4] = uniformValue[1];
    values[offset + 5] = uniformValue[2];
  }
}

} // namespace

int main()
{
  // Device-side Covfie and uniform fields should agree exactly for the same
  // float field value, including the Geant4 internal value corresponding to 1 T.
  const auto expectedField = std::array<float, 3>{{0.f, 0.f, static_cast<float>(tesla)}};
  const auto fieldPath     = MakeTempFieldPath();
  {
    std::ofstream output(fieldPath, std::ofstream::binary);
    if (!output.good()) {
      std::cerr << "Failed to create temporary field file: " << fieldPath << "\n";
      return 1;
    }
    adept_test::WriteConstantCovfieField(output, expectedField);
  }

  GeneralMagneticField hostField;
  if (!hostField.InitializeFromFile(fieldPath.string())) {
    std::cerr << "GeneralMagneticField failed to initialize from " << fieldPath << "\n";
    std::filesystem::remove(fieldPath);
    return 1;
  }

  const UniformMagneticField uniformField({expectedField[0], expectedField[1], expectedField[2]});

  GeneralMagneticField *deviceField                 = nullptr;
  float *deviceValues                               = nullptr;
  float values[kNumTestPoints * kNumComponents * 2] = {};

  ADEPT_DEVICE_API_CALL(Malloc(&deviceField, sizeof(GeneralMagneticField)));
  ADEPT_DEVICE_API_CALL(
      Memcpy(deviceField, &hostField, sizeof(GeneralMagneticField), ADEPT_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
  ADEPT_DEVICE_API_CALL(Malloc(&deviceValues, sizeof(values)));

  EvaluateFieldKernel<<<1, 1>>>(deviceField, uniformField, deviceValues);
  ADEPT_DEVICE_API_CALL(DeviceSynchronize());

  ADEPT_DEVICE_API_CALL(Memcpy(values, deviceValues, sizeof(values), ADEPT_DEVICE_API_SYMBOL(MemcpyDeviceToHost)));

  ADEPT_DEVICE_API_CALL(Free(deviceValues));
  ADEPT_DEVICE_API_CALL(Free(deviceField));
  std::filesystem::remove(fieldPath);

  for (int point = 0; point < kNumTestPoints; ++point) {
    const int offset = point * kNumComponents * 2;
    for (int component = 0; component < kNumComponents; ++component) {
      const auto generalValue = values[offset + component];
      const auto uniformValue = values[offset + kNumComponents + component];
      if (generalValue != uniformValue) {
        std::cerr << "Covfie and uniform device fields differ at point " << point << ", component " << component
                  << ": Covfie=" << generalValue << ", uniform=" << uniformValue << "\n";
        return 1;
      }
    }
  }

  return 0;
}
