// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include "ConstantCovfieField.hh"

#include <G4SystemOfUnits.hh>

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>

int main(int argc, char **argv)
{
  if (argc != 2 && argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <output.cvf> [Bx By Bz in tesla]\n";
    return 2;
  }

  std::array<float, 3> fieldValue{{0.f, 0.f, static_cast<float>(tesla)}};
  if (argc == 5) {
    fieldValue = {{static_cast<float>(std::atof(argv[2]) * tesla), static_cast<float>(std::atof(argv[3]) * tesla),
                   static_cast<float>(std::atof(argv[4]) * tesla)}};
  }

  std::ofstream output(argv[1], std::ofstream::binary);
  if (!output.good()) {
    std::cerr << "Failed to open " << argv[1] << " for writing\n";
    return 1;
  }

  adept_test::WriteConstantCovfieField(output, fieldValue);
  return 0;
}
