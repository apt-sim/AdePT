// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "geant4_mock.h"

#include <AdePT/ArgParser.h>

#include "G4GDMLParser.hh"

int main(int argc, char* argv[])
{
  // Only outputs are the data file(s)
  // Separate for now, but will want to unify
  OPTION_STRING(gdml_file, "example7.gdml");
  OPTION_STRING(g4hepem_file, "example7.g4hepem");

  // Build mock geant4 setup
  G4PVPlacement* world = geant4_mock();

  // Persist data
  G4GDMLParser gdmlParser;
  gdmlParser.SetAddPointerToName(false);
  gdmlParser.SetOutputFileOverwrite(true);
  gdmlParser.SetRegionExport(true);
  gdmlParser.SetEnergyCutsExport(true);
  gdmlParser.Write(gdml_file, world, false);

  return 0;
}