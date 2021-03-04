// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "geant4_mock.h"

#include <AdePT/ArgParser.h>

#include "G4GDMLParser.hh"
#include "G4ProductionCutsTable.hh"

int main(int argc, char* argv[])
{
  // Only outputs are the data file(s)
  // Separate for now, but will want to unify
  OPTION_STRING(gdml_file, "example7.gdml");
  OPTION_STRING(g4hepem_file, "example7.g4hepem");

  // Build mock geant4 setup
  // - Should create geometry, regions and cuts
  G4PVPlacement* world = geant4_mock();

  // Dump cuts couples to check
  G4ProductionCutsTable::GetProductionCutsTable()->DumpCouples();

  // Persist data
  // Add export of regions and energy cuts to see what these
  // do and how to use. Remove pointer from exported names for
  // now to aid reabability (we know we won't have duplicated names)
  G4GDMLParser gdmlParser;
  gdmlParser.SetAddPointerToName(false);
  gdmlParser.SetOutputFileOverwrite(true);
  gdmlParser.SetRegionExport(true);
  gdmlParser.SetEnergyCutsExport(true);
  gdmlParser.Write(gdml_file, world, false);

  return 0;
}