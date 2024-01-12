// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "geant4_mock.h"

#include <AdePT/base/ArgParser.h>

#include <G4GDMLParser.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4ProductionCutsTable.hh>
#include <Randomize.hh>

#include <G4HepEmData.hh>
#include <G4HepEmState.hh>
#include <G4HepEmStateInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmDataJsonIO.hh>

int main(int argc, char* argv[])
{
  // Only outputs are the data file(s)
  // Separate for now, but will want to unify
  OPTION_STRING(gdml_file, "example7.gdml");
  std::string g4hepem_file = gdml_file + ".g4hepem";

  // Build mock geant4 setup
  // - Should create geometry, regions and cuts
  G4PVPlacement* world = geant4_mock();

  // Construct and initialize the G4HepEmState data/tables
  G4HepEmState outState;
  InitG4HepEmState(&outState);

  // Persist data
  // GDML + G4HepEmData auxiliary info for connection
  G4GDMLParser gdmlParser;
  gdmlParser.SetOutputFileOverwrite(true);
  gdmlParser.SetRegionExport(true);

  // To connect up VecGeom logical volumes and G4HepEmData material cuts couples/data we
  // - Store a volume auxiliary whose value is the index of the Geant4 MCC
  // - Store the G4HepEMData output file name as a userinfo auxiliary
  auto* lvStore = G4LogicalVolumeStore::GetInstance();
  for (const G4LogicalVolume* lv : *lvStore) {
    G4GDMLAuxStructType aux{"g4matcutcouple", std::to_string(lv->GetMaterialCutsCouple()->GetIndex()), "", nullptr};
    gdmlParser.AddVolumeAuxiliary(aux, lv);
  }

  G4GDMLAuxStructType g4hepemAux{"g4hepemfile", g4hepem_file, "", nullptr};
  gdmlParser.AddAuxiliary(g4hepemAux);
  gdmlParser.Write(gdml_file, world);

  // Persist G4HepEm
  {
    std::ofstream jsonOS{ g4hepem_file.c_str() };
    if(!G4HepEmStateToJson(jsonOS, &outState))
    {
      std::cerr << "Failed to write G4HepEMState to " << g4hepem_file
                << std::endl;
      jsonOS.close();
      return 1;
    }
  }

  return 0;
}