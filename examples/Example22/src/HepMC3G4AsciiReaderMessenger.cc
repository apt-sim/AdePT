// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

//
/// \file eventgenerator/HepMC3/HepMCEx01/src/HepMC3G4AsciiReaderMessenger.cc
/// \brief Implementation of the HepMC3G4AsciiReaderMessenger class
//
//
#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "HepMC3G4AsciiReaderMessenger.hh"
#include "HepMC3G4AsciiReader.hh"

#include "G4Threading.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4AsciiReaderMessenger::HepMC3G4AsciiReaderMessenger(HepMC3G4AsciiReader *agen) : gen(agen)
{
  dir = new G4UIdirectory("/generator/hepmcAscii/");
  dir->SetGuidance("Reading HepMC event from an Ascii file");

  verbose = new G4UIcmdWithAnInteger("/generator/hepmcAscii/verbose", this);
  verbose->SetGuidance("Set verbose level");
  verbose->SetParameterName("verboseLevel", false, false);
  verbose->SetRange("verboseLevel>=0 && verboseLevel<=1");

  maxevent = new G4UIcmdWithAnInteger("/generator/hepmcAscii/maxevents", this);
  maxevent->SetGuidance("Set maximum number of events to be read");
  maxevent->SetParameterName("maxEvents", true);

  firstevent = new G4UIcmdWithAnInteger("/generator/hepmcAscii/firstevent", this);
  firstevent->SetGuidance("Set first event from the file");
  firstevent->SetParameterName("firstEvent", true);

  open = new G4UIcmdWithAString("/generator/hepmcAscii/open", this);
  open->SetGuidance("(re)open data file (HepMC Ascii format)");
  open->SetParameterName("input ascii file", true, true);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4AsciiReaderMessenger::~HepMC3G4AsciiReaderMessenger()
{
  delete verbose;
  delete firstevent;
  delete maxevent;
  delete open;

  delete dir;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void HepMC3G4AsciiReaderMessenger::SetNewValue(G4UIcommand *command, G4String newValues)
{
  if (command == verbose) {
    int level = verbose->GetNewIntValue(newValues);
    gen->SetVerboseLevel(level);
  } else if (command == open) {
    gen->SetFileName(newValues);
    if (G4Threading::G4GetThreadId() == -1) gen->Initialize();
  } else if (command == maxevent) {
    int maxe = maxevent->GetNewIntValue(newValues);
    gen->SetMaxNumberOfEvents(maxe);
  } else if (command == firstevent) {
    int firste = firstevent->GetNewIntValue(newValues);
    gen->SetFirstEventNumber(firste);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
G4String HepMC3G4AsciiReaderMessenger::GetCurrentValue(G4UIcommand *command)
{
  G4String cv;

  if (command == verbose) {
    cv = verbose->ConvertToString(gen->GetVerboseLevel());
  } else if (command == open) {
    cv = gen->GetFileName();
  }
  return cv;
}
