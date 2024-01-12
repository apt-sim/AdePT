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
  fDir = new G4UIdirectory("/generator/hepmcAscii/");
  fDir->SetGuidance("Reading HepMC event from an Ascii file");

  fVerbose = new G4UIcmdWithAnInteger("/generator/hepmcAscii/verbose", this);
  fVerbose->SetGuidance("Set verbose level");
  fVerbose->SetParameterName("verboseLevel", false, false);
  fVerbose->SetRange("verboseLevel>=0 && verboseLevel<=1");

  fMaxevent = new G4UIcmdWithAnInteger("/generator/hepmcAscii/maxevents", this);
  fMaxevent->SetGuidance("Set maximum number of events to be read");
  fMaxevent->SetParameterName("maxEvents", true);

  fFirstevent = new G4UIcmdWithAnInteger("/generator/hepmcAscii/firstevent", this);
  fFirstevent->SetGuidance("Set first event from the file");
  fFirstevent->SetParameterName("firstEvent", true);

  fOpen = new G4UIcmdWithAString("/generator/hepmcAscii/open", this);
  fOpen->SetGuidance("(re)open data file (HepMC Ascii format)");
  fOpen->SetParameterName("input ascii file", true, true);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4AsciiReaderMessenger::~HepMC3G4AsciiReaderMessenger()
{
  delete fVerbose;
  delete fFirstevent;
  delete fMaxevent;
  delete fOpen;

  delete fDir;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void HepMC3G4AsciiReaderMessenger::SetNewValue(G4UIcommand *command, G4String newValues)
{
  if (command == fVerbose) {
    int level = fVerbose->GetNewIntValue(newValues);
    gen->SetVerboseLevel(level);
  } else if (command == fOpen) {
    gen->SetFileName(newValues);
    if (G4Threading::G4GetThreadId() == -1) gen->Initialize();
  } else if (command == fMaxevent) {
    int maxe = fMaxevent->GetNewIntValue(newValues);
    gen->SetMaxNumberOfEvents(maxe);
  } else if (command == fFirstevent) {
    int firste = fFirstevent->GetNewIntValue(newValues);
    gen->SetFirstEventNumber(firste);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
G4String HepMC3G4AsciiReaderMessenger::GetCurrentValue(G4UIcommand *command)
{
  G4String cv;

  if (command == fVerbose) {
    cv = fVerbose->ConvertToString(gen->GetVerboseLevel());
  } else if (command == fOpen) {
    cv = gen->GetFileName();
  }
  return cv;
}
