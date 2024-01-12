// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

//
/// \file eventgenerator/HepMC3/HepMCEx01/src/HepMC3G4AsciiReader.cc
/// \brief Implementation of the HepMC3G4AsciiReader class
//
//

#include "HepMC3G4AsciiReader.hh"
#include "HepMC3G4AsciiReaderMessenger.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"

#include <iostream>
#include <fstream>

std::vector<HepMC3::GenEvent *> *HepMC3G4AsciiReader::fEvents = nullptr;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4AsciiReader::HepMC3G4AsciiReader() : fFilename("xxx.dat"), fVerbose(0)
{
  fMessenger = new HepMC3G4AsciiReaderMessenger(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4AsciiReader::~HepMC3G4AsciiReader()
{
  for (auto evt : *fEvents)
    delete evt;

  delete fEvents;
  delete fAsciiInput;
  delete fMessenger;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void HepMC3G4AsciiReader::Initialize()
{
  fAsciiInput = new HepMC3::ReaderAscii(fFilename.c_str());

  fEvents               = new std::vector<HepMC3::GenEvent *>();
  HepMC3::GenEvent *evt = nullptr;

  int counter = 0;

  if (fFirstEventNumber > 0) fAsciiInput->skip(fFirstEventNumber);

  // read file
  while (!fAsciiInput->failed() && counter < fMaxNumberOfEvents) {
    evt = new HepMC3::GenEvent();
    fAsciiInput->read_event(*evt);

    if (fAsciiInput->failed()) break;

    fEvents->push_back(evt);
    counter++;
  }
  G4cout << "Read " << fFilename << " file with " << fEvents->size() << " events." << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3::GenEvent *HepMC3G4AsciiReader::GenerateHepMCEvent(int eventId)
{
  HepMC3::GenEvent *evt = (*fEvents)[(eventId % fEvents->size())];

  if (fVerbose > 0) HepMC3::Print::content(*evt);

  return evt;
}
