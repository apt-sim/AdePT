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
HepMC3G4AsciiReader::HepMC3G4AsciiReader() : filename("xxx.dat"), verbose(0)
{
  messenger = new HepMC3G4AsciiReaderMessenger(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4AsciiReader::~HepMC3G4AsciiReader()
{
  for (auto evt : *fEvents)
    delete evt;

  delete fEvents;
  delete asciiInput;
  delete messenger;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void HepMC3G4AsciiReader::Initialize()
{
  asciiInput = new HepMC3::ReaderAscii(filename.c_str());

  fEvents               = new std::vector<HepMC3::GenEvent *>();
  HepMC3::GenEvent *evt = nullptr;

  int counter = 0;

  if (firstEventNumber > 0) asciiInput->skip(firstEventNumber);

  // read file
  while (!asciiInput->failed() && counter < maxNumberOfEvents) {
    evt = new HepMC3::GenEvent();
    asciiInput->read_event(*evt);

    if (asciiInput->failed()) break;

    fEvents->push_back(evt);
    counter++;
  }
  G4cout << "Read " << filename << " file with " << fEvents->size() << " events." << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3::GenEvent *HepMC3G4AsciiReader::GenerateHepMCEvent(int eventId)
{
  HepMC3::GenEvent *evt = (*fEvents)[(eventId % fEvents->size())];

  if (verbose > 0) HepMC3::Print::content(*evt);

  return evt;
}
