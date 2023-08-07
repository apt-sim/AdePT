// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

//
/// \brief Implementation of the HepMC3G4AsciiReader class
//
//

#include "HepMC3G4AsciiReader.hh"
#include "HepMC3G4AsciiReaderMessenger.hh"

#include <iostream>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4AsciiReader::HepMC3G4AsciiReader()
    : event{new HepMC3::GenEvent()}, messenger{new HepMC3G4AsciiReaderMessenger(this)}
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4AsciiReader::~HepMC3G4AsciiReader() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3::GenEvent *HepMC3G4AsciiReader::GenerateHepMCEvent(int eventId)
{
  if (eventId >= maxNumberOfEvents) return nullptr;
  if (!asciiInput) asciiInput.reset(new HepMC3::ReaderAscii(filename.c_str()));

  const auto advance = eventId - previousEventNumber;
  if (advance > 1) asciiInput->skip(advance - 1);

  event->clear();
  asciiInput->read_event(*event);

  previousEventNumber = eventId;
  if (asciiInput->failed()) return nullptr;

  if (verbose > 0) HepMC3::Print::content(*event);

  return event.get();
}
