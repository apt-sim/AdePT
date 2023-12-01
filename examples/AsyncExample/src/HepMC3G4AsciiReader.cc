// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

//
/// \brief Implementation of the HepMC3G4AsciiReader class
//
//

#include "HepMC3G4AsciiReader.hh"
#include "HepMC3G4AsciiReaderMessenger.hh"

#include "HepMC3/ReaderAscii.h"

#include <iostream>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4AsciiReader::HepMC3G4AsciiReader() : messenger{new HepMC3G4AsciiReaderMessenger(this)} {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4AsciiReader::~HepMC3G4AsciiReader() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3::GenEvent *HepMC3G4AsciiReader::GenerateHepMCEvent(int eventId)
{
  if (eventId >= maxNumberOfEvents) return nullptr;
  eventId += firstEventNumber;

  if (eventId >= static_cast<int>(events.size())) {
    HepMC3::ReaderAscii asciiInput{filename.c_str()};
    if (asciiInput.failed()) throw std::runtime_error{"Failed to read from " + filename};

    events.resize(maxNumberOfEvents);
    asciiInput.skip(firstEventNumber);

    for (int i = 0; i < maxNumberOfEvents; ++i) {
      auto &event = events[i];
      event.clear();
      asciiInput.read_event(event);

      if (asciiInput.failed()) {
        events.resize(i);
        if (maxNumberOfEvents != INT_MAX) {
          std::cerr << __FILE__ << ':' << __LINE__ << " Read " << i << " events whereas " << maxNumberOfEvents
                    << " were requested.\n";
        }
        break;
      }
    }
  }

  if (eventId >= static_cast<int>(events.size())) return nullptr;

  if (verbose > 0) HepMC3::Print::content(events[eventId]);

  return &events[eventId];
}
