// SPDX-FileCopyrightText: 2024 CERN
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
HepMC3G4AsciiReader::HepMC3G4AsciiReader() : fMessenger{new HepMC3G4AsciiReaderMessenger(this)} {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4AsciiReader::~HepMC3G4AsciiReader() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3::GenEvent *HepMC3G4AsciiReader::GenerateHepMCEvent(int eventId)
{
  if (eventId >= fMaxNumberOfEvents) return nullptr;
  if (fEvents.empty()) Read();
  if (eventId >= static_cast<int>(fEvents.size())) return nullptr;

  HepMC3::GenEvent *evt = &fEvents[eventId];
  if (fVerbose > 0) HepMC3::Print::content(*evt);

  return evt;
}

void HepMC3G4AsciiReader::Read()
{
  fEvents.clear();
  HepMC3::ReaderAscii asciiInput{fFilename.c_str()};
  if (asciiInput.failed()) throw std::runtime_error{"Failed to read from " + fFilename};

  if (fMaxNumberOfEvents != INT_MAX) fEvents.reserve(fMaxNumberOfEvents);
  // The function seems to use a weird convention. skip(0) skips one event.
  if (fFirstEventNumber > 0) asciiInput.skip(fFirstEventNumber - 1);

  for (int i = 0; i < fMaxNumberOfEvents; ++i) {
    fEvents.emplace_back();
    auto &event = fEvents.back();
    asciiInput.read_event(event);

    if (asciiInput.failed()) {
      if (fMaxNumberOfEvents != INT_MAX) {
        std::cerr << __FILE__ << ':' << __LINE__ << " Read " << i << " events whereas " << fMaxNumberOfEvents
                  << " were requested.\n";
      }
      break;
    }
  }
}
