// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

//
/// \file eventgenerator/HepMC3/HepMCEx01/include/HepMC3G4AsciiReaderMessenger.hh
/// \brief Definition of the HepMC3G4AsciiReaderMessenger class
//
//

#ifndef HEPMC3_G4_ASCII_READER_MESSENGER_H
#define HEPMC3_G4_ASCII_READER_MESSENGER_H

#include "G4UImessenger.hh"

class HepMC3G4AsciiReader;
class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;

class HepMC3G4AsciiReaderMessenger : public G4UImessenger {
public:
  HepMC3G4AsciiReaderMessenger(HepMC3G4AsciiReader* agen);
  ~HepMC3G4AsciiReaderMessenger();

  void SetNewValue(G4UIcommand* command, G4String newValues);
  G4String GetCurrentValue(G4UIcommand* command);

private:
  HepMC3G4AsciiReader* gen;

  G4UIdirectory* dir;
  G4UIcmdWithAnInteger* verbose;
  G4UIcmdWithAnInteger* maxevent;
  G4UIcmdWithAnInteger* firstevent;
  G4UIcmdWithAString* open;

};

#endif
