// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

//
/// \file eventgenerator/HepMC3/HepMCEx01/include/HepMC3G4AsciiReader.hh
/// \brief Definition of the HepMC3G4AsciiReader class
//
//

#ifndef HEPMC3_G4_ASCII_READER_H
#define HEPMC3_G4_ASCII_READER_H

#include "HepMC3G4Interface.hh"
#include "HepMC3/ReaderAscii.h"
#include "HepMC3/Print.h"

#include <climits>

class HepMC3G4AsciiReaderMessenger;

class HepMC3G4AsciiReader : public HepMC3G4Interface {
protected:
  G4String filename;
  int firstEventNumber  = 0;
  int maxNumberOfEvents = INT_MAX;
  HepMC3::ReaderAscii *asciiInput;

  static std::vector<HepMC3::GenEvent *> *fEvents;

  G4int verbose;
  HepMC3G4AsciiReaderMessenger *messenger;

  virtual HepMC3::GenEvent *GenerateHepMCEvent(int eventId);

public:
  HepMC3G4AsciiReader();
  ~HepMC3G4AsciiReader();

  // set/get methods
  void SetFileName(G4String name);
  G4String GetFileName() const;

  void SetVerboseLevel(G4int i);
  G4int GetVerboseLevel() const;

  void SetMaxNumberOfEvents(G4int i);
  G4int GetMaxNumberOfEvents() const;

  void SetFirstEventNumber(G4int i);
  G4int GetFirstEventNumber() const;

  // methods...
  void Initialize();
};

// ====================================================================
// inline functions
// ====================================================================

inline void HepMC3G4AsciiReader::SetFileName(G4String name)
{
  filename = name;
}

inline G4String HepMC3G4AsciiReader::GetFileName() const
{
  return filename;
}

inline void HepMC3G4AsciiReader::SetVerboseLevel(G4int i)
{
  verbose = i;
}

inline G4int HepMC3G4AsciiReader::GetVerboseLevel() const
{
  return verbose;
}

inline void HepMC3G4AsciiReader::SetMaxNumberOfEvents(G4int i)
{
  maxNumberOfEvents = i;
}

inline G4int HepMC3G4AsciiReader::GetMaxNumberOfEvents() const
{
  return maxNumberOfEvents;
}

inline void HepMC3G4AsciiReader::SetFirstEventNumber(G4int i)
{
  firstEventNumber = i;
}

inline G4int HepMC3G4AsciiReader::GetFirstEventNumber() const
{
  return firstEventNumber;
}

#endif
