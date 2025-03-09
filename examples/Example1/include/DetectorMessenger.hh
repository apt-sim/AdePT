// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef DETECTORMESSENGER_H
#define DETECTORMESSENGER_H

#include "G4UImessenger.hh"

class DetectorConstruction;
class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWith3VectorAndUnit;

/**
 * @brief Detector messenger.
 *
 * Provides UI commands to setup detector and readout geometry (prior to
 * initialization). Radius, length, and material of the detector, as well as
 * segmentation of the readout geometry can be changed.
 *
 */

class DetectorMessenger : public G4UImessenger {
public:
  DetectorMessenger(DetectorConstruction *);
  ~DetectorMessenger();

  /// Invokes appropriate methods based on the typed command
  virtual void SetNewValue(G4UIcommand *, G4String) final;

private:
  /// Detector construction to setup
  DetectorConstruction *fDetector = nullptr;
  /// Command to set the directory for detector settings /example22/detector
  G4UIdirectory *fDir = nullptr;
  /// Command printing current settings
  G4UIcmdWithoutParameter *fPrintCmd = nullptr;

  G4UIcmdWithAString *fFileNameCmd = nullptr;

  G4UIcmdWith3VectorAndUnit *fFieldCmd  = nullptr;
  G4UIcmdWithAString *fFieldFileNameCmd = nullptr;
};

#endif
