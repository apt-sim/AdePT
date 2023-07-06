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
  /// Command to set the directory common to all messengers in this example
  G4UIdirectory *fExampleDir = nullptr;
  /// Command to set the directory for detector settings /example17/detector
  G4UIdirectory *fDetectorDir = nullptr;
  /// Command to set the directory for AdePT integration settings /example17/adept
  G4UIdirectory *fAdeptDir = nullptr;
  // Command to set the seed for the random number generator
  G4UIcmdWithAnInteger *fSetSeed = nullptr;
  /// Command printing current settings
  G4UIcmdWithoutParameter *fPrintCmd = nullptr;

  G4UIcmdWithAString *fFileNameCmd     = nullptr;
  G4UIcmdWithAString *fRegionNameCmd   = nullptr;
  G4UIcmdWithAString *fSensVolNameCmd  = nullptr;
  G4UIcmdWithAString *fSensVolGroupCmd = nullptr;

  G4UIcmdWith3VectorAndUnit *fFieldCmd = nullptr;
  /// Activation of AdePT
  G4UIcmdWithABool *fActivationCmd = nullptr;
  /// Verbosity for AdeptIntegration (should it be here?)
  G4UIcmdWithAnInteger *fVerbosityCmd = nullptr;
  /// Buffer threshold for injecting into AdePT
  G4UIcmdWithAnInteger *fBufferThresholdCmd = nullptr;
  /// Total number of track slots for the gpu
  G4UIcmdWithADouble *fTrackSlotsCmd = nullptr;
};

#endif
