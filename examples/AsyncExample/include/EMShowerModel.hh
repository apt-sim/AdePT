// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
#ifndef EMSHOWERMODEL_HH
#define EMSHOWERMODEL_HH

#include "G4VFastSimulationModel.hh"
#include <AdePT/copcore/SystemOfUnits.h>

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmState.hh>
#include <G4HepEmStateInit.hh>

#include <memory>
#include <unordered_map>

class EMShowerMessenger;
class G4FastSimHitMaker;
class G4VPhysicalVolume;
class AdeptIntegration;

/**
 * @brief Example fast simulation model for EM showers.
 *
 * Parametrisation of electrons, positrons, and gammas. It is triggered if those
 * particles enter the detector so that there is sufficient length for the
 * shower development (max depth, controlled by the UI command).
 *
 * Using AdePT
 */

class EMShowerModel : public G4VFastSimulationModel {
public:
  EMShowerModel(G4String, G4Region *, std::shared_ptr<AdeptIntegration>);
  EMShowerModel(G4String);
  ~EMShowerModel();

  /// There are no kinematics constraints. True is returned.
  G4bool ModelTrigger(const G4FastTrack &) final override;
  /// Model is applicable to electrons, positrons, and photons.
  G4bool IsApplicable(const G4ParticleDefinition &) final override;

  /// Take particle out of the full simulation (kill it at the entrance
  /// depositing all the energy). Simulate the full shower using AdePT library.
  void DoIt(const G4FastTrack &, G4FastStep &) final override;

  void Flush() final override;

  /// Print current settings.
  void Print() const;

  /// Set verbosity for integration
  void SetVerbosity(int verbosity) { fVerbosity = verbosity; }

private:
  /// Messenger for configuration
  std::unique_ptr<EMShowerMessenger> fMessenger;

  /// AdePT integration (shared with all workers)
  std::shared_ptr<AdeptIntegration> fAdept;

  /// Verbosity
  int fVerbosity{0};

  /// @brief Counts number of tracks passed to AdePT
  unsigned int fTrackCounter{0};
  /// @brief Save last event ID to determine when new event starts
  int fLastEventId{-1};
  /// @brief Cycle number (injection/flush) for each event
  unsigned short fCycleNumber{0};
};
#endif /* EMSHOWERMODEL_HH */