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
#ifndef SENSITIVEDETECTOR_HH
#define SENSITIVEDETECTOR_HH

#include <unordered_map>
#include "SimpleHit.hh"

#include "G4VFastSimSensitiveDetector.hh"
#include "G4VSensitiveDetector.hh"

class G4HCofThisEvent;
class G4TouchableHistory;

/**
 * @brief Sensitive detector.
 *
 * Describes how to store the energy deposited within the detector.
 * It derives from two classes: G4VSensitiveDetector and
 * G4VFastSimSensitiveDetector. Addition of G4VFastSimSensitiveDetector is
 * necessary in order to handle the energy deposits from the fast simulation.
 *
 * Two ProcessHits() methods are introduced to handle energy deposited from full
 * (detailed) simulation, and from fast simulation. The common part is handled
 * by RetrieveAdnSetupHit() method.
 *
 */
class G4VPhysicalVolume;

class SensitiveDetector : public G4VSensitiveDetector, public G4VFastSimSensitiveDetector {
public:
  SensitiveDetector(G4String aName);
  SensitiveDetector(G4String aName, std::set<const G4VPhysicalVolume*> *aSensitivePhysicalVolumes);
  virtual ~SensitiveDetector();
  /// Create hit collection
  virtual void Initialize(G4HCofThisEvent *HCE) final;
  /// Process energy deposit from the full simulation.
  virtual G4bool ProcessHits(G4Step *aStep, G4TouchableHistory *aROhist) final;
  /// Process energy deposit from the fast simulation.
  virtual G4bool ProcessHits(const G4FastHit *aHit, const G4FastTrack *aTrack, G4TouchableHistory *aROhist) final;

  virtual G4bool ProcessHits(int hitID, double energy) final;

  SimpleHit *RetrieveAndSetupHit(G4TouchableHistory *aTouchable);

  std::unordered_map<std::string, int> fSensitive_volume_index;
  std::unordered_map<size_t, size_t> *fScoringMap;

private:
  /// Collection of hits
  SimpleHitsCollection *fHitsCollection = nullptr;
  /// ID of collection of hits
  G4int fHitCollectionID = -1;
  /// Number of sensitive detectors
  G4int fNumSensitive{0};
  /// Physical Volumes where we want to score
  std::set<const G4VPhysicalVolume*> *fSensitivePhysicalVolumes;
};

#endif /* SENSITIVEDETECTOR_HH */
