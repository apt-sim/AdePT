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
#ifndef PRIMARYGENERATORACTION_HH
#define PRIMARYGENERATORACTION_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "ParticleGun.hh"
#include "globals.hh"

class G4Event;
class DetectorConstruction;
class PrimaryGeneratorMessenger;

/**
 * @brief Generator of particles
 *
 * Creates single particle events using a particle gun. Particle gun can be
 * configured using UI commands /gun/.
 *
 */

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
  PrimaryGeneratorAction(DetectorConstruction *);
  virtual ~PrimaryGeneratorAction();

  void Print() const;
  void PrintPrimaries(G4Event* aEvent) const;
  void SetHepMC() {fUseHepMC = true;}
  void SetDefaultKinematic();
  void SetRndmBeam(G4double val) { fRndmBeam = val; }
  void SetRndmDirection(G4double val) { fRndmDirection = val; }
  void SetRandomizeGun(G4bool val) { fRandomizeGun = val; }
  void AddParticle(G4ParticleDefinition* val, float weight=-1, double energy=-1);
  void SetMinPhi(G4double val) { fMinPhi = val; }
  void SetMaxPhi(G4double val) { fMaxPhi = val; }
  void SetMinTheta(G4double val) { fMinTheta = val; }
  void SetMaxTheta(G4double val) { fMaxTheta = val; }
  /** @brief Checks that the user-provided weights sum to 1 or less, distributes the remaining weight
   * among the particles with undefined weight.
   */
  void ReWeight();

  void SetPrintGun(G4double val) { fPrintGun = val; }

  virtual void GeneratePrimaries(G4Event *) final;

private:
  /// Particle gun
  ParticleGun *fParticleGun;
  DetectorConstruction *fDetector;
  G4double fRndmBeam; // lateral random beam extension in fraction sizeYZ/2
  G4double fRndmDirection;
  G4double fPrintGun;

  // HepMC3 reader
  G4VPrimaryGenerator* fHepmcAscii;
  G4bool fUseHepMC;

  //Gun randomization
  bool fRandomizeGun;
  std::vector<G4ParticleDefinition*> *fParticleList;
  std::vector<float> *fParticleWeights;
  std::vector<float> *fParticleEnergies;
  bool fInitializationDone;
  G4double fMinPhi;
  G4double fMaxPhi;
  G4double fMinTheta;
  G4double fMaxTheta;

  PrimaryGeneratorMessenger *fGunMessenger;
};

#endif /* PRIMARYGENERATORACTION_HH */
