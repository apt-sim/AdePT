// SPDX-FileCopyrightText: 2024 CERN
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
#include "ParticleGun.hh"
#include "globals.hh"

class G4Event;
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
  PrimaryGeneratorAction() : G4VUserPrimaryGeneratorAction(), fParticleGun{new ParticleGun()} {}
  virtual ~PrimaryGeneratorAction() = default;
  virtual void GeneratePrimaries(G4Event *aEvent) final { fParticleGun->GeneratePrimaries(aEvent); }

private:
  /// Particle gun
  std::unique_ptr<ParticleGun> fParticleGun;
};

#endif /* PRIMARYGENERATORACTION_HH */
