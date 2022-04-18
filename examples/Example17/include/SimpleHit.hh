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
#ifndef SIMPLEHIT_HH
#define SIMPLEHIT_HH

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"

class G4AttDef;
class G4AttValue;
class G4LogicalVolume;

/**
 * @brief Hit class to store energy deposited in the sensitive detector.
 *
 * Hit class registers position and energy deposited within the sensitive
 * detector. Cell ID is stored using identifiers of readout segmentation (z,
 * phi, rho). Additionally, pointer to cell logical volume, its position and
 * rotation are saved for visualisation purposes. Time allows to filter hits in
 * visualisation. Type of hit allows to distinguish between hits originating
 * from full simulation (type 0) and fast simulation (type 1).
 *
 */

class SimpleHit : public G4VHit {
public:
  SimpleHit();
  SimpleHit(const SimpleHit &aRight);
  virtual ~SimpleHit();

  const SimpleHit &operator=(const SimpleHit &aRight);

  inline void *operator new(size_t);
  inline void operator delete(void *aHit);
  /// Retrieve atributes' names in order to allow filtering
  virtual const std::map<G4String, G4AttDef> *GetAttDefs() const;
  /// Create attributes for the visualisation.
  virtual std::vector<G4AttValue> *CreateAttValues() const;
  /// Print hit properties.
  virtual void Print();
  /// Set position
  inline void SetPos(G4ThreeVector aXYZ) { fPos = aXYZ; }
  /// Get position
  inline G4ThreeVector GetPos() const { return fPos; }
  /// Set energy
  inline void SetEdep(G4double aEdep) { fEdep = aEdep; }
  /// Add energy to previous value
  inline void AddEdep(G4double aEdep) { fEdep += aEdep; }
  /// Get energy
  inline G4double GetEdep() const { return fEdep; }
  /// Set time
  inline void SetTime(G4double aTime) { fTime = aTime; }
  /// Get time
  inline G4double GetTime() const { return fTime; }
  /// Set type (0 = full sim, 1 = fast sim)
  inline void SetType(G4int aType) { fType = aType; }
  /// Get type (0 = full sim, 1 = fast sim)
  inline G4int GetType() const { return fType; }

public:
  /// Energy deposit
  G4double fEdep = 0;
  /// Position
  G4ThreeVector fPos;
  /// Time
  G4double fTime = -1;
  /// Type: 0 = full sim, 1 = fast sim
  G4int fType = -1;
};

typedef G4THitsCollection<SimpleHit> SimpleHitsCollection;

extern G4ThreadLocal G4Allocator<SimpleHit> *SimpleHitAllocator;

inline void *SimpleHit::operator new(size_t)
{
  if (!SimpleHitAllocator) SimpleHitAllocator = new G4Allocator<SimpleHit>;
  return (void *)SimpleHitAllocator->MallocSingle();
}

inline void SimpleHit::operator delete(void *aHit)
{
  SimpleHitAllocator->FreeSingle((SimpleHit *)aHit);
}

#endif /* SIMPLEHIT_HH */
