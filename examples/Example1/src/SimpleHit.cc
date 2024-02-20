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
#include "SimpleHit.hh"

#include "G4VisAttributes.hh"
#include "G4Tubs.hh"
#include "G4Colour.hh"
#include "G4AttDefStore.hh"
#include "G4AttDef.hh"
#include "G4AttValue.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4VVisManager.hh"
#include "G4LogicalVolume.hh"

G4ThreadLocal G4Allocator<SimpleHit> *SimpleHitAllocator;

SimpleHit::SimpleHit() : G4VHit() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SimpleHit::~SimpleHit() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SimpleHit::SimpleHit(const SimpleHit &aRight) : G4VHit()
{
  fEdep = aRight.fEdep;
  fTime = aRight.fTime;
  fPos  = aRight.fPos;
  fType = aRight.fType;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

const SimpleHit &SimpleHit::operator=(const SimpleHit &aRight)
{
  fEdep = aRight.fEdep;
  fTime = aRight.fTime;
  fPos  = aRight.fPos;
  fType = aRight.fType;
  return *this;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

const std::map<G4String, G4AttDef> *SimpleHit::GetAttDefs() const
{
  G4bool isNew;
  std::map<G4String, G4AttDef> *store = G4AttDefStore::GetInstance("SimpleHit", isNew);
  if (isNew) {
    (*store)["HitType"] = G4AttDef("HitType", "Hit Type", "Physics", "", "G4String");
    (*store)["Energy"]  = G4AttDef("Energy", "Energy Deposited", "Physics", "G4BestUnit", "G4double");
    (*store)["Time"]    = G4AttDef("Time", "Time", "Physics", "G4BestUnit", "G4double");
    (*store)["Pos"]     = G4AttDef("Pos", "Position", "Physics", "G4BestUnit", "G4ThreeVector");
  }
  return store;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

std::vector<G4AttValue> *SimpleHit::CreateAttValues() const
{
  std::vector<G4AttValue> *values = new std::vector<G4AttValue>;
  values->push_back(G4AttValue("HitType", "HadSimpleHit", ""));
  values->push_back(G4AttValue("Energy", G4BestUnit(fEdep, "Energy"), ""));
  values->push_back(G4AttValue("Time", G4BestUnit(fTime, "Time"), ""));
  values->push_back(G4AttValue("Pos", G4BestUnit(fPos, "Length"), ""));
  return values;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SimpleHit::Print()
{
  G4cout << "\tHit " << fEdep / MeV << " MeV\n";
}
