// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
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
/// \file electromagnetic/TestEm1/include/TrackingAction.hh
/// \brief Definition of the TrackingAction class
//
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef TRACKINGACTION_HH
#define TRACKINGACTION_HH

#include "G4UserTrackingAction.hh"
#include "G4Region.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class DetectorConstruction;

class TrackingAction : public G4UserTrackingAction {

public:
  TrackingAction(DetectorConstruction* aDetector);
  ~TrackingAction(){};

  virtual void PreUserTrackingAction(const G4Track *);
  virtual void PostUserTrackingAction(const G4Track *);

  void setInsideEcal(bool insideEcal){fInsideEcal = insideEcal;}
  bool getInsideEcal(){return fInsideEcal;}

  inline G4Region* getGPURegion(){return fGPURegion;}
  inline G4Region* getCurrentRegion(){return fCurrentRegion;}
  inline void setCurrentRegion(G4Region* aCurrentRegion){fCurrentRegion = aCurrentRegion;}
  inline G4VPhysicalVolume* getCurrentVolume(){return fCurrentVolume;}
  inline void setCurrentVolume(G4VPhysicalVolume* aCurrentVolume){fCurrentVolume = aCurrentVolume;}

private:
  DetectorConstruction* fDetector;
  bool fInsideEcal;
  G4Region* fCurrentRegion;
  G4VPhysicalVolume* fCurrentVolume;
  G4Region* fGPURegion;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif // TRACKINGACTION_HH
