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
#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include <VecGeom/base/Stopwatch.h>
class G4Run;
class DetectorConstruction;

/**
 * @brief Run action
 *
 * Create analysis file and define control histograms for showers in detectors.
 * Histograms are configured taking into account the dimensions of the detector
 * (and number of readout cells).
 *
 */

class RunAction : public G4UserRunAction {
public:
  /// Constructor. Defines the histograms.
  RunAction(DetectorConstruction *aDetector);
  virtual ~RunAction();

  /// Open the file for the analysis
  virtual void BeginOfRunAction(const G4Run *) final;
  /// Write and close the file
  virtual void EndOfRunAction(const G4Run *) final;

private:
  /// Pointer to detector construction to retrieve the detector dimensions to
  /// setup the histograms
  DetectorConstruction *fDetector;
  vecgeom::Stopwatch fTimer;
};

#endif /* RUNACTION_HH */
