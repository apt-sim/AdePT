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
#include "EventAction.hh"
#include "EventActionMessenger.hh"
#include "SimpleHit.hh"
//#include "Histograms.h"

#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4SystemOfUnits.hh"

#include <TProfile.h>

#include <mutex>
#include <sstream>

EventAction::EventAction() : G4UserEventAction(), fHitCollectionID(-1), fMessenger{new EventActionMessenger(this)} {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventAction::~EventAction() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::BeginOfEventAction(const G4Event *)
{
  // Prevent crashes in MT mode
  TH1::AddDirectory(false);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::EndOfEventAction(const G4Event *aEvent)
{
  if (fVerbosity == 0) return;

  const auto eventId = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();

  // Get hits collection ID (only once)
  if (fHitCollectionID == -1) {
    fHitCollectionID = G4SDManager::GetSDMpointer()->GetCollectionID("hits");
  }
  // Get hits collection
  auto hitsCollection = static_cast<SimpleHitsCollection *>(aEvent->GetHCofThisEvent()->GetHC(fHitCollectionID));

  if (hitsCollection == nullptr) {
    G4ExceptionDescription msg;
    msg << "Cannot access hitsCollection ID " << fHitCollectionID;
    G4Exception("EventAction::GetHitsCollection()", "MyCode0001", FatalException, msg);
  }

  G4double totalEnergy = 0;
  std::stringstream msg;
  auto energyPerVol = std::make_shared<TH1D>(("Event_" + std::to_string(eventId) + "_energyPerVol").c_str(),
                                             "Total Energy deposited per Volume;E / GeV", 250, 1, 501);
  std::map<std::string, double> energyPerLogicalVolume;

  for (size_t iHit = 0; iHit < hitsCollection->entries(); iHit++) {
    SimpleHit const *hit = static_cast<SimpleHit *>(hitsCollection->GetHit(iHit));
    const double hitEn   = hit->GetEdep();
    totalEnergy += hitEn;
    G4String vol_name = hit->GetPhysicalVolumeName();

    energyPerVol->Fill(hitEn / GeV);
    std::string newName = vol_name;
    if (newName.find("RPC") != std::string::npos) newName = "RPC";
    if (const auto pos = newName.find("Part"); pos != std::string::npos) newName = newName.substr(0, pos);
    if (const auto pos = newName.find("Layer"); pos != std::string::npos) newName = newName.substr(0, pos);
    if (const auto pos = newName.find("Bundle"); pos != std::string::npos) newName = newName.substr(0, pos);
    if (const auto pos = newName.find("Module"); pos != std::string::npos) newName = newName.substr(0, pos);
    if (const auto pos = newName.find("GHA"); pos != std::string::npos) newName = newName.substr(0, pos + 3);
    if (const auto pos = newName.find("HTS"); pos != std::string::npos) newName = newName.substr(0, pos + 4);
    energyPerLogicalVolume[newName] += hitEn / GeV;

    if (hitEn > 1 && fVerbosity > 1)
      msg << "EndOfEventAction " << eventId << " : id " << std::setw(5) << iHit << "  edep " << std::setprecision(2)
          << std::setw(12) << std::fixed << hitEn / MeV << " [MeV] logical " << vol_name << "\n";
  }

  msg << "EndOfEventAction " << eventId << " : Total " << std::setw(12) << totalEnergy / GeV << " GeV";

  {
    static std::mutex eventActionPrinterMutex;
    std::scoped_lock lock{eventActionPrinterMutex};
    std::cout << "\n" << msg.str() << "\n";
  }

  auto totalE = std::make_shared<TH1D>("TotalE", "Total Energy deposition per event;E / GeV", 200, 0, 5000);
  totalE->Fill(totalEnergy / GeV);
  auto energyPerLVol = std::make_shared<TProfile>(("Event_" + std::to_string(eventId) + "_energyPerLVol").c_str(),
                                                  "Total Energy deposited per Volume;Volume name;E / GeV",
                                                  energyPerLogicalVolume.size(), 0, energyPerLogicalVolume.size(), "s");
  for (const auto &[name, energy] : energyPerLogicalVolume) {
    energyPerLVol->Fill(name.c_str(), energy);
  }

  // AsyncExHistos::registerHisto(totalE);
  // AsyncExHistos::registerHisto(energyPerVol);
  // AsyncExHistos::registerHisto(energyPerLVol);
}
