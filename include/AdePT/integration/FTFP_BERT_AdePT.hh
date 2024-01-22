// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef FTFP_BERT_AdePT_h
#define FTFP_BERT_AdePT_h 1

#include <CLHEP/Units/SystemOfUnits.h>

#include "globals.hh"
#include "G4VModularPhysicsList.hh"
#include <AdePT/integration/AdePTTrackingManager.hh>
#include <AdePT/integration/PhysListAdePT.hh>

class FTFP_BERT_AdePT : public G4VModularPhysicsList {
public:
  FTFP_BERT_AdePT(G4int ver = 1);
  virtual ~FTFP_BERT_AdePT() = default;

  FTFP_BERT_AdePT(const FTFP_BERT_AdePT &) = delete;
  FTFP_BERT_AdePT &operator=(const FTFP_BERT_AdePT &) = delete;
  
  PhysListAdePT *GetAdePTPhysicsList(){return fAdePTPhysicsList;}

private:
  // NEVER INITIALIZED!
  PhysListAdePT *fAdePTPhysicsList{nullptr};
};

#endif
