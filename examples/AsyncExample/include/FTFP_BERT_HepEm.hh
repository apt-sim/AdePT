// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef FTFP_BERT_HepEm_h
#define FTFP_BERT_HepEm_h 1

#include <CLHEP/Units/SystemOfUnits.h>

#include "globals.hh"
#include "G4VModularPhysicsList.hh"

class FTFP_BERT_HepEm : public G4VModularPhysicsList {
public:
  FTFP_BERT_HepEm(G4int ver = 1);
  virtual ~FTFP_BERT_HepEm() = default;

  FTFP_BERT_HepEm(const FTFP_BERT_HepEm &) = delete;
  FTFP_BERT_HepEm &operator=(const FTFP_BERT_HepEm &) = delete;
};

#endif
