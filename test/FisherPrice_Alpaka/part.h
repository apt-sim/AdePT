// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file part.h
 * @brief class to represent electrons, positrons and photons. It stores a 3-momentum and 3-position vector and has a
 * type to denote the particle type. All data is public.
 * @author Davide Costanzo (d.costanzo@sheffield.ac.uk) and Mark Hodgkinson (d.costanzo@sheffield.ac.uk)
 */

#ifndef PART_H
#define PART_H

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "vec3.h"

class part {
public:
  /** @brief Nominal constructor */
  ALPAKA_FN_ACC part() {}

  /** @brief Constructor which sets the position 3-vector, momentum 3-vector and type.
   * Takes a @sa vec3 position and momentum and a char type as arguments.
   */
  ALPAKA_FN_ACC part(vec3 pos, vec3 mom, char ptype)
  {
    m_pos   = pos;
    m_mom   = mom;
    m_ptype = ptype;
  }

  /** @brief Returns the magnitude of the momentum 3-vector
   * Returns a float representing the magnitude, which is computed via @sa vec3::length()
   */
  ALPAKA_FN_ACC float momentum() const { return m_mom.length(); }

  ALPAKA_FN_ACC vec3 getPos() const { return m_pos; }
  ALPAKA_FN_ACC vec3 getMom() const { return m_mom; }
  ALPAKA_FN_ACC char getPType() const { return m_ptype; }

  ALPAKA_FN_ACC void setPos(const vec3 &pos) { m_pos = pos; }
  ALPAKA_FN_ACC void setMom(const vec3 &mom) { m_mom = mom; }
  ALPAKA_FN_ACC void setPType(const char &ptype) { m_ptype = ptype; }

private:
  vec3 m_pos;   ///< position x, y, z in mm
  vec3 m_mom;   ///< momentum px, py, pz in MeV
  char m_ptype; ///< type (should be an enum) 22 = gamma, 11=e-, -11=e+
};

#endif // PART_H
