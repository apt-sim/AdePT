// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file sensitive.h
 * @brief represents a sensitive particle detector
 * @author Davide Costanzo (d.costanzo@sheffield.ac.uk) and Mark Hodgkinson (d.costanzo@sheffield.ac.uk)
 */

#ifndef SENSITIVE_H
#define SENSITIVE_H

#include "part.h"

class sensitive {
public:
  /** @brief nominal constructor sets the class data to default values */
  sensitive()
  {
    m_zmin   = 0.0;
    m_zmax   = 1000.;
    m_totalE = 0.0;
  }

  /** @brief constructor sets user supplied values of zmin and zmaz. m_totalE
   * is set to zero
   * */
  sensitive(float zmin, float zmax)
  {
    m_zmin   = zmin;
    m_zmax   = zmax;
    m_totalE = 0.0;
  }

  float m_zmin, m_zmax; ///< variables to represent extent of detector in z direction
  float m_totalE;       ///< total energy depoisted in this detector

  /** @brief add energy to the sensitive detector from a particle.
   * mypart is the particle being considered and E is its energy loss.
   */
  inline void add(const part *mypart, float E)
  {
    if (mypart->m_pos.z() > m_zmin && mypart->m_pos.z() < m_zmax) m_totalE += E;
  }
};

#endif // SENSITIVE_H
