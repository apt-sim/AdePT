// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file particleStack.h
 * @brief stores an array of particles, particle, of size @t ARRAYSIZE
 * @author Davide Costanzo (d.costanzo@sheffield.ac.uk) and Mark Hodgkinson (d.costanzo@sheffield.ac.uk)
 */

#ifndef PARTICLESTACK_H
#define PARTICLESTACK_H

#include "particle.h"

#include <alpaka/alpaka.hpp>

template <unsigned int ARRAYSIZE>
class particleStack {
private:
  /** An array, of particle, of size ARRAYSIZE */
  particle *m_stack[ARRAYSIZE];
  /** integer to track how many particle are currently contained in the array */
  int m_nPart;

public:
  /** @brief constructor which sets all array elements to zero */
  ALPAKA_FN_ACC particleStack()
  {
    m_nPart = 0;
    for (int ii = 0; ii < ARRAYSIZE; ii++)
      m_stack[ii] = 0;
  }

  /** @brief check whether the number of particle currently stored is zero or not.
   * returns the result of the query as bool
   */
  ALPAKA_FN_ACC bool empty()
  {
    if (0 == m_nPart)
      return true;
    else
      return false;
  }

  /** @brief gets the particle at the end of the array
   * returns a pointer of type particle directly from the stored array m_stack.
   * @remark assumes the array m_stack is NOT empty.
   */
  ALPAKA_FN_ACC particle *top() { return m_stack[m_nPart - 1]; }

  /** @brief set the particle pointer stored at the end of the array to zero */
  ALPAKA_FN_ACC void pop()
  {
    m_stack[m_nPart - 1] = 0;
    m_nPart--;
  }

  /** @brief add a particle to the array.
   * myPart is a pointer to a particle */
  ALPAKA_FN_ACC void push(particle *myPart)
  {
    m_stack[m_nPart] = myPart;
    m_nPart++;
  }

  /** @brief gets the stored arrays size.
   * returns the stored integer m_nPart */
  ALPAKA_FN_ACC int size() { return m_nPart; }
};
#endif // PARTICLESTACK
