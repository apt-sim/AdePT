#include "part.h"

#ifndef PARTICLESTACK_H
#define PARTICLESTACK_H

#include <alpaka/alpaka.hpp>

template <unsigned int ARRAYSIZE> class particleStack {
  private:
    part *m_stack[ARRAYSIZE]; // allocate space for ARRAYSIZE pointers in the stack
    int nPart; 

  public:
        ALPAKA_FN_ACC particleStack() {nPart=0; for (int ii=0; ii< ARRAYSIZE; ii++) m_stack[ii]=0; } // set all pointers to zero

        ALPAKA_FN_ACC bool empty() {if (nPart ==0 ) return true; else return false; }

        ALPAKA_FN_ACC part* top() { return m_stack[nPart-1]; } // FIXME If stack is empty we are in big trouble!

        ALPAKA_FN_ACC void pop() {m_stack[nPart-1]=0; nPart--; }

        ALPAKA_FN_ACC void push(part *myPart) { m_stack[nPart]=myPart; nPart++; }

        ALPAKA_FN_ACC int size() {return nPart;}

};
#endif // PARTICLESTACK

