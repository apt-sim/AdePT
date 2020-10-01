#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "vec3.h"

#ifdef __CUDACC__
#include "cuda.h"
#endif

#ifndef PART_H
#define PART_H

class part {
  public:
        part() {}
        // put this in the public for now ;( 
        vec3 m_pos; // position x, y, z in mm
        vec3 m_mom;  // momentum px, py, pz in MeV
        char m_ptype; // type (should be an enum) 22 = gamma, 11=e-, -11=e+
        // c'tor
        part(vec3 pos, vec3 mom, char ptype) { m_pos = pos; m_mom = mom; m_ptype = ptype; }

        float momentum() const {return m_mom.length(); }
};
  
#endif // PART_H

