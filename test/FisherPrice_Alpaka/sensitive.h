#include "part.h"

#ifndef SENSITIVE_H
#define SENSITIVE_H


class sensitive{ 
   public:
     sensitive() {m_zmin=0.0; m_zmax=1000.; m_totalE=0.0; }
     #ifdef __CUDACC__
     __device__ 
     #endif 
     sensitive(float zmin, float zmax) {m_zmin = zmin; m_zmax = zmax; m_totalE=0.0; }
     float m_zmin, m_zmax;
     float m_totalE;

     inline void add(const part *mypart, float E ) { if (mypart->m_pos.z() > m_zmin && mypart->m_pos.z() < m_zmax) m_totalE += E; }

}; 

#endif // SENSITIVE_H
    




