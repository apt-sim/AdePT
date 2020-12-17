// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TRACKH
#define TRACKH

#include <CopCore/Ranluxpp.h>
#include <CopCore/PhysicalConstants.h>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavStateIndex.h>

#include <cfloat> // for FLT_MAX

enum TrackStatus { alive, dead };

struct track {
  RanluxppDouble rng_state;
  int            index{0};
  double         energy{10};
   
  vecgeom::Vector3D<double> pos;
  vecgeom::Vector3D<double> dir;
  vecgeom::NavStateIndex    current_state;
  vecgeom::NavStateIndex    next_state;

  int          mother_index{0};
  TrackStatus  status{alive};

  float interaction_length{FLT_MAX};
  float total_length{0.0};        // length since track start
   
  float energy_loss{0};           // primitive version of scoring
  int   number_of_secondaries{0}; // primitive version of scoring

  unsigned char current_process{0}; //    
  char          pdg{0};             // Large enough for e/g. Then uint16_t for p/n

  __device__ __host__ double uniform() { return rng_state.Rndm(); }

  __device__ __host__ int    charge() const
  { int chrg= (pdg== -11)-(pdg==11); return chrg; }

  __device__ __host__ double mass() const // Rest mass
  { return  (pdg== 22) ? 0 : copcore::units::kElectronMassC2; } 

  __host__ void print( int id = -1, bool verbose = false) const;
   
  __device__ __host__ void SwapStates()
  {
    auto state          = this->current_state;
    this->current_state = this->next_state;
    this->next_state    = state;
  }
};

inline
__global__
void GetNavStateIndices( const track& trk, int& currentTouchIndex, int& nextTouchIndex )
{
   // Needed because type of NavStateIndex (in track) is vecgeom:L:cxx::NavStateIndex
   // Solution for printing by Andrei Gheata.
   //
   // trk.current_state.IsOnBoundary();
   // std::cout << " current: "; // <<    setw(3) << next_state.IsOnBoundary();
   assert ( trk.current_state != nullptr && "Invalid Current state in track");
   auto currentLevel       = trk.current_state.GetLevel();
   auto currentTouchVolume = trk.current_state.ValueAt(currentLevel);
   currentTouchIndex = trk.current_state.GetNavIndex(currentLevel);
   // std::cout << " next: "; // <<    setw(3) << next_state.IsOnBoundary();
   assert ( trk.next_state != nullptr && "Invalid Next state in track");
   auto nextLevel       = trk.next_state.GetLevel();
   auto nextTouchVolume = trk.next_state.ValueAt(nextLevel);   
   nextTouchIndex = trk.next_state.GetNavIndex(nextLevel);
}
// void track::getNavStateIndex( )

#include <cstdio>
#include <iomanip>

inline
__host__
void track::print( int id , bool verbose ) const
{
   using std::setw;
   int oldPrec= std::cout.precision(5);
   
   static std::string particleName[3] = { "e-", "g", "e+" };
   std::cout // << " Track "
             << setw(4) << id << " "
          // << " addr= " << this   << " "
          //  << " pdg = " << setw(4) << (int) pdg
             << setw(2) << particleName[charge()+1] << " "  // Ok for e-/g/e+ only
             << " Pos: "
             << setw(11) << pos[0] << " , "
             << setw(11) << pos[1] << " , "
             << setw(11) << pos[2]
             << " s= "   << setw( 8 ) << interaction_length
             << " l= "   << setw( 8 ) << total_length
             << " kE= "  << setw( 5 ) << energy
             << " Dir: = "
             << setw(11) << dir[0] << " , "
             << setw(11) << dir[1] << " , "
             << setw(11) << dir[2] << " "
             << setw(11) << (status==alive ? "Alive": "Dead " ) << " "
             << " proc= "  << setw(1) << (int) current_process << " "
      ;
   if( verbose ) {
      std::cout << "<";
      // current_state.Print();
      int currentIndex, nextIndex; 
      GetNavStateIndices<<<1,1>>>( *this, currentIndex, nextIndex );
      cudaDeviceSynchronize();  // Needed ??
      
      std::cout << " current: " << currentIndex << " " 
                << setw(3) << (current_state.IsOnBoundary() ? " in" : "bnd" ) << " ";
      // std::cout << " - ";
      // current_state.printValueSequence(std::cout);      
      std::cout << " next: " << nextIndex << " "
                << setw(3) << (next_state.IsOnBoundary() ? " in" : "bnd" ) << " ";
      std::cout << " - ";      
      // next_state.printValueSequence(std::cout);
      std::cout << ">";
   }
   std::cout << std::endl;
   std::cout.precision(oldPrec);
}



#endif
