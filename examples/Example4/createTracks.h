
#ifndef CREATE_TRACKS_H
#define CREATE_TRACKS_H  1

#include <VecGeom/base/Config.h>
#include <VecGeom/volumes/PlacedVolume.h>

#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif

#include <CopCore/SystemOfUnits.h>

using copcore::units::cm;
using copcore::units::GeV;
using copcore::units::MeV;

//  Extents of box in which initial tracks will be generated

// constexpr double  maxP = 1.0 * GeV;

#include <CopCore/Ranluxpp.h>

__device__ void createOneTrack(unsigned int  index,
                             uint64_t      rngBase,
                             track       & aTrack,
                             unsigned int  eventId
   )
{
  constexpr double  minX = -2.0 * cm, maxX = 2.0 * cm;
  constexpr double  minY = -3.0 * cm, maxY = 3.0 * cm;
  constexpr double  minZ = -5.0 * cm, maxZ = 5.0 * cm;

  constexpr double maxStepSize = 0.1 * ( (maxX - minX) + (maxY - minY) + (maxZ - minZ) );
   
  // Very basic initial state for RNG ... to be improved
  aTrack.rng_state.SetSeed( rngBase + (uint64_t) index);
   
  float r = aTrack.uniform(); // curand_uniform(states);  
  // aTrack.charge = ( r < 0.45 ? -1 : ( r< 0.9 ? 0 : +1 ) );
  constexpr  int  pdgElec = 11 , pdgGamma = 22;
  aTrack.pdg = ( r < 0.45 ? pdgElec : ( r< 0.9 ? pdgGamma : -pdgElec ) );

  // Make the first tracks electrons -- for now
  if( index < 20 ) aTrack.pdg = pdgElec;
  
  aTrack.index = index;
  aTrack.eventId = eventId;
  
  aTrack.pos[0] =  minX + aTrack.uniform() * ( maxX - minX );
  aTrack.pos[1] =  minY + aTrack.uniform() * ( maxY - minY );
  aTrack.pos[2] =  minZ + aTrack.uniform() * ( maxZ - minZ );

  constexpr double  minP = 25.0 * MeV;  
  double  px, py, pz;
  px = 4 * MeV + minP * 2.0 * ( aTrack.uniform() - 0.5 );   // later: -maxP to +maxP
  py = 0       + minP * 2.0 * ( aTrack.uniform() - 0.5 );
  pz = 3 * MeV + minP * 2.0 * ( aTrack.uniform() - 0.5 );

  double  pmag2 =  px*px + py*py + pz*pz;
  double  inv_pmag = 1.0 / std::sqrt(pmag2);
  aTrack.dir[0] = px * inv_pmag; 
  aTrack.dir[1] = py * inv_pmag; 
  aTrack.dir[2] = pz * inv_pmag;

  aTrack.interaction_length = 0.001 * (index+1) * maxStepSize ; // aTrack.uniform() * maxStepSize;
  
  double  mass = aTrack.mass();
  aTrack.energy = pmag2 / ( sqrt( mass * mass + pmag2 ) + mass);
  // More accurate than   ( sqrt( mass * mass + pmag2 ) - mass);
}

// this GPU kernel function is used to create and initialize 
//     .. the particles' state 

__global__ void createTracks( adept::BlockData<track> *trackBlock,
                              const vecgeom::VPlacedVolume *world,                              
                              unsigned int numTracks,
                              unsigned int eventId,
                              unsigned int   runId = 0
                          )
{
  /* initialize the tracks with random particles */
  int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pclIdx >= numTracks) return;

  track* pTrack =   trackBlock->NextElement();

  uint64_t  rngBase =     runId * (uint64_t(1)<<52)
                      + eventId * (uint64_t(1)<<36);

  createOneTrack( pclIdx, rngBase, *pTrack, eventId );

  LoopNavigator::LocatePointIn(world, pTrack->pos, pTrack->current_state, true);
  pTrack->next_state = pTrack->current_state;  
}

#endif
