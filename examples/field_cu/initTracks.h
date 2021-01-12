
#ifndef INIT_TRACKS_H
#define INIT_TRACKS_H  1

using copcore::units::meter;
using copcore::units::GeV;
using copcore::units::MeV;

constexpr double  minX = -2.0 * meter, maxX = 2.0 * meter;
constexpr double  minY = -3.0 * meter, maxY = 3.0 * meter;
constexpr double  minZ = -5.0 * meter, maxZ = 5.0 * meter;

// constexpr double  maxP = 1.0 * GeV;
constexpr double  minP = 25.0 * MeV;

constexpr double maxStepSize = 0.1 * ( (maxX - minX) + (maxY - minY) + (maxZ - minZ) );

#include <CopCore/Ranluxpp.h>

__device__ void initOneTrack(unsigned int  index,
                             uint64_t      rngBase,
                             track       & aTrack,
                             unsigned int  eventId
   )
{
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
  
  aTrack.pos[0] = 0.0; // minX + aTrack.uniform() * ( maxX - minX );
  aTrack.pos[1] = 0.0; // minY + aTrack.uniform() * ( maxY - minY );
  aTrack.pos[2] = 0.0; // minZ + aTrack.uniform() * ( maxZ - minZ );

  double  px, py, pz;
  px = 4 * MeV  ; // + minP * 2.0 * ( aTrack.uniform() - 0.5 );   // later: -maxP to +maxP
  py = 0        ; // + minP * 2.0 * ( aTrack.uniform() - 0.5 );
  pz = 3 * MeV  ; // + minP * 2.0 * ( aTrack.uniform() - 0.5 );

  double  pmag2 =  px*px + py*py + pz*pz;
  double  inv_pmag = 1.0 / std::sqrt(pmag2);
  aTrack.dir[0] = px * inv_pmag; 
  aTrack.dir[1] = py * inv_pmag; 
  aTrack.dir[2] = pz * inv_pmag;

  aTrack.interaction_length = 0.001 * (index+1) * maxStepSize ; // aTrack.uniform() * maxStepSize;
  
  // double  mass = ( aTrack.pdg == pdgGamma ) ?  0.0 : kElectronMassC2 ; // rest mass
  double  mass = aTrack.mass();
  aTrack.energy = pmag2 / ( sqrt( mass * mass + pmag2 ) + mass);
  // More accurate than   ( sqrt( mass * mass + pmag2 ) - mass);
}

// this GPU kernel function is used to create and initialize 
//     .. the particles' state 

__global__ void initTracks( adept::BlockData<track> *trackBlock,
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

  initOneTrack( pclIdx, rngBase, *pTrack, eventId );
}


__global__ void overwriteTracks( adept::BlockData<track> *trackBlock,
                                 unsigned int numTracks,
                                 unsigned int eventId,
                                 unsigned int   runId = 0                                 
   )
{
  /* initialize the tracks with random particles */
  int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if( trackBlock->GetNholes() > 0 ) return;
  // We can only overwrite if there are no holes !
  
  numTracks = max ( numTracks, (unsigned int) trackBlock->GetNused() );
  if (pclIdx >= numTracks ) return;
  
  track & trk = (*trackBlock)[pclIdx];
  uint64_t  rngBase =     runId * (uint64_t(1)<<52)
                      + eventId * (uint64_t(1)<<36);
  
  initOneTrack( pclIdx, rngBase, (*trackBlock)[pclIdx], eventId );
}

#endif
