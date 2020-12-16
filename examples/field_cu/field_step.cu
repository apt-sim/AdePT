// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  12 Nov 2020

#include <cstdio>
#include <iomanip>

#include <CopCore/SystemOfUnits.h>
#include <CopCore/PhysicalConstants.h>

// #include <CopCore/Ranluxpp.h>

#include <VecGeom/base/Vector3D.h>
#include "track.h"

#include <AdePT/BlockData.h>

#include "ConstBzFieldStepper.h"

using floatX_t = double;  //  float type for X = position
using floatE_t = double;  //  float type for E = energy  & momentum

using  TrackBlock_t    = adept::BlockData<track>;

template<unsigned int N>
struct FieldPropagationBuffer
{
  int      charge[N];
  floatX_t position[3][N];
  floatE_t momentum[3][N];
  int      index[N];
  bool     active[N];
};

using copcore::units::kElectronMassC2;

using copcore::units::meter;
using copcore::units::GeV;
using copcore::units::MeV;

constexpr floatX_t  minX = -2.0 * meter, maxX = 2.0 * meter;
constexpr floatX_t  minY = -3.0 * meter, maxY = 3.0 * meter;
constexpr floatX_t  minZ = -5.0 * meter, maxZ = 5.0 * meter;

// constexpr floatE_t  maxP = 1.0 * GeV;

constexpr floatX_t maxStepSize = 0.1 * ( (maxX - minX) + (maxY - minY) + (maxZ - minZ) );

#include <CopCore/Ranluxpp.h>

__device__ void initOneTrack(unsigned int  index,
                             uint64_t      rngBase,
                             track       & aTrack
   )
{
  // Very basic initial state for RNG ... to be improved
  aTrack.rng_state.SetSeed( rngBase + (uint64_t) index);
   
  float r = aTrack.uniform(); // curand_uniform(states);  
  // aTrack.charge = ( r < 0.45 ? -1 : ( r< 0.9 ? 0 : +1 ) );
  constexpr  int  pdgElec = 11 , pdgGamma = 22;
  aTrack.pdg = ( r < 0.45 ? pdgElec : ( r< 0.9 ? pdgGamma : -pdgElec ) );

  aTrack.pos[0] = 0.0; // minX + aTrack.uniform() * ( maxX - minX );
  aTrack.pos[1] = 0.0; // minY + aTrack.uniform() * ( maxY - minY );
  aTrack.pos[2] = 0.0; // minZ + aTrack.uniform() * ( maxZ - minZ );

  floatE_t  px, py, pz;
  px = 4 * MeV ; // maxP * 2.0 * ( aTrack.uniform() - 0.5 );   // -maxP to +maxP
  py = 0; // maxP * 2.0 * ( aTrack.uniform() - 0.5 );
  pz = 3 * MeV ; // maxP * 2.0 * ( aTrack.uniform() - 0.5 );

  floatE_t  pmag2 =  px*px + py*py + pz*pz;
  floatE_t  inv_pmag = 1.0 / std::sqrt(pmag2);
  aTrack.dir[0] = px * inv_pmag; 
  aTrack.dir[1] = py * inv_pmag; 
  aTrack.dir[2] = pz * inv_pmag;

  aTrack.interaction_length = 0.001 * index * maxStepSize ; // aTrack.uniform() * maxStepSize;
  
  floatE_t  mass = ( aTrack.pdg == pdgGamma ) ?  0.0 : kElectronMassC2 ; // rest mass
  aTrack.energy = pmag2 / ( sqrt( mass * mass + pmag2 ) + mass);
}

// this GPU kernel function is used to initialize 
//     .. the particles' state ?

__global__ void initTracks( adept::BlockData<track> *trackBlock,
                            unsigned int numTracks,                            
                            unsigned int eventId,
                            unsigned int   runId = 101
                          )
{
  /* initialize the tracks with random particles */
  int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pclIdx >= numTracks) return;

  track* pTrack =   trackBlock->NextElement();

  uint64_t  rngBase =     runId * (uint64_t(1)<<52)
                      + eventId * (uint64_t(1)<<36);

  initOneTrack( pclIdx, rngBase, *pTrack ); // , &states[pclIdx] );
}

constexpr float BzValue = 0.1 * copcore::units::tesla; 

// VECCORE_ATT_HOST_DEVICE
__host__  __device__ 
void EvaluateField( const floatX_t pos[3], float fieldValue[3] )
{
    fieldValue[0]= 0.0;
    fieldValue[1]= 0.0;
    fieldValue[2]= BzValue;        
}

#ifdef USE_VECTOR3D
#include <VecGeom/Vector3D.h>
#endif

__host__ __device__
void moveInField(track& track)
{
  floatX_t  step= track.interaction_length;

  // Charge for e+ / e-  only    ( gamma / other neutrals also ok.) 
  int    charge = track.charge(); // (track.pdg == -11) - (track.pdg == 11);
  
  if ( charge == 0.0 ) return;
  
  // floatX_t pclPosition[3];

  // Evaluate initial field value
  // EvaluateField( pclPosition3d, fieldVector );

  // float restMass = ElectronMass;  // For now ... 
  floatE_t kinE = track.energy;
  floatE_t momentumMag = sqrt( kinE * ( kinE + 2.0 * kElectronMassC2) );
  
  // Collect position, momentum
  // floatE_t momentum[3] = { momentumMag * track.dir[0], 
  //                          momentumMag * track.dir[1], 
  //                          momentumMag * track.dir[2] } ;
#ifdef VECTOR3D    
  vecGeom::Vector3D<floatX_t> positionOut3d(  track.pos );
  vecGeom::Vector3D<floatX_t> directionOut3d( track.dir );
#endif
  
  ConstBzFieldStepper  helixBz(BzValue);

#if 0    
  track.pos[0] += 0.1 * ( 1. + 0.0001 * step );
  track.pos[1] += 0.2;
  track.pos[2] += 0.3;
  track.direction[0] += 0.3;
  track.direction[1] += 0.2;
  track.direction[2] += 0.1;
#endif    

  // For now all particles ( e-, e+, gamma ) can be propagated using this
  //   for gammas  charge = 0 works, and ensures that it goes straight.
#ifndef USE_VECTOR3D
  floatX_t xOut, yOut, zOut, dirX, dirY, dirZ;  
  helixBz.DoStep( track.pos[0], track.pos[1], track.pos[2],
                  track.dir[0], track.dir[1], track.dir[2],
                  charge, momentumMag, step,
                  xOut, yOut, zOut, dirX, dirY, dirZ );                  

  // Update position, direction
  track.pos[0] = xOut;
  track.pos[1] = yOut;
  track.pos[2] = zOut;
  track.dir[0] = dirX;
  track.dir[1] = dirY;
  track.dir[2] = dirZ;  
#else  
  helixBz.DoStep( track.pos, track.dir, charge, momentumMag, step,
                  positionOut3d, directionOut3d);

  // Update position, direction
  track.pos = positionOut3d;  
  // track.pos[0] = positionOut3d[0];
  // track.pos[1] = positionOut3d[1];
  // track.pos[2] = positionOut3d[2];
  track.dir = directionOut3d;
  // track.dir[0] = directionOut3d[0];
  // track.dir[1] = directionOut3d[1];
  // track.dir[2] = directionOut3d[2];
#endif

  // Alternative: load into local variables ?
  // float xIn= track.position[0], yIn= track.position[1], zIn = track.position[2];
  // float dirXin= track.direction[0], dirYin = track.direction[1], dirZin = track.direction[2];


}

// V1 -- one per warp
__global__ void moveInField_glob(adept::BlockData<track> *trackBlock, int numTracksChk )
{
  int maxIndex = trackBlock->GetNused() + trackBlock->GetNholes();   

  // Non-block version:
  //   int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int pclIdx  = blockIdx.x * blockDim.x + threadIdx.x;  pclIdx < maxIndex;
           pclIdx += blockDim.x * gridDim.x)
  {
     track &aTrack= (*trackBlock)[pclIdx];

     // check if you are not outside the used block
     if (pclIdx >= maxIndex || aTrack.status == dead) continue;
  
     moveInField(aTrack);
  }
}

void reportOneTrack( const track & aTrack, int id = -1 )
{
   using std::setw;
   
   std::cout << " Track " << setw(4) << id
             << " addr= " << & aTrack   << " "
             << " pdg = " << setw(4) << aTrack.pdg
             << " x,y,z = "
             << setw(12) << aTrack.pos[0] << " , "
             << setw(12) << aTrack.pos[1] << " , "
             << setw(12) << aTrack.pos[2]
             << " step = " << setw( 12 ) << aTrack.interaction_length
             << " kinE = " << setw( 10 ) << aTrack.energy
             << " Dir-x,y,z = "
             << setw(12) << aTrack.dir[0] << " , "
             << setw(12) << aTrack.dir[1] << " , "
             << setw(12) << aTrack.dir[2]
             << std::endl;
}

void reportTracks( TrackBlock_t* trackBlock, unsigned int numTracks )
{
  // unsigned int sizeOfTrack = TrackBlock_t::SizeOfAlignAware;
  // size_t  bytesForTracks   = TrackBlock_t::SizeOfInstance(numTracks);
  // mallocManaged(&buffer2, blocksize);
  
  // track tracksEnd_host[SmallNum];
  // cudaMemcpy(tracksEnd_host, trackBlock_dev, SmallNum * sizeOfTrack, // sizeof(track),
  //            cudaMemcpyDeviceToHost );

  // std::cout << " TrackBlock addr= " << trackBlock   << " " << std::endl;
  for( int i = 0; i<numTracks ; i++) {
     track& aTrack = (*trackBlock)[i];
     reportOneTrack( aTrack, i );
  }
}

int main( int argc, char** argv )
{
  constexpr int numBlocks=2, numThreadsPerBlock=16;
  int  totalNumThreads = numBlocks * numThreadsPerBlock;
  
  const int numTracks = totalNumThreads; // Constant at first ...
  
  std::cout << " Bz = " << BzValue / copcore::units::tesla << " T " << std::endl;
  
  // Track capacity of the block
  constexpr int capacity = 1 << 16;

  // 1. Create container of Tracks  BlockData<track>
  // -----------------------------------------------------
  std::cout << " Creating track buffer for " << capacity << " tracks -" // " on GPU device."
            << " in Unified Memory." 
            << std::endl;
  
  // Allocate a block of tracks with capacity larger than the total number of spawned threads
  // Note that if we want to allocate several consecutive block in a buffer, we have to use
  // Block_t::SizeOfAlignAware rather than SizeOfInstance to get the space needed per block
  size_t blocksize = TrackBlock_t::SizeOfInstance(capacity);
  char *buffer2    = nullptr;
  cudaError_t allocErr= cudaMallocManaged(&buffer2, blocksize);  // Allocated in Unified memory ... (baby steps)

  // auto trackBlock_dev  = TrackBlock_t::MakeInstanceAt(capacity, buffer2);  
  auto trackBlock_uniq = TrackBlock_t::MakeInstanceAt(capacity, buffer2);

  // 2.  Initialise track - on device
  // --------------------------------
  std::cout << " Initialising tracks." << std::endl;
  std::cout << " Max step size = " << maxStepSize << std::endl;

  unsigned  int runId= 101, eventId = 1;
  unsigned  int numTracksEv1 = numTracks / 2;
  initTracks<<<numBlocks, numThreadsPerBlock>>>(trackBlock_uniq, numTracksEv1, eventId, runId );
  initTracks<<<numBlocks, numThreadsPerBlock>>>(trackBlock_uniq, numTracks-numTracksEv1, ++eventId, runId );  
  cudaDeviceSynchronize();

  const unsigned int SmallNum= std::max( 2, numTracks);
  // track tracksStart_host[SmallNum];
  
  // cudaMemcpy(tracksStart_host, trackBlock_uniq, SmallNum*sizeof(SimpleTrack), cudaMemcpyDeviceToHost );

  std::cout << std::endl;
  std::cout << " Initialised tracks: " << std::endl;
  reportTracks( trackBlock_uniq, numTracks );  

  // 3.  Move tracks in field - for one step
  // ----------------------------------------
  std::cout << " Calling move in field (host)." << std::endl;
  for( int i = 0; i<SmallNum ; i++){
     // (*block)[particle_index].energy = energy;     
     track& aTrack = (*trackBlock_uniq)[i];
     // moveInField( aTrack );

     track  ghostTrack = aTrack;
     // reportOneTrack( ghostTrack, i );
     
     moveInField( ghostTrack );
     
     // std::cout << " Track " << i << " addr = " << &aTrack << std::endl;
     // std::cout << " Track " << i << " pdg = " << aTrack.pdg
     //          << " x,y,z = " << aTrack.position[0] << " , " << aTrack.position[1]
     //          << " , " << aTrack.position[3] << std::endl;
     reportOneTrack( ghostTrack, i );   
  }
  // std::cout << " Tracks moved in host: " << std::endl;
  // reportTracks( trackBlock_uniq, numTracks );

  std::cout << std::endl;
  std::cout << " Calling move in field (device)" << std::endl;

  int maxIndex = trackBlock_uniq->GetNused() + trackBlock_uniq->GetNholes();     
  std::cout  << " maxIndex = " << maxIndex
             << " numTracks = " << numTracks << std::endl;
  
  moveInField_glob<<<numBlocks, numThreadsPerBlock>>>(trackBlock_uniq, numTracks);
  //*********
  cudaDeviceSynchronize();  

  // 4.  Report result of movement
  // 
  //          See where they went ?
  std::cout << " Ending tracks: " << std::endl;
  reportTracks( trackBlock_uniq, numTracks );
}

