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

#include "uniformMagField.h"
#include "fieldPropagator.h"

#include "trackBlock.h"
// using trackBlock_t  = adept::BlockData<track>;

using copcore::units::meter;
using copcore::units::GeV;
using copcore::units::MeV;

constexpr double  minX = -2.0 * meter, maxX = 2.0 * meter;
constexpr double  minY = -3.0 * meter, maxY = 3.0 * meter;
constexpr double  minZ = -5.0 * meter, maxZ = 5.0 * meter;

// constexpr double  maxP = 1.0 * GeV;

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
  px = 4 * MeV ; // maxP * 2.0 * ( aTrack.uniform() - 0.5 );   // -maxP to +maxP
  py = 0; // maxP * 2.0 * ( aTrack.uniform() - 0.5 );
  pz = 3 * MeV ; // maxP * 2.0 * ( aTrack.uniform() - 0.5 );

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

     
static float BzValue = 0.1 * copcore::units::tesla;

static float BfieldValue[3] = { 0.001 * copcore::units::tesla,
                               -0.001 * copcore::units::tesla,
                               BzValue };

// V1 -- field along Z axis
__global__ void fieldPropagatorBz_glob(adept::BlockData<track> *trackBlock, float Bz )
{
  vecgeom::Vector3D<double> endPosition;
  vecgeom::Vector3D<double> endDirection;

  int maxIndex = trackBlock->GetNused() + trackBlock->GetNholes();   
  
  // Non-block version:
  //   int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int pclIdx  = blockIdx.x * blockDim.x + threadIdx.x;  pclIdx < maxIndex;
           pclIdx += blockDim.x * gridDim.x)
  {
     track& aTrack= (*trackBlock)[pclIdx];

     // check if you are not outside the used block
     if (pclIdx >= maxIndex || aTrack.status == dead) continue;

     fieldPropagatorConstBz(aTrack, Bz, endPosition, endDirection);

     // Update position, direction     
     aTrack.pos = endPosition;  
     aTrack.dir = endDirection;
  }
}

// V2 -- constant field any direction 
__global__ void fieldPropagatorAnyDir_glob(adept::BlockData<track> *trackBlock,
                                           // float Bx, float By, float Bz,
                                           uniformMagField Bfield )  // by value !?
                                           // const uniformMagField& Bfield )   
{
  // template <type T> using Vector3D = vecgeom::Vector3D<T>;
  vecgeom::Vector3D<double> endPosition;
  vecgeom::Vector3D<double> endDirection;
  
  int maxIndex = trackBlock->GetNused() + trackBlock->GetNholes();   

  float Bvalue[3];
  Bfield.ObtainField( Bvalue );
  
  ConstFieldHelixStepper helixAnyB=  ConstFieldHelixStepper( Bvalue );
  
  // Non-block version:
  //   int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int pclIdx  = blockIdx.x * blockDim.x + threadIdx.x;  pclIdx < maxIndex;
           pclIdx += blockDim.x * gridDim.x)
  {
     track& aTrack= (*trackBlock)[pclIdx];

     // check if you are not outside the used block
     if (pclIdx >= maxIndex || aTrack.status == dead) continue;

     fieldPropagatorConstBgeneral(aTrack, helixAnyB, endPosition, endDirection);

     // Update position, direction     
     aTrack.pos = endPosition;  
     aTrack.dir = endDirection;
  }
}

int main( int argc, char** argv )
{
  // template<type T>
  using ThreeVector = vecgeom::Vector3D<double>; 
  constexpr int numBlocks=2, numThreadsPerBlock=16;
  int  totalNumThreads = numBlocks * numThreadsPerBlock;
  bool useBzOnly = true;

  if( argc > 1 )
     useBzOnly = false;
  
  const int numTracks = totalNumThreads; // Constant at first ...

  std::cout << "Magnetic field used: " << std::endl;
  if( !useBzOnly ){
     std::cout << "  Bx = " << BfieldValue[0] / copcore::units::tesla << " T " << std::endl;
     std::cout << "  By = " << BfieldValue[1] / copcore::units::tesla << " T " << std::endl;
  } 
  std::cout << "  Bz = " << BzValue / copcore::units::tesla << " T " << std::endl;

  // uniformMagField Bfield( BfieldValue );
  uniformMagField BfieldObj( BfieldValue );
  
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
  size_t blocksize = trackBlock_t::SizeOfInstance(capacity);
  char *buffer2    = nullptr;
  cudaError_t allocErr= cudaMallocManaged(&buffer2, blocksize);  // Allocated in Unified memory ... (baby steps)

  // auto trackBlock_dev  = trackBlock_t::MakeInstanceAt(capacity, buffer2);  
  auto trackBlock_uniq = trackBlock_t::MakeInstanceAt(capacity, buffer2); // Unified memory => _uniq

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

  std::cout << std::endl;
  std::cout << " Initialised tracks: " << std::endl;
  printTracks( trackBlock_uniq, false, numTracks );  

  overwriteTracks<<<numBlocks, numThreadsPerBlock>>>(trackBlock_uniq, numTracks, ++eventId, runId );
  cudaDeviceSynchronize();  
  std::cout << " Overwritten tracks: " << std::endl;
  printTracks( trackBlock_uniq, false, numTracks );  
  
  // Copy to array for host to cross-check
  track tracksStart_host[SmallNum];  
  // memcpy(tracksStart_host, &(*trackBlock_uniq)[0], SmallNum*sizeof(track));
  for ( int i = 0; i < SmallNum ; i++ ){
     tracksStart_host[i] = (*trackBlock_uniq)[i];
     // Print copy to check 
     // std::cout << " Orig: ";  (*trackBlock_uniq)[i].print( i );
     // std::cout << " Copy: ";  (tracksStart_host[i]).print( i );
  }
  
  // Else if stored on device: 
  //  cudaMemcpy(tracksStart_host, &(*trackBlock_dev)[0], SmallNum*sizeof(track), cudaMemcpyDeviceToHost );
  
  // 3. Propagate tracks -- on device
  if( useBzOnly ){
     fieldPropagatorBz_glob<<<numBlocks, numThreadsPerBlock>>>(trackBlock_uniq, BzValue );
     //*********
  } else {
     fieldPropagatorAnyDir_glob<<<numBlocks, numThreadsPerBlock>>>(trackBlock_uniq,
                                                                   BfieldObj );
  }
  cudaDeviceSynchronize();  

  // 4. Check results on host
  std::cout << " Calling move in field (host)." << std::endl;

  vecgeom::Vector3D<float> magFieldVec( BfieldValue[0],
                                        BfieldValue[1],
                                        BfieldValue[2] );
  ConstFieldHelixStepper  helixStepper( magFieldVec); // -> BfieldObj );  // Re-use it (expensive sqrt & div.)
  
  for( int i = 0; i<SmallNum ; i++){
     ThreeVector endPosition, endDirection;
     track  hostTrack = tracksStart_host[i];  // (*trackBlock_uniq)[i];
     // hostTrack.pos = 

     if( useBzOnly ){     
        fieldPropagatorConstBz( hostTrack, BzValue, endPosition, endDirection );
     } else {
        fieldPropagatorConstBgeneral( hostTrack, helixStepper, endPosition, endDirection );        
     }
     
     double move       = (endPosition  - hostTrack.pos).Mag();
     double deflection = (endDirection - hostTrack.dir).Mag();
     
     // Update position, direction     
     hostTrack.pos = endPosition;  
     hostTrack.dir = endDirection;

     track  devTrackVal= (*trackBlock_uniq)[i];
     ThreeVector posDiff = hostTrack.pos - devTrackVal.pos;     
     ThreeVector dirDiff = hostTrack.dir - devTrackVal.dir;

     constexpr double tol = 1.0e-07;
     bool badPosition  = posDiff.Mag() > tol * move;
     bool badDirection = dirDiff.Mag() > tol * deflection;
     
     if( badPosition || badDirection ){
        std::cout << std::endl;        
        std::cout << " Difference seen for Track " << i
                  << " addr = " << & (*trackBlock_uniq)[i]
                  << std::endl;
        std::cout << std::endl;
        // std::cout << " Track " << i << " addr = " << &aTrack << std::endl;
        // std::cout << " Track " << i << " pdg = " << aTrack.pdg
        //          << " x,y,z = " << aTrack.position[0] << " , " << aTrack.position[1]
        //          << " , " << aTrack.position[3] << std::endl;
        std::cout << " Ref (host) = ";
        hostTrack.print( i );

        std::cout << " Device     = ";
        devTrackVal.print( i );

        if( badPosition ){
           std::cout << " Position  diff = " << posDiff << " mag = " << posDiff.Mag() << " vs move      = " << move << " " << std::endl;
        }
        if( badDirection ){
           std::cout << " Direction diff = " << dirDiff << " mag = " << dirDiff.Mag() << " vs deflection = " << deflection << " " << std::endl;
        }
        std::cout << std::endl;
        // printTrack( hostTrack, i );
     }
  }
  // std::cout << " Tracks moved in host: " << std::endl;
  // printTrackBlock( trackBlock_uniq, numTracks );

  std::cout << std::endl;
  std::cout << " Calling move in field (device)" << std::endl;

  int maxIndex = trackBlock_uniq->GetNused() + trackBlock_uniq->GetNholes();     
  std::cout  << " maxIndex = " << maxIndex
             << " numTracks = " << numTracks << std::endl;

  // 5.  Report result of movement
  // 
  //          See where they went ?
  std::cout << " Ending tracks: " << std::endl;
  printTracks( trackBlock_uniq, false, numTracks );
}

