// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  12 Nov 2020

#include <cstdio>
#include <iomanip>

#include <CopCore/SystemOfUnits.h>
#include <CopCore/PhysicalConstants.h>

// #include <CopCore/Ranluxpp.h>

#include <VecGeom/base/Vector3D.h>
// #include "track.h"

#include <AdePT/BlockData.h>

#include "ConstBzFieldStepper.h"

using floatX_t = double;  //  float type for X = position
using floatE_t = double;  //  float type for E = energy  & momentum

// using SimpleTrack = track;

struct SimpleTrack {
  int      index{0};
  int      pdg{0};
  floatE_t energy{0};  // kineticEnergy;
  floatX_t position[3]{0};
  vecgeom::Vector3D<double> pos;
  vecgeom::Vector3D<double> dir;
  floatX_t interaction_length;    // Current step size
};

using  TrackBlock_t    = adept::BlockData<SimpleTrack>;

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

#include <curand.h>
#include <curand_kernel.h>

__device__ void initOneTrack(int            index,
                             SimpleTrack   &track,
                             curandState_t *states
   )
{
  float r = curand_uniform(states);  
  // track.charge = ( r < 0.45 ? -1 : ( r< 0.9 ? 0 : +1 ) );
  constexpr  int  pdgElec = 11 , pdgGamma = 22;
  track.pdg = ( r < 0.45 ? pdgElec : ( r< 0.9 ? pdgGamma : -pdgElec ) );

  track.pos[0] = 0.0;   // minX + curand_uniform(states) * ( maxX - minX );
  track.pos[1] = 0.0; // minY + curand_uniform(states) * ( maxY - minY );
  track.pos[2] = 0.0; // minZ + curand_uniform(states) * ( maxZ - minZ );

  floatE_t  px, py, pz;
  px = 4 * MeV ; // maxP * 2.0 * ( curand_uniform(states) - 0.5 );   // -maxP to +maxP
  py = 0; // maxP * 2.0 * ( curand_uniform(states) - 0.5 );
  pz = 3 * MeV ; // maxP * 2.0 * ( curand_uniform(states) - 0.5 );

  floatE_t  pmag2 =  px*px + py*py + pz*pz;
  floatE_t  inv_pmag = 1.0 / std::sqrt(pmag2);
  track.dir[0] = px * inv_pmag; 
  track.dir[1] = py * inv_pmag; 
  track.dir[2] = pz * inv_pmag;

  track.interaction_length = 0.001 * index * maxStepSize ; // curand_uniform(states) * maxStepSize;
  
  floatE_t  mass = ( track.pdg == pdgGamma ) ?  0.0 : kElectronMassC2 ; // rest mass
  track.energy = pmag2 / ( sqrt( mass * mass + pmag2 ) + mass);
}

// this GPU kernel function is used to initialize 
//     .. the particles' state ?

__global__ void initTracks(adept::BlockData<SimpleTrack> *trackBlock,
                           curandState_t *states,
                           int maxIndex
                          )
{
  /* initialize the tracks with random particles */
  int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pclIdx >= maxIndex) return;

  SimpleTrack* pTrack =   trackBlock->NextElement();

  initOneTrack( pclIdx, *pTrack, &states[pclIdx] );
}

__global__ void initCurand(unsigned long long runSeed, curandState_t *states)
{
  /* initialize the state */
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* All threads gets the same seed, a different sequence number, no offset */
  curand_init(runSeed, id, 0, &states[id]);
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
void moveInField(SimpleTrack& track)
{
  floatX_t  step= track.interaction_length;

  // Charge for e+ / e-  only    ( gamma / other neutrals also ok.) 
  int    charge = (track.pdg == -11) - (track.pdg == 11);
  
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
__global__ void moveInField_glob(adept::BlockData<SimpleTrack> *trackBlock,
                            int maxIndex)
{
  int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;
  SimpleTrack &track= (*trackBlock)[pclIdx];

  // check if you are not outside the used block
  if (pclIdx >= maxIndex ) return;  // || !track.active ) return;

  moveInField(track);  
}

void reportOneTrack( const SimpleTrack & track, int id = -1 )
{
   using std::setw;
   
   std::cout << " Track " << setw(4) << id
             << " addr= " << & track   << " "
             << " pdg = " << setw(4) << track.pdg
             << " x,y,z = "
             << setw(12) << track.pos[0] << " , "
             << setw(12) << track.pos[1] << " , "
             << setw(12) << track.pos[2]
             << " step = " << setw( 12 ) << track.interaction_length
             << " kinE = " << setw( 10 ) << track.energy
             << " Dir-x,y,z = "
             << setw(12) << track.dir[0] << " , "
             << setw(12) << track.dir[1] << " , "
             << setw(12) << track.dir[2]
             << std::endl;
}

void reportTracks( TrackBlock_t* trackBlock, unsigned int numTracks )
{
  // unsigned int sizeOfTrack = TrackBlock_t::SizeOfAlignAware;
  // size_t  bytesForTracks   = TrackBlock_t::SizeOfInstance(numTracks);
  // mallocManaged(&buffer2, blocksize);
  
  // SimpleTrack tracksEnd_host[SmallNum];
  // cudaMemcpy(tracksEnd_host, trackBlock_dev, SmallNum * sizeOfTrack, // sizeof(SimpleTrack),
  //            cudaMemcpyDeviceToHost );

  // std::cout << " TrackBlock addr= " << trackBlock   << " " << std::endl;
  for( int i = 0; i<numTracks ; i++) {
     SimpleTrack& track = (*trackBlock)[i];
     reportOneTrack( track, i );
  }
}

int main()
{
  constexpr int numBlocks=2, numThreadsPerBlock=4;
  int  totalNumThreads = numBlocks * numThreadsPerBlock;
  
  const int numTracks = totalNumThreads; // Constant at first ...
  
  std::cout << " Bz = " << BzValue / copcore::units::tesla << " T " << std::endl;
  
  // Initialize Curand
  //   How-to see: https://docs.nvidia.com/cuda/curand/device-api-overview.html

  curandState_t *randState_dev;
  auto cudErr= cudaMalloc((void **)&randState_dev,
                          totalNumThreads * sizeof(curandState_t));

  // curandStateMRG32k3a_t  *devMRGStates;

  unsigned long long runSeed = 12345; 
  std::cout << " Initialising curand with run seed = ." << runSeed << std::endl;

  initCurand<<<numBlocks, numThreadsPerBlock>>>(runSeed, randState_dev);
  cudaDeviceSynchronize();

  // Track capacity of the block
  constexpr int capacity = 1 << 16;

  // 1. Create container of Tracks  BlockData<SimpleTrack>
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

  initTracks<<<numBlocks, numThreadsPerBlock>>>(trackBlock_uniq, randState_dev, numTracks);
  cudaDeviceSynchronize();

  const unsigned int SmallNum= std::max( 2, numTracks);
  // SimpleTrack tracksStart_host[SmallNum];
  
  // cudaMemcpy(tracksStart_host, trackBlock_uniq, SmallNum*sizeof(SimpleTrack), cudaMemcpyDeviceToHost );

  std::cout << std::endl;
  std::cout << " Initialised tracks: " << std::endl;
  reportTracks( trackBlock_uniq, numTracks );  

  // 3.  Move tracks in field - for one step
  // ----------------------------------------
  std::cout << " Calling move in field (host)." << std::endl;
  for( int i = 0; i<SmallNum ; i++){
     // (*block)[particle_index].energy = energy;     
     SimpleTrack& track = (*trackBlock_uniq)[i];
     // moveInField( track );

     SimpleTrack  ghostTrack = track;
     // reportOneTrack( ghostTrack, i );
     
     moveInField( ghostTrack );
     
     // std::cout << " Track " << i << " addr = " << &track << std::endl;
     // std::cout << " Track " << i << " pdg = " << track.pdg
     //          << " x,y,z = " << track.position[0] << " , " << track.position[1]
     //          << " , " << track.position[3] << std::endl;
     reportOneTrack( ghostTrack, i );   
  }
  // std::cout << " Tracks moved in host: " << std::endl;
  // reportTracks( trackBlock_uniq, numTracks );

  std::cout << std::endl;
  std::cout << " Calling move in field (device)" << std::endl;

  moveInField_glob<<<numBlocks, numThreadsPerBlock>>>(trackBlock_uniq, numTracks);
  //*********
  cudaDeviceSynchronize();  

  // 4.  Report result of movement
  // 
  //          See where they went ?
  std::cout << " Ending tracks: " << std::endl;
  reportTracks( trackBlock_uniq, numTracks );
}

