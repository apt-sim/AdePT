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

#include "ConstFieldHelixStepper.h"

#include "IterationStats.h"
// Statistics for propagation chords
IterationStats      *chordIterStats     = nullptr;
__device__
IterationStats_impl *chordIterStats_impl = nullptr;
// Needed for fieldPropagatorConstBz.h -- for now

#include "fieldPropagatorConstBz.h"
#include "fieldPropagatorConstBany.h"

#include "trackBlock.h"
// using trackBlock_t  = adept::BlockData<track>;

#include "initTracks.h"

static float BzValue = 0.1 * copcore::units::tesla;

static float BfieldValue[3] = { 0.001 * copcore::units::tesla,
                               -0.001 * copcore::units::tesla,
                               BzValue };

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

  uniformMagField Bfield( BfieldValue );
  
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

  // 2.  Initialise tracks - on device
  // ---------------------------------
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

  // Overwrite the tracks - to get simpler, predictable order (for easier comparisons)
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
  fieldPropagatorConstBz   fieldPropagatorBz;
  fieldPropagatorConstBany fieldPropagatorBany;
  if( useBzOnly ){
     // Uniform field - parallel to z axis
     moveInField<<<numBlocks, numThreadsPerBlock>>>(trackBlock_uniq, fieldPropagatorBz, BzValue );
     //*********
  }
  else
  {
     // Uniform field - not along z axis
     moveInField<<<numBlocks, numThreadsPerBlock>>>(trackBlock_uniq, fieldPropagatorBany, Bfield );
     //*********
  }
  cudaDeviceSynchronize();  

  // 4. Check results on host
  std::cout << " Calling move in field (host)." << std::endl;

  vecgeom::Vector3D<float> magFieldVec( BfieldValue[0],
                                        BfieldValue[1],
                                        BfieldValue[2] );
  ConstFieldHelixStepper  helixStepper( magFieldVec); // -> Bfield );  // Re-use it (expensive sqrt & div.)
  
  for( int i = 0; i<SmallNum ; i++){
     ThreeVector endPosition, endDirection;
     track  hostTrack = tracksStart_host[i];  // (*trackBlock_uniq)[i];
     // hostTrack.pos = 

     if( useBzOnly ){     
        // fieldPropagatorConstBz( hostTrack, BzValue, endPosition, endDirection );
        // fieldPropagatorConstBz::moveInField( hostTrack, BzValue, endPosition, endDirection );
        fieldPropagatorBz.stepInField( hostTrack, BzValue, endPosition, endDirection );
     } else {
        // fieldPropagatorConstBgeneral( hostTrack, helixStepper, endPosition, endDirection );
        fieldPropagatorBany.stepInField( hostTrack, helixStepper, endPosition, endDirection );
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

