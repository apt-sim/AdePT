// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TRACKBLOCKH
#define TRACKBLOCKH

#include "track.h"

using trackBlock_t  = adept::BlockData<track>;

__host__
void printTracks( trackBlock_t* trackBlock,       //     adept::BlockData<track>*
                  bool          verbose = false,  // include info on navigation state ?
                  int           numTracks = -1 )  // only print those with index < numTracks
{

  int numLive = trackBlock->GetNused() ;
  int lastTrack= numLive + trackBlock->GetNholes();
  numTracks = ( numTracks == -1 ) ? lastTrack : min( numTracks, numLive );
  if( numTracks != numLive ){ std::cout << " Printing " << numTracks << " out of " << numLive 
                                        << " live tracks." << std::endl; } 
  
  // std::cout << " TrackBlock addr= " << trackBlock   << " " << std::endl;
  int numPrinted=0;
  for( int i = 0; i<lastTrack ; i++)
  {
     track& trk = (*trackBlock)[i];
     if( trk.status == alive ) {
        // printTrack( trk, i, verbose );
        trk.print( i, verbose );
        if( ++numPrinted >= numTracks ) return;
     }
  }
}


// {
  // unsigned int sizeOfTrack = trackBlock_t::SizeOfAlignAware;
  // size_t  bytesForTracks   = trackBlock_t::SizeOfInstance(numTracks);
  // mallocManaged(&buffer2, blocksize);
  
  // track tracksEnd_host[SmallNum];
  // cudaMemcpy(tracksEnd_host, trackBlock_dev, SmallNum * sizeOfTrack, // sizeof(track),
  //            cudaMemcpyDeviceToHost );
// }
#endif
