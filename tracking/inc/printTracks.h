// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef PRINT_TRACKS_H
#define PRINT_TRACKS_H

#include "track.h"

__host__
void printTracks( adept::BlockData<track>* trackBlock,       //     adept::BlockData<track>*
                  bool          verbose = false,  // include info on navigation state ?
                  int           numTracks = -1 )  // only print those with index < numTracks
{
  cudaDeviceSynchronize();  // Sync to get new values before printing
   
  int numLive = trackBlock->GetNused() ;
  int lastTrack= numLive + trackBlock->GetNholes();
  numTracks = ( numTracks == -1 ) ? lastTrack : min( numTracks, numLive );
  if( numTracks != numLive ){
     std::cout << " Printing " << numTracks << " out of " << numLive 
               << " live tracks." << std::endl;
  } 
  
  // std::cout << " TrackBlock addr= " << trackBlock   << " " << std::endl;
  int numPrinted=0;
  for( int i = 0; i<lastTrack ; i++)
  {
     track& trk = (*trackBlock)[i];
     if( trk.status == alive ) {
        if( ++numPrinted < numTracks )
           trk.print( i, verbose );
           
        if( trk.index == 0 )
           trk.index = trk.mother_index * 100 + i;
     }
  }
}
#endif
