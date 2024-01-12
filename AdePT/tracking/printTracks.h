// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef PRINT_TRACKS_H
#define PRINT_TRACKS_H

#include <AdePT/tracking/track.h>

__global__ void printTrack(adept::BlockData<track> *trackBlock, int itrack, bool verbose)
{
  track &trk = (*trackBlock)[itrack];
  trk.print(itrack, verbose);
}

__host__ void printTracks(adept::BlockData<track> *trackBlock, //     adept::BlockData<track>*
                          bool verbose  = false,               // include info on navigation state ?
                          int numTracks = -1)                  // only print those with index < numTracks
{
  // host version printing device-resident tracks
  int numLive   = trackBlock->GetNused();
  int lastTrack = numLive + trackBlock->GetNholes();
  numTracks     = (numTracks == -1) ? lastTrack : min(numTracks, numLive);

  int numPrinted = 0;
  for (int i = 0; i < lastTrack; i++) {
    track &trk = (*trackBlock)[i];
    if (trk.status == alive) {
      if (numPrinted++ < numTracks) printTrack<<<1, 1>>>(trackBlock, i, verbose);
    }
  }
  cudaDeviceSynchronize();
}
#endif
