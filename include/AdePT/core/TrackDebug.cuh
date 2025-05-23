// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRACK_DEBUG_CUH
#define ADEPT_TRACK_DEBUG_CUH

// Track debug information
struct TrackDebug {
  uint64_t track_id{0};
  long min_step{0};
  long max_step{10000};
  bool active{false};
};

__device__ TrackDebug gTrackDebug;

#endif
