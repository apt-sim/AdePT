// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Track debug information
struct TrackDebug {
  int event_id{-1};
  uint64_t track_id{0};
  long min_step{0};
  long max_step{10000};
  bool active{false};
};

__device__ TrackDebug gTrackDebug;
