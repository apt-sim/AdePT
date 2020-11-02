// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

//#include <iostream>
//#include <cassert>
#include <AdePT/BlockData.h>

/** @brief Data structures */
struct MyTrack {
  int index{0};
  float energy{0};
};

struct MyHit {
  float edep{0};
};

// portable kernel functions have to reside in a backend-dependent inline namespace to avoid symbol duplication
// when building executables/libraries running same functions on different backends
inline namespace COPCORE_IMPL {

/** @brief Generate a number of primaries */
VECCORE_ATT_HOST_DEVICE
void generateAndStorePrimary(int id, adept::BlockData<MyTrack> *tracks)
{
  auto track = tracks->NextElement();
  if (!track) COPCORE_EXCEPTION("generateAndStorePrimary: Not enough space for tracks");

  track->index  = id;
  track->energy = 100.;
}

// Mandatory callable function decoration
COPCORE_CALLABLE_FUNC(generateAndStorePrimary)

/** @brief This demonstrates the use of a kernel function in a namespace */
namespace devfunc {
VECCORE_ATT_HOST_DEVICE
void selectTrack(int id, adept::BlockData<MyTrack> *tracks, adept::mpmc_bounded_queue<int> *queue)
{
  auto track    = (*tracks)[id];
  bool selected = (track.index % 1000 == 0);
  if (selected) queue->enqueue(id);
}
COPCORE_CALLABLE_FUNC(selectTrack)
} // end namespace devfunc

VECCORE_ATT_HOST_DEVICE
void processTrack(MyTrack const &track, MyHit &hit)
{
  hit.edep += 0.1 * track.energy;
}
COPCORE_CALLABLE_FUNC(processTrack)

VECCORE_ATT_HOST_DEVICE
void updateTrack(MyTrack &track)
{
  track.energy -= 0.1 * track.energy;
}
COPCORE_CALLABLE_FUNC(updateTrack)

} // End namespace COPCORE_IMPL
