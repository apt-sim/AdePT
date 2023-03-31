// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_INTEGRATION_COMMONSTRUCT_H
#define ADEPT_INTEGRATION_COMMONSTRUCT_H

#include <vector>

// Common data structures used by the integration with Geant4
namespace adeptint {

/// @brief Auxiliary logical volume data. This stores in the same structure the material-cuts couple index,
/// the sensitive volume handler index and the flag if the region is active for AdePT.
struct VolAuxData {
  int fSensIndex{-1}; ///< index of handler for sensitive volumes (-1 means non-sensitive)
  int fMCIndex{0};    ///< material-cut cuple index in G4HepEm
  int fGPUregion{0};  ///< GPU region index (currently 1 or 0, meaning tracked on GPU or not)
};

/// @brief Track data exchanged between Geant4 and AdePT
struct TrackData {
  double position[3];
  double direction[3];
  double energy{0};
  int pdg{0};

  TrackData() = default;
  TrackData(int pdg_id, double ene, double x, double y, double z, double dirx, double diry, double dirz)
      : position{x, y, z}, direction{dirx, diry, dirz}, energy{ene}, pdg{pdg_id}
  {
  }

  inline bool operator<(TrackData const &t)
  {
    if (pdg != t.pdg) return pdg < t.pdg;
    if (energy != t.energy) return energy < t.energy;
    if (position[0] != t.position[0]) return position[0] < t.position[0];
    if (position[1] != t.position[1]) return position[1] < t.position[1];
    if (position[2] != t.position[2]) return position[2] < t.position[2];
    if (direction[0] != t.direction[0]) return direction[0] < t.direction[0];
    if (direction[1] != t.direction[1]) return direction[1] < t.direction[1];
    if (direction[2] != t.direction[2]) return direction[2] < t.direction[2];
    return false;
  }
};

/// @brief Buffer holding input tracks to be transported on GPU and output tracks to be
/// re-injected in the Geant4 stack
struct TrackBuffer {
  std::vector<TrackData> toDevice; ///< Tracks to be transported on the device
  std::vector<TrackData> fromDevice_sorted; ///< Tracks from device sorted by energy
  TrackData *fromDevice{nullptr};  ///< Tracks coming from device to be transported on the CPU
  int eventId{-1};                 ///< Index of current transported event
  int startTrack{0};               ///< Track counter for the current event
  int numFromDevice{0};            ///< Number of tracks coming from device for the current transported buffer
  int nelectrons{0};               ///< Number of electrons in the input buffer
  int npositrons{0};               ///< Number of positrons in the input buffer
  int ngammas{0};                  ///< Number of gammas in the input buffer

  void Clear()
  {
    toDevice.clear();
    numFromDevice = 0;
    nelectrons = npositrons = ngammas = 0;
  }
};

} // end namespace adeptint
#endif
