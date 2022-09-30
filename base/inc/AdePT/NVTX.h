// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file NVTX.h
 * @brief Helper for annotating workflows with NVTX profiling information.
 */

#ifndef NVTX_H
#define NVTX_H

#if defined USE_NVTX

#include "nvToolsExt.h"

#include <numeric>
#include <array>
#include <cstdint>
#include <string>

class NVTXTracer {
  static constexpr std::array<uint32_t, 7> _colours = {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff,
                                                       0xff00ffff, 0xffff0000, 0xffffffff};
  std::string _name;
  nvtxRangeId_t _id;
  std::array<unsigned long, 10> _lastOccups{};
  decltype(_lastOccups)::iterator _occupIt = _lastOccups.begin();

public:
  NVTXTracer(const std::string &name) { setTag(name, true); }
  ~NVTXTracer() { nvtxRangeEnd(_id); }

  void setTag(const std::string& name, bool first = false)
  {
    if (_name == name) return;

    _name = name;

    if (!first)
      nvtxRangeEnd(_id);
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version               = NVTX_VERSION;
    eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType             = NVTX_COLOR_ARGB;
    eventAttrib.color                 = nextColour();
    eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii         = _name.c_str();
    _id                               = nvtxRangeStartEx(&eventAttrib);
  }

  void setOccupancy(unsigned long occupancy)
  {
    // Require the occupancy to be larger than the majority of previous iterations to call it rising
    const bool rising =
        2 * std::count_if(_lastOccups.begin(), _lastOccups.end(), [occupancy](auto elm) { return occupancy > elm + 1; }) >
        _lastOccups.size();

    if (rising) {
      setTag("occupancy rising");
    } else if (_name == "occupancy rising") {
      setTag("peak occupancy (" + std::to_string(occupancy) + " in-flight)");
    } else {
      setTag("occupancy falling");
    }

    *_occupIt = occupancy;
    if (++_occupIt == _lastOccups.end()) _occupIt = _lastOccups.begin();
  }

  static uint32_t nextColour()
  {
    static int colour = 0;
    auto idx          = colour++ % _colours.size();
    return _colours[idx];
  }
};

#else

class NVTXTracer {
public:
  NVTXTracer(const char *) {}
  void setTag(const char *) {}
  void setOccupancy(unsigned long) {}
};

#endif

#endif
