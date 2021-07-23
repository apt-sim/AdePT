// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file NVTX.h
 * @brief Helper for annotating workflows with NVTX profiling information.
 */

#ifndef NVTX_H
#define NVTX_H

#if defined USE_NVTX && !defined __CUDACC__

#include "nvToolsExt.h"

#include <array>
#include <cstdint>
#include <string>

class NVTXTracer {
  static constexpr std::array<uint32_t, 7> _colours = {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff,
                                                       0xff00ffff, 0xffff0000, 0xffffffff};
  std::string _name;
  nvtxRangeId_t _id;

public:
  NVTXTracer(const char *name) : _name(name)
  {
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version               = NVTX_VERSION;
    eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType             = NVTX_COLOR_ARGB;
    eventAttrib.color                 = nextColour();
    eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii         = name;
    _id                               = nvtxRangeStartEx(&eventAttrib);
  }
  ~NVTXTracer() { nvtxRangeEnd(_id); }

  __host__ void setTag(const char *name)
  {
    if (_name == name) return;

    _name = name;

    nvtxRangeEnd(_id);
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version               = NVTX_VERSION;
    eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType             = NVTX_COLOR_ARGB;
    eventAttrib.color                 = nextColour();
    eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii         = name;
    _id                               = nvtxRangeStartEx(&eventAttrib);
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
};

#endif

#endif
