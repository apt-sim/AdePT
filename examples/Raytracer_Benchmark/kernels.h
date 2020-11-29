// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/BlockData.h>

inline namespace COPCORE_IMPL {

// Alocate slots for the BlockData
VECCORE_ATT_HOST_DEVICE
void generateRays(int id, adept::BlockData<Ray_t> *rays)
{
  auto ray = rays->NextElement();
  if (!ray) COPCORE_EXCEPTION("generateRays: Not enough space for rays");
}

COPCORE_CALLABLE_FUNC(generateRays)

} // End namespace COPCORE_IMPL
