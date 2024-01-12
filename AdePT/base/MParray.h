// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file MParray.h
 * @brief Multi-producer array for handling track indices
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_MPARRAY_H_
#define ADEPT_MPARRAY_H_

#include <AdePT/base/MParrayT.h>

namespace adept {
using MParray = MParrayT<int>;
} // End namespace adept

#endif // ADEPT_MPARRAY_H_
