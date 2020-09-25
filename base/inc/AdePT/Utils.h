// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Utils.h
 * @brief General utility functions.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_UTILS_H_
#define ADEPT_UTILS_H_

namespace adept {
namespace utils {
/**
 * @brief Rounds up a value to upper aligned version
 * @param value Value to round-up
 */
template <typename Type>
VECCORE_ATT_HOST_DEVICE static Type round_up_align(Type value, size_t padding)
{
  size_t remainder = ((size_t)value) % padding;
  if (remainder == 0) return value;
  return (value + padding - remainder);
}

} // End namespace utils
} // End namespace adept

#endif // ADEPT_UTILS_H_
