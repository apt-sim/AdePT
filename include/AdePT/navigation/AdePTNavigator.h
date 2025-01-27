// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file AdePTNavigator.h
 * @brief Top switch for different navigators
 */

#ifndef ADEPT_NAVIGATOR_H_
#define ADEPT_NAVIGATOR_H_

#include <AdePT/navigation/BVHNavigator.h>
#ifdef ADEPT_USE_SURF
#include <AdePT/navigation/SurfNavigator.h>
#endif

// inline namespace COPCORE_IMPL {
#ifdef ADEPT_USE_SURF
#ifdef ADEPT_USE_SURF_SINGLE
using AdePTNavigator = SurfNavigator<float>;
#else
using AdePTNavigator = SurfNavigator<double>;
#endif
#else
using AdePTNavigator = BVHNavigator;
#endif
// } // End namespace COPCORE_IMPL
#endif // ADEPT_NAVIGATOR_H_
