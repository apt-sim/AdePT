// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_PRECISION_HH
#define ADEPT_PRECISION_HH

#ifdef ADEPT_MIXED_PRECISION
    using rk_integration_t = float;
#else
    using rk_integration_t = double;
#endif

#endif