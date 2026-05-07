// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ADEPT_MIXED_PRECISION
using rk_integration_t = float;
#else
using rk_integration_t = double;
#endif
