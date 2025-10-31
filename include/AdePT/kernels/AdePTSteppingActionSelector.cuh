// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "AdePTSteppingAction.cuh"

#ifndef ADEPT_STEPACTION_TYPE
#error "ADEPT_STEPACTION_TYPE is not defined (0=NONE, 1=CMS, 2=LHCb)"
#endif

namespace adept::SteppingAction {
#if (ADEPT_STEPACTION_TYPE == 1)
using Action = CMSAction;
#elif (ADEPT_STEPACTION_TYPE == 2)
using Action = LHCbAction;
#else
using Action = NoAction;
#endif

using Params = typename Action::Params;

} // namespace adept::SteppingAction
