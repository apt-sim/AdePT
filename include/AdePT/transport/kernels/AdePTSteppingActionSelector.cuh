// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <AdePT/AdePTBuildFlags.hh>
#include "AdePTSteppingAction.cuh"

#include <AdePT/transport/config/SteppingActionParams.hh>

#include <type_traits>

#ifndef ADEPT_STEPACTION_TYPE
#error "ADEPT_STEPACTION_TYPE is not defined (0=NONE, 1=CMS, 2=LHCb, 3=ATLAS)"
#endif

namespace adept::SteppingAction {
#if (ADEPT_STEPACTION_TYPE == 1)
using Action = CMSAction;
#elif (ADEPT_STEPACTION_TYPE == 2)
using Action = LHCbAction;
#elif (ADEPT_STEPACTION_TYPE == 3)
using Action = ATLASAction;
#else
using Action = NoAction;
#endif

using Params = typename Action::Params;
static_assert(std::is_same_v<Params, adept::SteppingActionParamsT>,
              "the host-visible SteppingActionParamsT must select the parameters of the selected stepping action");

} // namespace adept::SteppingAction
