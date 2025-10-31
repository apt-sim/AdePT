// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "AdePTSteppingAction.cuh"

#ifndef ADEPT_STEP_KIND
#  error "ADEPT_STEP_KIND is not defined (0=NONE, 1=CMS, 2=LHCb)"
#endif

namespace adept::SteppingAction {
#if   (ADEPT_STEP_KIND==1)
  using Action        = CMSAction;
#elif (ADEPT_STEP_KIND==2)
  using Action        = LHCbAction;
#else
  using Action        = NoAction;
#endif

using Params = typename Action::Params;

} // namespace adept::SteppingAction
