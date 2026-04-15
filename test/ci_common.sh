#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

select_devadept_lcg_setup() {
  local requested_setup=${1:-}

  if [[ -n "${requested_setup}" ]]; then
    [[ -f "${requested_setup}" ]] || {
      printf '[ci-common] ERROR: LCG setup script not found: %s\n' "${requested_setup}" >&2
      return 1
    }
    printf '%s\n' "${requested_setup}"
    return 0
  fi

  if [[ -f /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el10-gcc13-opt/setup.sh ]]; then
    printf '%s\n' /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el10-gcc13-opt/setup.sh
    return 0
  fi

  if [[ -f /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el9-gcc13-opt/setup.sh ]]; then
    printf '%s\n' /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el9-gcc13-opt/setup.sh
    return 0
  fi

  printf '[ci-common] ERROR: No devAdePT gcc13 LCG view found for el10 or el9\n' >&2
  return 1
}
