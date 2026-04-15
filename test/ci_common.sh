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

  if [[ -f /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el9-gcc13-opt/setup.sh ]]; then
    printf '%s\n' /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el9-gcc13-opt/setup.sh
    return 0
  fi

  printf '[ci-common] ERROR: No devAdePT gcc13 EL9 LCG view found\n' >&2
  return 1
}

normalize_lcg_cuda_env() {
  local toolkit_root=${CUDA_TOOLKIT_ROOT:-${CUDAToolkit_ROOT:-}}

  if [[ -z "${toolkit_root}" ]]; then
    printf '[ci-common] ERROR: CUDA toolkit root is not set after sourcing the LCG setup\n' >&2
    return 1
  fi

  if [[ ! -x "${toolkit_root}/bin/nvcc" ]]; then
    printf '[ci-common] ERROR: nvcc not found at %s/bin/nvcc\n' "${toolkit_root}" >&2
    return 1
  fi

  export CUDA_TOOLKIT_ROOT="${toolkit_root}"
  export CUDAToolkit_ROOT="${toolkit_root}"
  export CUDA_HOME="${toolkit_root}"
  export CUDACXX="${toolkit_root}/bin/nvcc"
  export PATH="${toolkit_root}/bin:${PATH}"
}
