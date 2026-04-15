#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -eq 0 ]; then
  exit 0
fi

if command -v clang-format-19 >/dev/null 2>&1; then
  exec clang-format-19 -style=file -i "$@"
fi

if command -v clang-format >/dev/null 2>&1; then
  version_output=$(clang-format --version || true)
  if printf '%s\n' "${version_output}" | grep -Eq 'version 19([.][0-9]+)*([[:space:]]|$)'; then
    exec clang-format -style=file -i "$@"
  fi
fi

printf '%s\n' "clang-format 19 is required. Install clang-format-19 or provide clang-format version 19 on PATH." >&2
exit 1
