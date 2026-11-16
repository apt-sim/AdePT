# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

# Run example1 with a handoff validation macro. Passes only if the job succeeds and prints the PASSED verdict.
execute_process(COMMAND ${EXAMPLE1} -m ${MACRO} OUTPUT_VARIABLE output ERROR_VARIABLE output RESULT_VARIABLE result)
message("${output}")
if(NOT result EQUAL 0)
  message(FATAL_ERROR "example1 failed: ${result}")
endif()
if(NOT output MATCHES "HandoffValidation: PASSED")
  message(FATAL_ERROR "no 'HandoffValidation: PASSED' verdict")
endif()
