// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ADEPT_ENABLE_NSYS_PROFILING) || defined(__DOXYGEN__)

#include <cerrno>
#include <cstdlib>
#include <limits>

namespace adept::transport::detail {

bool StartTransportProfilerCapture();
void StopTransportProfilerCapture();

inline bool GetBoolEnvVar(const char *name)
{
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return false;
  return !(value[0] == '0' && value[1] == '\0');
}

inline unsigned int GetUnsignedEnvVar(const char *name, unsigned int defaultValue = 0)
{
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0' || value[0] < '0' || value[0] > '9') return defaultValue;
  char *end         = nullptr;
  errno             = 0;
  const auto parsed = std::strtoul(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed > std::numeric_limits<unsigned int>::max()) {
    return defaultValue;
  }
  return static_cast<unsigned int>(parsed);
}

/**
 * @brief Scoped controller for optional CUDA profiler API capture in the transport loop.
 *
 * ScopedTransportProfiler reads the AdePT profiling environment variables once at
 * construction and starts/stops the CUDA profiler API capture as transport-loop
 * iterations advance. It is intended for `nsys --capture-range=cudaProfilerApi`
 * runs where initialization kernels should be skipped and only a representative
 * transport window should be captured.
 *
 * Runtime control is provided by `ADEPT_NSYS_CAPTURE_TRANSPORT`,
 * `ADEPT_NSYS_CAPTURE_START_AFTER_ITERATIONS`, and
 * `ADEPT_NSYS_CAPTURE_STOP_AFTER_ITERATIONS`. The iteration window is half-open:
 * `[start, stop)`.
 *
 * The object only stops a capture if it successfully started that capture
 * itself. AdePT currently has a single active transport loop; if multiple
 * transport loops are introduced, the shared capture state in the implementation
 * must be made thread-safe.
 */
class ScopedTransportProfiler {
public:
  /**
   * @brief Read the profiling environment and initialize the scoped controller.
   */
  ScopedTransportProfiler()
      : fRequested(GetBoolEnvVar("ADEPT_NSYS_CAPTURE_TRANSPORT")),
        fStartIteration(GetUnsignedEnvVar("ADEPT_NSYS_CAPTURE_START_AFTER_ITERATIONS")),
        fStopIteration(GetUnsignedEnvVar("ADEPT_NSYS_CAPTURE_STOP_AFTER_ITERATIONS"))
  {
  }

  /**
   * @brief Stop an active capture during stack unwinding or normal teardown.
   */
  ~ScopedTransportProfiler() { Stop(); }

  ScopedTransportProfiler(const ScopedTransportProfiler &)            = delete;
  ScopedTransportProfiler &operator=(const ScopedTransportProfiler &) = delete;

  /**
   * @brief Update the capture state before launching work for a transport iteration.
   *
   * @param iteration Zero-based transport-loop iteration. Capture starts before
   * this iteration when it is equal to the configured start value, and stops
   * before this iteration when it reaches the configured stop value.
   */
  void BeginIteration(unsigned int iteration)
  {
    // The configured range is half-open: [start, stop). The stop check runs
    // before launching the first non-captured iteration.
    if (fStopIteration > 0 && iteration >= fStopIteration) {
      if (fActive) Stop();
      fStopped = true;
    }
    if (fRequested && !fActive && !fStopped && iteration == fStartIteration) {
      fActive = StartTransportProfilerCapture();
    }
  }

  /**
   * @brief Stop the CUDA profiler capture if this object started it.
   */
  void Stop()
  {
    if (!fActive) return;
    StopTransportProfilerCapture();
    fActive = false;
  }

private:
  bool fRequested{false};
  bool fActive{false};
  bool fStopped{false};
  unsigned int fStartIteration{0};
  unsigned int fStopIteration{0};
};

} // namespace adept::transport::detail

#endif
