// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/// Minimal JSON output for the geometry validation files (host only).

#pragma once

#include <cstdio>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>

namespace adept::validation_json {

/// JSON string literal.
inline std::string Str(std::string const &text)
{
  std::string out = "\"";
  for (char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char buffer[8];
      std::snprintf(buffer, sizeof buffer, "\\u%04x", c);
      out += buffer;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

/// JSON number with full double precision (non-finite values are written as null).
inline std::string Num(double value)
{
  if (!(value == value) || value > std::numeric_limits<double>::max() || value < -std::numeric_limits<double>::max())
    return "null";
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

/// JSON array of the first `n` values of `values`, formatted by `format`.
template <typename T, typename F>
std::string Array(T const &values, std::size_t n, F &&format)
{
  std::string out = "[";
  for (std::size_t i = 0; i < n; ++i)
    out += (i ? "," : "") + format(values[i]);
  return out + "]";
}

template <typename T>
std::string Array3(T const *values)
{
  return Array(values, 3, [](T v) { return Num(v); });
}

} // namespace adept::validation_json
