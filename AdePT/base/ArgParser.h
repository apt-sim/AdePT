// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CopCore/Global.h>

#include <algorithm>
#include <iostream>

inline namespace COPCORE_IMPL {
double getDoubleOpt(char **begin, char **end, const std::string &option, double defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    double ret;
    sscanf(*itr, "%lf", &ret);
    return ret;
  }
  std::cout << "INFO: using default " << defaultval << " for option " << option << "\n";
  return defaultval;
}

double getIntOpt(char **begin, char **end, const std::string &option, int defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    int ret;
    sscanf(*itr, "%d", &ret);
    return ret;
  }
  std::cout << "INFO: using default " << defaultval << " for option " << option << "\n";
  return defaultval;
}

bool getBoolOpt(char **begin, char **end, const std::string &option, bool defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    bool ret =
        !((*(itr[0]) == 'n') || (*(itr[0]) == 'N') || (*(itr[0]) == 'f') || (*(itr[0]) == 'F') || (*(itr[0]) == '0'));
    return ret;
  }
  std::cout << "INFO: using default " << defaultval << " for option " << option << "\n";
  return defaultval;
}

std::string getStringOpt(char **begin, char **end, const std::string &option, const std::string &defaultval)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    std::string ret(*itr);
    std::cout << " From getStringOpt: ret=<" << ret << ">\n";
    return ret;
  }
  std::cout << "INFO: using default " << defaultval << " for option " << option << "\n";
  return defaultval;
}

#define OPTION_INT(name, defaultval) int name = getIntOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_DOUBLE(name, defaultval) double name = getDoubleOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_BOOL(name, defaultval) bool name = getBoolOpt(argv, argc + argv, "-" #name, defaultval)
#define OPTION_STRING(name, defaultval) std::string name = getStringOpt(argv, argc + argv, "-" #name, defaultval)
} // End namespace COPCORE_IMPL
