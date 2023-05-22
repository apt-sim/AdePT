// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file BenchmarkManager.h
 * @brief Benchmarking utilities: Timers, accumulators and formatted output.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#if defined BENCHMARK

// Entry management
#include <map>
#include <string>
// Timing
#include <chrono>
// Output
#include <iostream>
#include <fstream>
#include <iomanip>
// Output files and directories
#include <filesystem>

// The clock to use for measurements
#define CLOCK std::chrono::system_clock
// The precission with which we will provide the results
#define PRECISSION std::chrono::microseconds

/**
 * @brief A type containing a timestamp, accumulated time, and whether the timer is running
 * @details The timestamp marks the start of the current measuring period, the accumulated duration
 * represents the total time measured under the same tag. It can also be set and added to manually
 * to function as an accumulator.
 */
struct TimeInfo {
  std::chrono::time_point<CLOCK> start;                                 ///< Start timestamp
  std::chrono::duration<CLOCK::rep, CLOCK::period> accumulatedDuration; ///< Total time measured under the same tag
  bool counting;                                                        ///< Whether the timer is running
};

template <class TTag>
class BenchmarkManager {

private:
  std::map<TTag, TimeInfo> fTimers;           ///< Maps a tag to a TimeInfo struct
  const std::string fOutputDir = "benchmark"; ///< Output directory

public:
  BenchmarkManager(){};

  ~BenchmarkManager(){};

  /** @brief Sets a timestamp associated to this timer tag */
  void timerStart(TTag tag)
  {
    fTimers[tag].counting = true;
    fTimers[tag].start    = CLOCK::now();
  }

  /** @brief Compares the current time with the start of the timer and adds to the accumulated duration */
  void timerStop(TTag tag)
  {
    if (fTimers[tag].counting) {
      fTimers[tag].counting = false;
      fTimers[tag].accumulatedDuration += (CLOCK::now() - fTimers[tag].start);
    }
  }

  /** @brief Returns the accumulated duration for the specified tag in seconds */
  double getDurationSeconds(TTag tag)
  {
    return (float)(std::chrono::duration_cast<PRECISSION>(fTimers[tag].accumulatedDuration)).count() /
           PRECISSION::period::den;
  }

  /** @brief Sets the accumulated duration in seconds for the specified tag */
  void setDurationSeconds(TTag tag, double duration)
  {
    if (fTimers.find(tag) == fTimers.end()) {
      // Initialize if the timer didn't exist
      fTimers[tag] = TimeInfo{
          CLOCK::now(), std::chrono::duration_cast<CLOCK::duration>(std::chrono::duration<double>(duration)), false};
    } else {
      fTimers[tag].accumulatedDuration =
          std::chrono::duration_cast<CLOCK::duration>(std::chrono::duration<double>(duration));
    }
  }

  /** @brief Adds to the accumulated duration for the specified tag */
  void addDurationSeconds(TTag tag, double duration)
  {
    if (fTimers.find(tag) == fTimers.end()) {
      // Initialize if the timer didn't exist
      fTimers[tag] = TimeInfo{CLOCK::now(), std::chrono::seconds(0), false};
    }
    fTimers[tag].accumulatedDuration +=
        std::chrono::duration_cast<CLOCK::duration>(std::chrono::duration<double>(duration));
  }

  /** @brief Empties the timer map */
  void reset() { fTimers.clear(); }

  /** @brief Removes a timer from the map */
  void removeTimer(TTag tag) { fTimers.erase(tag); }

  /** @brief Checks if an timer is in the map */
  bool hasTimer(TTag tag) const { return fTimers.find(tag) != fTimers.end(); }

  /** @brief Export a CSV file with the timer names as labels and the accumulated time for each */
  // If the file is not empty, write only the times
  void exportCSV(std::string filename = "benchmark")
  {
    std::string path = fOutputDir + "/" + filename + ".csv";

    // Create the output directory if it doesn't exist.
    std::filesystem::create_directory(fOutputDir);

    bool first_write = !std::filesystem::exists(path);

    std::ofstream output_file;
    output_file.open(path, std::ofstream::app);

    // Write the header only the first time
    if (first_write) {
      for (auto iter = fTimers.begin(); iter != fTimers.end(); ++iter) {
        output_file << iter->first;
        if (std::next(iter) != fTimers.end()) output_file << ", ";
      }
      output_file << std::endl;
    }

    // Print 6 decimal digits
    output_file << std::setprecision(6) << std::fixed;
    for (auto iter = fTimers.begin(); iter != fTimers.end(); ++iter) {
      output_file << getDurationSeconds(iter->first);
      if (std::next(iter) != fTimers.end()) output_file << ", ";
    }
    output_file << std::endl;

    std::cout << "BENCHMARK: Results saved to benchmark/" << filename << ".csv" << std::endl;
  }
};

#else

template <class TTag>
class BenchmarkManager {
public:
  BenchmarkManager(){};
  ~BenchmarkManager(){};
  void timerStart(TTag tag) {}
  void timerStop(TTag tag) {}
  double getDurationSeconds(TTag tag) { return 0; }
  void addDurationSeconds(TTag tag, double duration) {}
  void reset() {}
  void removeTimer(TTag tag) {}
  bool hasTimer(TTag tag) { return false; }
  void exportCSV(TTag filename) {}
};

#endif

#endif