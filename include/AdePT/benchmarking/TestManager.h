// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file TestManager.h
 * @brief Benchmarking utilities: Timers, accumulators and formatted output.
 */

#ifndef TEST_H
#define TEST_H

// The clock to use for measurements
#define CLOCK std::chrono::system_clock
// The precision with which we will provide the results
#define PRECISION std::chrono::microseconds

/**
 * @brief A type containing a timestamp, accumulated time, and whether the timer is running
 * @details The timestamp marks the start of the current measuring period, the accumulated duration
 * represents the total time measured under the same tag.
 */

// #if defined TEST

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

struct TimeInfo {
  std::chrono::time_point<CLOCK> start;                                 ///< Start timestamp
  std::chrono::duration<CLOCK::rep, CLOCK::period> accumulatedDuration; ///< Total time measured under the same tag
  bool counting;                                                        ///< Whether the timer is running
};

template <class TTag>
class TestManager {

private:
  std::map<TTag, TimeInfo> fTimers;     ///< Maps a tag to a TimeInfo struct
  std::map<TTag, double> fAccumulators; ///< Maps a tag to a double accumulator
  std::string fOutputDir;               ///< Output directory
  std::string fOutputFilename;          ///< Output filename
  inline static std::mutex fWriteMutex;        ///< Mutex for writing to file

public:
  TestManager(){};

  ~TestManager(){};

  /** @brief Sets a timestamp associated to this timer tag */
  void timerStart(TTag tag)
  {
    auto &tagtimer    = fTimers[tag]; // Avoid counting the overhead from map access
    tagtimer.counting = true;
    tagtimer.start    = CLOCK::now();
  }

  /** @brief Compares the current time with the start of the timer and adds to the accumulated duration */
  void timerStop(TTag tag)
  {
    if (fTimers[tag].counting) {
      auto stopTimestamp = CLOCK::now(); // Avoid counting the overhead from map access
      auto &tagtimer     = fTimers[tag];
      tagtimer.counting  = false;
      tagtimer.accumulatedDuration += stopTimestamp - tagtimer.start;
    }
  }

  /** @brief Returns the accumulated duration for the specified tag in seconds */
  double getDurationSeconds(TTag tag)
  {
    return (double)(std::chrono::duration_cast<PRECISION>(fTimers[tag].accumulatedDuration)).count() /
           PRECISION::period::den;
  }

  /** @brief Sets the accumulator value for the specified tag */
  void setAccumulator(TTag tag, double value) { fAccumulators[tag] = value; }

  /** @brief Returns the accumulator value for the specified tag */
  double getAccumulator(TTag tag) { return fAccumulators[tag]; }

  /** @brief Adds to the accumulator for the specified tag */
  void addToAccumulator(TTag tag, double value)
  {
    if (fAccumulators.find(tag) == fAccumulators.end()) {
      // Initialize if the accumulator didn't exist
      fAccumulators[tag] = value;
    } else {
      fAccumulators[tag] += value;
    }
  }

  /** @brief Empties the maps */
  void reset()
  {
    fTimers.clear();
    fAccumulators.clear();
  }

  /** @brief Removes a timer from the map */
  void removeTimer(TTag tag) { fTimers.erase(tag); }

  /** @brief Checks if an timer is in the map */
  bool hasTimer(TTag tag) const { return fTimers.find(tag) != fTimers.end(); }

  /** @brief Removes an accumulator from the map */
  void removeAccumulator(TTag tag) { fAccumulators.erase(tag); }

  /** @brief Checks if an accumulator is in the map */
  bool hasAccumulator(TTag tag) const { return fAccumulators.find(tag) != fAccumulators.end(); }

  /** @brief Returns the timer map */
  std::map<TTag, TimeInfo> *getTimers() { return &fTimers; }

  /** @brief Returns the accumulator map */
  std::map<TTag, double> *getAccumulators() { return &fAccumulators; }

  /** @brief Sets the output directory variable */
  void setOutputDirectory(std::string aOutputDir) { fOutputDir = aOutputDir; }

  /** @brief Returns the output directory */
  std::string getOutputDirectory() { return fOutputDir; }

  /** @brief Sets the output filename variable */
  void setOutputFilename(std::string aOutputFilename) { fOutputFilename = aOutputFilename; }

  /** @brief Returns the output filename */
  std::string getOutputFilename() { return fOutputFilename; }

  /** @brief Export a CSV file with the timer names as labels and the accumulated time for each */
  // If the file is not empty, write only the times
  void exportCSV(bool overwrite = true)
  {
    std::string aOutputDir;
    std::string aOutputFilename;
    fOutputDir.empty() ? aOutputDir = "benchmark" : aOutputDir = fOutputDir;
    fOutputFilename.empty() ? aOutputFilename = "benchmark" : aOutputFilename = fOutputFilename;

    std::string path = aOutputDir + "/" + aOutputFilename + ".csv";

    // Create the output directory if it doesn't exist.
    std::filesystem::create_directory(aOutputDir);

    bool first_write = !std::filesystem::exists(path);

    std::ofstream output_file;

    // Acquire Mutex
    std::lock_guard<std::mutex> lock(fWriteMutex);

    output_file.open(path, overwrite ? std::ofstream::trunc : std::ofstream::app);

    // Write the header only the first time
    if (first_write || overwrite) {
      // Timers
      for (auto iter = fTimers.begin(); iter != fTimers.end(); ++iter) {
        output_file << iter->first;
        if (std::next(iter) != fTimers.end() || !fAccumulators.empty()) output_file << ", ";
      }
      // Accumulators
      for (auto iter = fAccumulators.begin(); iter != fAccumulators.end(); ++iter) {
        output_file << iter->first;
        if (std::next(iter) != fAccumulators.end()) output_file << ", ";
      }
      output_file << std::endl;
    }

    // Print 6 decimal digits
    output_file << std::setprecision(6) << std::fixed;
    // Timers
    for (auto iter = fTimers.begin(); iter != fTimers.end(); ++iter) {
      output_file << getDurationSeconds(iter->first);
      if (std::next(iter) != fTimers.end() || !fAccumulators.empty()) output_file << ", ";
    }
    // Accumulators
    for (auto iter = fAccumulators.begin(); iter != fAccumulators.end(); ++iter) {
      output_file << iter->second;
      if (std::next(iter) != fAccumulators.end()) output_file << ", ";
    }
    output_file << std::endl;

    std::cout << "TEST: Results saved to: " << aOutputDir << "/" << aOutputFilename << ".csv" << std::endl;
  }
};

/*
#else

template <class TTag>
class TestManager {
public:
  TestManager(){};
  ~TestManager(){};
  void timerStart(TTag tag) {}
  void timerStop(TTag tag) {}
  double getDurationSeconds(TTag tag) { return 0; }
  void setAccumulator(TTag tag, double value) {}
  double getAccumulator(TTag tag) { return 0; }
  void addToAccumulator(TTag tag, double duration) {}
  void reset() {}
  void removeTimer(TTag tag) {}
  bool hasTimer(TTag tag) { return false; }
  void removeAccumulator(TTag tag) {}
  bool hasAccumulator(TTag tag) { return 0; }
  std::map<TTag, TimeInfo>* getTimers() { return 0; }
  std::map<TTag, double>* getAccumulators() { return 0;}
  void setOutputDirectory(std::string aOutputDir) {}
  std::string getOutputDirectory(){ return ""; }
  void setOutputFilename(std::string aOutputFilename) {}
  std::string getOutputFilename(){ return ""; }
  void exportCSV() {}
};

#endif
*/

#endif
