// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file TestManagerStore.h
 * @brief Functionality to store TestManager snapshots at different points during the execution
 */

#ifndef TEST_AGGREGATOR_H
#define TEST_AGGREGATOR_H

//#if defined TEST

#include <mutex>
#include <vector>
#include <map>
#include <AdePT/benchmarking/TestManager.h>

template <class TTag>
class TestManagerStore {

private:
  static TestManagerStore *fInstance; ///< This class is a Singleton
  static std::mutex fInitMutex;            ///< Mutex for initialization
  static std::mutex fAccessMutex;          ///< Mutex for adding BenchmarkStates
  std::vector<std::map<TTag, double>>
      *fBenchmarkStates; ///< This vector holds one map with the information from each TestManager

protected:
  TestManagerStore() { fBenchmarkStates = new std::vector<std::map<int, double>>(); };
  ~TestManagerStore(){};

public:
  /** @brief Thread-safe singleton accessor */
  static TestManagerStore<TTag> *GetInstance()
  {
    std::lock_guard<std::mutex> lock(fInitMutex);
    if (fInstance == nullptr) {
      fInstance = new TestManagerStore();
    }
    return fInstance;
  }

  /** @brief Save the contents of the timers and accumulators of the given TestManager */
  void RecordState(TestManager<TTag> *aTestManager)
  {
    std::map<TTag, double> aBenchmarkState;
    // Do not make a distinction between timers and accumulators
    for (auto iter = aTestManager->getTimers()->begin(); iter != aTestManager->getTimers()->end(); ++iter) {
      aBenchmarkState[iter->first] = aTestManager->getDurationSeconds(iter->first);
    }
    for (auto iter = aTestManager->getAccumulators()->begin(); iter != aTestManager->getAccumulators()->end();
         ++iter) {
      aBenchmarkState[iter->first] = iter->second;
    }
    // Make sure only one thread is adding a State at a time
    std::lock_guard<std::mutex> lock(fAccessMutex);
    fBenchmarkStates->push_back(aBenchmarkState);
  }

  /** @brief Returns the stored BenchmarkStates */
  std::vector<std::map<TTag, double>> *GetStates() { return fBenchmarkStates; }

  /** @brief Clears the stored BenchmarkStates */
  void Reset() { fBenchmarkStates->clear(); }
};

template <typename TTag>
TestManagerStore<TTag> *TestManagerStore<TTag>::fInstance{nullptr};
template <typename TTag>
std::mutex TestManagerStore<TTag>::fInitMutex;
template <typename TTag>
std::mutex TestManagerStore<TTag>::fAccessMutex;

#else

template <class TTag>
class TestManagerStore {
public:
  static TestManagerStore<TTag> *GetInstance() { return 0; }
  void RecordState(TestManager<TTag> *aTestManager) {}
  std::vector<std::map<TTag, double>> *GetStates() { return 0; }
  void Reset() {}
};

//#endif

#endif