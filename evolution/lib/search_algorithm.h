// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_SEARCH_ALGORITHM_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_SEARCH_ALGORITHM_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "evolution/lib/current_time.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/search_algorithm_spec.pb.h"
#include "evolution/lib/search_algorithm_stats.pb.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

// The start time of the search when no search restarts have taken place.
constexpr Integer kBeginningOfExperiment = -1;  // NOLINT

// Signal that no restart is queued at the moment.
constexpr Integer kNoQueuedRestart = -1;  // NOLINT

enum class ExchangeStatus { UNKNOWN, OK, REJECTED };

// Interface for algorithms for the PopulationServer.
class ThreadSafeSearchAlgorithmInterface {
 public:
  virtual ~ThreadSafeSearchAlgorithmInterface() {}

  // Used to checkpoint the state of the algorithm.
  virtual std::string Serialize(
      // stats can be nullptr if no stats are needed.
      SearchAlgorithmStats* stats) = 0;
  virtual void Deserialize(const std::string& serialized) = 0;

  // Exchanges models.
  virtual ExchangeStatus Exchange(
      const std::vector<Individual>& puts,
      // Number of individuals to return. If -1, returns the whole population
      // (it is recommended that the whole population be returned sorted with
      // the best algorithms at the beginning. This is in case the server has
      // to trim the snapshot to reduce space on disk). It is not guaranteed
      // that the search algorithm will return `num_to_get` algorithms; it may
      // return fewer if not possible (e.g. returns none if the population is
      // empty).
      Integer num_to_get,
      // gets can be nullptr if num_to_get==0.
      std::vector<Individual>* gets,
      // stats can be nullptr if no stats are needed.
      SearchAlgorithmStats* stats,
      // Will be filled with individuals that have been killed as a result of
      // this exchange.
      std::vector<Individual>* kills) = 0;

  // Returns a report to include in the logs. Details are up to the algorithm.
  virtual std::string Report() const = 0;
};

// Interface for algorithms that are thread-compatible but not thread-safe.
// It is easier to code this because there's no need to implement any locking
// and because there is no need to implement a mechanism to keep track of
// progress. They can be made thread-safe by wrapping them in a
// LockedPopulationServerAlgorithm.
class ThreadCompatibleSearchAlgorithmInterface {
 public:
  virtual ~ThreadCompatibleSearchAlgorithmInterface() {}

  // Used to checkpoint the state of the algorithm.
  virtual std::string Serialize() = 0;
  virtual void Deserialize(const std::string& serialized) = 0;

  // Exchanges models. See ThreadSafeSearchAlgorithmInterface::Exchange for more
  // details.
  virtual void Exchange(const std::vector<Individual>& puts, Integer num_to_get,
                        std::vector<Individual>* gets,
                        std::vector<Individual>* kills) = 0;

  // Returns a report to include in the logs. Details are up to the algorithm.
  virtual std::string Report() const = 0;

  // Restarts this search algorithm, removing all information prior to the
  // given timestamp (`start_nanos`). In particular, removes all
  // individuals with an origin older than start_nanos. Required of algorithms
  // that support restart.
  virtual void SearchRestart(Integer start_nanos);
};

// Turns a thread-compatible algorithm into a thread-safe algorithm with a
// single lock. Allows for simple implementations of the algorithms but may be
// inefficient.
class LockedPopulationServerAlgorithm final
    : public ThreadSafeSearchAlgorithmInterface {
 public:
  LockedPopulationServerAlgorithm(
      std::unique_ptr<ThreadCompatibleSearchAlgorithmInterface>
          thread_compatible_algorithm,
      const SearchAlgorithmSpec& spec,
      CurrentTimeInterface* mock_current_time = nullptr);

  // See base class for method descriptions.
  std::string Serialize(SearchAlgorithmStats* stats) override;
  void Deserialize(const std::string& serialized) override;
  ExchangeStatus Exchange(const std::vector<Individual>& puts,
                          Integer num_to_get, std::vector<Individual>* gets,
                          SearchAlgorithmStats* stats,
                          std::vector<Individual>* kills) override;
  std::string Report() const override;

 private:
  void AccumulateProgress(const std::vector<Individual> individuals,
                          std::vector<double>* progress)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void TrackImprovement(const std::vector<Individual> individuals)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void ResetImprovement() ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);
  SearchAlgorithmStats GetStats() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);
  bool IsOutOfSync(const Individual& individual) const;

  // Looks for a restart signal and queues the restart if found.
  void QueueSearchRestartIfReceivedSignal(absl::Span<const Individual> puts)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  void QueueSearchRestartIfConditionMet() ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Indicates whether the restart condition has been met by this server.
  bool IsSearchRestartConditionMet(const SearchRestartIfNoImprovementSpec& spec)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);
  bool IsSearchRestartConditionMet(const SearchRestartHurdle& spec)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Indicates that a global search restart should take place as soon as
  // possible.
  void QueueSearchRestart(Integer restart_nanos)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  std::vector<Individual> PurgeIndividualsFromBeforeSearchRestart(
      absl::Span<const Individual> puts) ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Propagates the search restart signal so it can reach other workers.
  void SendSearchRestartSignal(std::vector<Individual>* gets)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  void ClearSearchRestartSignal(std::vector<Individual>* purged_puts);

  // Trigger a global search restart if one has been queued.
  void SearchRestartIfQueued() ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  const std::unique_ptr<CurrentTimeInterface> owned_current_time_;
  CurrentTimeInterface* const current_time_;
  const SearchAlgorithmSpec spec_;
  const Integer process_start_nanos_;
  mutable absl::Mutex lock_;
  std::unique_ptr<ThreadCompatibleSearchAlgorithmInterface> algorithm_
      ABSL_GUARDED_BY(lock_);
  std::vector<double> experiment_progress_ ABSL_GUARDED_BY(lock_);
  std::vector<double> out_of_sync_progress_ ABSL_GUARDED_BY(lock_);

  // The maximum fitnesses seen since the last search restart.
  std::vector<double> max_fitnesses_ ABSL_GUARDED_BY(lock_);

  // The experiment progress when the last increase in fitness occurred. The
  // indexes for the relevant fitness and the relevant experiment progress are
  // set by the SearchRestartSpec. Used for determining if should restart the
  // search.
  double last_fitness_improvement_at_ ABSL_GUARDED_BY(lock_);

  // The time of the last restart or kBeginningOfExperiment if no restarts.
  Integer search_restart_nanos_ ABSL_GUARDED_BY(lock_);

  // The experiment progress at which the last restart occurred or empty if no
  // restarts.
  std::vector<double> search_restart_experiment_progress_;

  // The timestamp for a queued restart. When the restart is underway, this will
  // become the `search_restart_nanos_`. If no restart is queued, this is
  // kNoQueuedRestart.
  Integer queued_search_restart_nanos_ ABSL_GUARDED_BY(lock_);
};

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_SEARCH_ALGORITHM_H_
