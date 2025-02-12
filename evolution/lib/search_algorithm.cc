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

#include "evolution/lib/search_algorithm.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "evolution/lib/current_time.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/mocking.h"
#include "evolution/lib/printing.h"
#include "evolution/lib/search_algorithm.pb.h"
#include "evolution/lib/search_algorithm_spec.pb.h"
#include "evolution/lib/search_algorithm_stats.pb.h"
#include "evolution/lib/search_algorithm_stats_util.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

using ::std::fill;
using ::std::numeric_limits;
using ::std::unique_ptr;
using ::std::vector;

void ThreadCompatibleSearchAlgorithmInterface::SearchRestart(
    Integer search_start_nanos) {
  LOG(FATAL) << "Search restart not implemented.";
}

LockedPopulationServerAlgorithm::LockedPopulationServerAlgorithm(
    unique_ptr<ThreadCompatibleSearchAlgorithmInterface>
        thread_compatible_algorithm,
    const SearchAlgorithmSpec& spec, CurrentTimeInterface* mock_current_time)
    : owned_current_time_(MakeOwnedIfNoMock<CurrentTime>(mock_current_time)),
      current_time_(GetPointerToMockOrOwned(mock_current_time,
                                            owned_current_time_.get())),
      spec_(spec),
      process_start_nanos_(current_time_->Nanos()),
      algorithm_(std::move(thread_compatible_algorithm)),
      last_fitness_improvement_at_(0.0),
      search_restart_nanos_(kBeginningOfExperiment),
      queued_search_restart_nanos_(kNoQueuedRestart) {}

std::string LockedPopulationServerAlgorithm::Serialize(
    SearchAlgorithmStats* stats) {
  absl::MutexLock lock(&lock_);
  if (stats != nullptr) {
    *stats = GetStats();
  }
  LockedSearchAlgorithmState state;
  state.set_thread_compatible_algorithm_state(algorithm_->Serialize());
  *state.mutable_stats() = GetStats();
  state.set_last_fitness_improvement_at(last_fitness_improvement_at_);

  state.set_search_restart_nanos(search_restart_nanos_);
  state.set_queued_search_restart_nanos(queued_search_restart_nanos_);
  for (double value : search_restart_experiment_progress_) {
    state.add_search_restart_experiment_progress(value);
  }
  return state.SerializeAsString();
}

void LockedPopulationServerAlgorithm::Deserialize(
    const std::string& serialized) {
  LockedSearchAlgorithmState state;
  CHECK(state.ParseFromString(serialized));
  absl::MutexLock lock(&lock_);
  algorithm_->Deserialize(state.thread_compatible_algorithm_state());
  experiment_progress_ =
      vector<double>(state.stats().experiment_progress().begin(),
                     state.stats().experiment_progress().end());
  out_of_sync_progress_ =
      vector<double>(state.stats().out_of_sync_progress().begin(),
                     state.stats().out_of_sync_progress().end());
  max_fitnesses_ = vector<double>(state.stats().max_fitnesses().begin(),
                                  state.stats().max_fitnesses().end());
  last_fitness_improvement_at_ = state.last_fitness_improvement_at();

  search_restart_nanos_ = state.search_restart_nanos();
  search_restart_experiment_progress_.clear();
  for (double value : state.search_restart_experiment_progress()) {
    search_restart_experiment_progress_.push_back(value);
  }
  CHECK_NE(search_restart_nanos_, 0);
  queued_search_restart_nanos_ = state.queued_search_restart_nanos();
}

ExchangeStatus LockedPopulationServerAlgorithm::Exchange(
    const vector<Individual>& puts, Integer num_to_get,
    vector<Individual>* gets, SearchAlgorithmStats* stats,
    vector<Individual>* kills) {
  absl::MutexLock lock(&lock_);

  // Decide whether updates are out of sync.
  bool out_of_sync = false;
  for (const Individual& individual : puts) {
    if (IsOutOfSync(individual)) {
      out_of_sync = true;
    }
  }
  if (out_of_sync) {
    AccumulateProgress(puts, &out_of_sync_progress_);
  }

  QueueSearchRestartIfReceivedSignal(puts);
  vector<Individual> purged_puts =
      PurgeIndividualsFromBeforeSearchRestart(puts);
  AccumulateProgress(purged_puts, &experiment_progress_);
  TrackImprovement(purged_puts);
  ClearSearchRestartSignal(&purged_puts);
  algorithm_->Exchange(purged_puts, num_to_get, gets, kills);
  QueueSearchRestartIfConditionMet();
  SearchRestartIfQueued();
  SendSearchRestartSignal(gets);

  if (stats != nullptr) {
    *stats = GetStats();
  }

  return ExchangeStatus::OK;
}

std::string LockedPopulationServerAlgorithm::Report() const {
  absl::MutexLock lock(&lock_);
  std::stringstream report;
  const double experiment_progress =
      ExperimentProgressOrZero(experiment_progress_, 0);
  const double delta_experiment_progress =
      experiment_progress - last_fitness_improvement_at_;
  report << "AlgorithmReport: " << "best=" << FitnessOrLowest(max_fitnesses_, 0)
         << " @ " << last_fitness_improvement_at_ << ", " << "now @"
         << experiment_progress << " "
         << "(delta_progress = " << delta_experiment_progress << " )";
  report << ". Details: " << algorithm_->Report();
  return report.str();
}

void LockedPopulationServerAlgorithm::AccumulateProgress(
    const vector<Individual> individuals, vector<double>* progress) {
  // Check for `worker_progress` being present in the first individual, which
  // indicates that it was set by the manager to report the worker's progress.
  // If so, grab that worker progress and exit.
  if (!individuals.empty() &&
      !individuals[0].data().worker_progress().empty()) {
    auto individual = individuals[0].data();
    if (progress->empty()) {
      for (const double value : individual.worker_progress()) {
        progress->push_back(value);
      }
    } else {
      const Integer num_values = progress->size();
      CHECK_EQ(individual.individual_progress_size(), num_values);
      for (Integer index = 0; index < num_values; ++index) {
        (*progress)[index] += individual.worker_progress(index);
      }

      // Clear the `worker_progress` fields to ensure the data doesn't stick
      // around and pollute future generations of this individual.
      individual.clear_worker_progress();
    }

    // Verify that no other individuals have `worker_progress` set, to ensure
    // the algorithm can still be used without utilizing the client manager.
    for (int i = 1; i < individuals.size(); ++i) {
      CHECK_EQ(individuals[i].data().worker_progress_size(), 0);
    }

    return;
  }

  // Otherwise, grab `individual_progress` updates from all individuals.
  for (const Individual& individual : individuals) {
    // `worker_progress` shouldn't be set on any of them.
    CHECK_EQ(individual.data().worker_progress_size(), 0);

    if (progress->empty()) {
      for (const double value : individual.data().individual_progress()) {
        progress->push_back(value);
      }
    } else {
      const Integer num_values = progress->size();
      CHECK_EQ(individual.data().individual_progress_size(), num_values);
      for (Integer index = 0; index < num_values; ++index) {
        (*progress)[index] += individual.data().individual_progress(index);
      }
    }
  }
}

void LockedPopulationServerAlgorithm::TrackImprovement(
    const vector<Individual> individuals) {
  for (const Individual& individual : individuals) {
    if (individual.data().fitnesses().empty()) continue;
    if (max_fitnesses_.empty()) {
      for (const double value : individual.data().fitnesses()) {
        max_fitnesses_.push_back(value);
        if (spec_.has_search_restart_if_no_improvement()) {
          last_fitness_improvement_at_ =
              experiment_progress_[spec_.search_restart_if_no_improvement()
                                       .experiment_progress_index()];
        }
      }
    } else {
      const Integer num_values = max_fitnesses_.size();
      CHECK_EQ(individual.data().fitnesses_size(), num_values);
      for (Integer index = 0; index < num_values; ++index) {
        if (individual.data().fitnesses(index) > max_fitnesses_[index]) {
          max_fitnesses_[index] = individual.data().fitnesses(index);
          if (spec_.has_search_restart_if_no_improvement() &&
              index == spec_.search_restart_if_no_improvement()
                           .experiment_progress_index()) {
            last_fitness_improvement_at_ =
                experiment_progress_[spec_.search_restart_if_no_improvement()
                                         .experiment_progress_index()];
          }
        }
      }
    }
  }
}

void LockedPopulationServerAlgorithm::ResetImprovement() {
  if (max_fitnesses_.empty()) {
    // This is near the beginning of the experiment, so max_fitnesses_ is empty
    // because we don't even know how many objectives there are until we hear
    // from a worker. Nothing to do here.
  } else {
    fill(max_fitnesses_.begin(), max_fitnesses_.end(),
         numeric_limits<double>::lowest());
  }

  if (!experiment_progress_.empty() &&  // If not beginning of experiment.
                                        // If this server can trigger restarts
                                        // based on improvement.
      spec_.has_search_restart_if_no_improvement()) {
    last_fitness_improvement_at_ =
        experiment_progress_[spec_.search_restart_if_no_improvement()
                                 .experiment_progress_index()];
  }
}

SearchAlgorithmStats LockedPopulationServerAlgorithm::GetStats() const {
  SearchAlgorithmStats stats;
  stats.set_time_nanos(current_time_->Nanos());
  for (const double value : experiment_progress_) {
    stats.add_experiment_progress(value);
  }
  for (const double value : out_of_sync_progress_) {
    stats.add_out_of_sync_progress(value);
  }
  for (const double value : max_fitnesses_) {
    stats.add_max_fitnesses(value);
  }
  return stats;
}

bool LockedPopulationServerAlgorithm::IsOutOfSync(
    const Individual& individual) const {
  // This is a sufficient but not necessary condition. If too much information
  // is rejected, we can improve this.
  CHECK(individual.data().has_earliest_algorithm_state_nanos());
  return individual.data().earliest_algorithm_state_nanos() <
         process_start_nanos_;
}

bool LockedPopulationServerAlgorithm::IsSearchRestartConditionMet(
    const SearchRestartIfNoImprovementSpec& spec) {
  const double current_progress = ExperimentProgressOrZero(
      experiment_progress_, spec.experiment_progress_index());
  return current_progress - last_fitness_improvement_at_ > spec.max_wait();
}

bool LockedPopulationServerAlgorithm::IsSearchRestartConditionMet(
    const SearchRestartHurdle& spec) {
  const double last_restart_progress = ExperimentProgressOrZero(
      search_restart_experiment_progress_, spec.experiment_progress_index());
  const double current_progress = ExperimentProgressOrZero(
      experiment_progress_, spec.experiment_progress_index());
  const double progress_since_restart =
      current_progress - last_restart_progress;
  if (progress_since_restart >= spec.experiment_progress()) {
    CHECK(spec.fitness_indexes_size() == spec.desired_fitnesses_size());
    for (int i = 0; i < spec.fitness_indexes_size(); ++i) {
      const double current_fitness =
          FitnessOrLowest(max_fitnesses_, spec.fitness_indexes(i));
      const double desired_fitness = spec.desired_fitnesses(i);
      if (current_fitness < desired_fitness) {
        return true;
      }
    }
    return false;
  }
  return false;
}

void LockedPopulationServerAlgorithm::QueueSearchRestartIfReceivedSignal(
    absl::Span<const Individual> puts) {
  for (const Individual& individual : puts) {
    if (individual.data().has_search_restart_signal()) {
      CHECK(individual.data().search_restart_signal().has_restart_nanos());
      const Integer restart_nanos =
          individual.data().search_restart_signal().restart_nanos();
      if (restart_nanos > search_restart_nanos_ &&
          restart_nanos > queued_search_restart_nanos_) {
        Print() << "Detected externally triggered search restart." << Flush();
        QueueSearchRestart(restart_nanos);
      }
    }
  }
}

void LockedPopulationServerAlgorithm::QueueSearchRestartIfConditionMet() {
  bool condition_met = false;
  if (spec_.has_search_restart_if_no_improvement() &&
      IsSearchRestartConditionMet(spec_.search_restart_if_no_improvement())) {
    condition_met = true;
  }
  for (const SearchRestartHurdle& search_restart_hurdle :
       spec_.search_restart_hurdles()) {
    if (IsSearchRestartConditionMet(search_restart_hurdle)) {
      condition_met = true;
      break;
    } else {
    }
  }
  if (condition_met) {
    const Integer restart_nanos = current_time_->Nanos();
    Print() << "Triggering search restart." << Flush();
    QueueSearchRestart(restart_nanos);
  }
}

void LockedPopulationServerAlgorithm::QueueSearchRestart(
    const Integer restart_nanos) {
  Print() << "Queuing search restart: " << "new restart timestamp = "
          << restart_nanos << ", " << "currently queued restart timestamp = "
          << queued_search_restart_nanos_ << ", "
          << "current restart timestamp = " << search_restart_nanos_ << ", "
          << "timestamp now = " << current_time_->Nanos() << Flush();
  queued_search_restart_nanos_ = restart_nanos;
}

vector<Individual>
LockedPopulationServerAlgorithm::PurgeIndividualsFromBeforeSearchRestart(
    absl::Span<const Individual> puts) {
  vector<Individual> purged_puts;
  if (search_restart_nanos_ != kBeginningOfExperiment) {
    for (const Individual& individual : puts) {
      CHECK(individual.data().has_origin_nanos())
          << "If using search restarts, all Individual protos must have the "
          << "`origin_nanos` field set.";
      if (individual.data().origin_nanos() >= search_restart_nanos_) {
        purged_puts.push_back(individual);
      }
    }
  } else {
    purged_puts.insert(purged_puts.end(), puts.begin(), puts.end());
  }
  return purged_puts;
}

void LockedPopulationServerAlgorithm::SendSearchRestartSignal(
    vector<Individual>* gets) {
  if (gets != nullptr && search_restart_nanos_ != kBeginningOfExperiment) {
    for (Individual& individual : *gets) {
      individual.mutable_data()
          ->mutable_search_restart_signal()
          ->set_restart_nanos(search_restart_nanos_);
    }
  }
}

void LockedPopulationServerAlgorithm::ClearSearchRestartSignal(
    vector<Individual>* purged_puts) {
  for (Individual& individual : *purged_puts) {
    if (individual.data().has_search_restart_signal()) {
      individual.mutable_data()->clear_search_restart_signal();
    }
  }
}

void LockedPopulationServerAlgorithm::SearchRestartIfQueued() {
  if (queued_search_restart_nanos_ == kNoQueuedRestart) {
    return;  // No need to restart.
  }

  // Setting this timestamp eventually triggers the restart in other servers.
  search_restart_nanos_ = queued_search_restart_nanos_;

  // Keep track of at what point in the experiment the restart took place.
  search_restart_experiment_progress_.clear();
  for (double value : experiment_progress_) {
    search_restart_experiment_progress_.push_back(value);
  }

  // Restart this server.
  algorithm_->SearchRestart(search_restart_nanos_);

  queued_search_restart_nanos_ = kNoQueuedRestart;

  ResetImprovement();

  Print() << "Executed search restart: " << "from timestamp: "
          << search_restart_nanos_ << ", " << "@"
          << last_fitness_improvement_at_ << Flush();
}

}  // namespace evolution
}  // namespace brain
