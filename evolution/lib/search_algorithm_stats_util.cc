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

#include "evolution/lib/search_algorithm_stats_util.h"

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "evolution/lib/search_algorithm_stats.pb.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

using ::std::numeric_limits;
using ::std::vector;

SearchAlgorithmStats AggregateStats(
    absl::Span<const SearchAlgorithmStats> all_stats) {
  SearchAlgorithmStats aggregate_stats;

  if (all_stats.empty()) {
    return aggregate_stats;
  } else if (all_stats.size() == 1) {
    return all_stats[0];
  }

  vector<double> exp_progresses;
  vector<double> oos_progresses;
  vector<double> fitnesses;
  vector<double> max_num_distinct_hashes;

  for (auto& stats : all_stats) {
    // Set "aggregate time" to the max time across `all_stats`.
    aggregate_stats.set_time_nanos(
        fmax(aggregate_stats.time_nanos(), stats.time_nanos()));

    // Aggregate experiment_progress by performing an element-wise vector sum.
    CHECK(exp_progresses.empty() || stats.experiment_progress_size() == 0 ||
          stats.experiment_progress_size() == exp_progresses.size());
    for (Integer i = 0; i < stats.experiment_progress_size(); ++i) {
      if (i >= exp_progresses.size()) {
        exp_progresses.push_back(stats.experiment_progress(i));
      } else {
        exp_progresses[i] += stats.experiment_progress(i);
      }
    }

    // Aggregate out_of_sync_progress by performing an element-wise vector sum.
    CHECK(oos_progresses.empty() || stats.out_of_sync_progress_size() == 0 ||
          stats.out_of_sync_progress_size() == oos_progresses.size());
    for (Integer i = 0; i < stats.out_of_sync_progress_size(); ++i) {
      if (i >= oos_progresses.size()) {
        oos_progresses.push_back(stats.out_of_sync_progress(i));
      } else {
        oos_progresses[i] += stats.out_of_sync_progress(i);
      }
    }

    // Aggregate max_fitnesses by performing an element-wise vector max.
    CHECK(fitnesses.empty() || stats.max_fitnesses_size() == 0 ||
          stats.max_fitnesses_size() == fitnesses.size());
    for (Integer i = 0; i < stats.max_fitnesses_size(); ++i) {
      if (i >= fitnesses.size()) {
        fitnesses.push_back(stats.max_fitnesses(i));
      } else {
        fitnesses[i] = fmax(fitnesses[i], stats.max_fitnesses(i));
      }
    }

    // Aggregate max_num_distinct_hashes using an element-wise vector max.
    CHECK(max_num_distinct_hashes.empty() ||
          stats.num_distinct_hashes_size() == 0 ||
          stats.num_distinct_hashes_size() == max_num_distinct_hashes.size());
    for (Integer i = 0; i < stats.num_distinct_hashes_size(); ++i) {
      if (i >= max_num_distinct_hashes.size()) {
        max_num_distinct_hashes.push_back(stats.num_distinct_hashes(i));
      } else {
        max_num_distinct_hashes[i] =
            fmax(max_num_distinct_hashes[i], stats.num_distinct_hashes(i));
      }
    }
  }

  for (double& progress : exp_progresses) {
    aggregate_stats.add_experiment_progress(progress);
  }

  for (double& progress : oos_progresses) {
    aggregate_stats.add_out_of_sync_progress(progress);
  }

  for (double& fitness : fitnesses) {
    aggregate_stats.add_max_fitnesses(fitness);
  }

  for (double& max_num_distinct_hash : max_num_distinct_hashes) {
    aggregate_stats.add_num_distinct_hashes(max_num_distinct_hash);
  }

  return aggregate_stats;
}

double ExperimentProgressOrZero(absl::Span<const double> experiment_progress,
                                Integer index) {
  if (experiment_progress.empty()) {
    return 0.0;
  } else {
    return experiment_progress[index];
  }
}

double FitnessOrLowest(absl::Span<const double> fitnesses, Integer index) {
  if (fitnesses.empty()) {
    return numeric_limits<double>::lowest();
  } else {
    return fitnesses[index];
  }
}

}  // namespace evolution
}  // namespace brain
