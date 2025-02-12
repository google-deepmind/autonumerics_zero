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

#include "evolution/lib/nsga2_search_algorithm.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "evolution/lib/candidate.h"
#include "evolution/lib/current_time.h"
#include "evolution/lib/fitnesses_reduction.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/mocking.h"
#include "evolution/lib/nsga2_search_algorithm_spec.pb.h"
#include "evolution/lib/nsga2_search_algorithm_state.pb.h"
#include "evolution/lib/nsga2_util.h"
#include "evolution/lib/rng.h"
#include "evolution/lib/search_algorithm.pb.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

using ::std::make_pair;
using ::std::make_shared;
using ::std::nullopt;
using ::std::optional;
using ::std::pair;
using ::std::shared_ptr;
using ::std::string;
using ::std::vector;

void PerformSelectionStage(const Nsga2SelectionStageSpec& stage_spec,
                           vector<vector<shared_ptr<Candidate>>>& frontiers,
                           vector<shared_ptr<Individual>>& parent_population,
                           RNG& rng);

Nsga2SearchAlgorithm::Nsga2SearchAlgorithm(
    const Nsga2SearchAlgorithmSpec& spec, const RNGSeed rng_seed,
    CurrentTimeInterface* mock_current_time)
    : spec_(spec),
      next_selected_idx_(0),
      puts_count_(0),
      rng_(rng_seed),
      owned_current_time_(MakeOwnedIfNoMock<CurrentTime>(mock_current_time)),
      current_time_(GetPointerToMockOrOwned(mock_current_time,
                                            owned_current_time_.get())) {
  CHECK_GT(spec_.child_population_size(), 1);
  CHECK(!spec_.stages().empty());
  for (const Nsga2SelectionStageSpec& stage_spec : spec_.stages()) {
    CHECK_GT(stage_spec.parent_population_size(), 0);
  }
  CHECK(spec_.has_use_elitism());
  if (spec_.has_fitnesses_reduction()) {
    fitnesses_reduction_fn_ =
        BuildFitnessesReductionFn(spec_.fitnesses_reduction());
  } else {
    fitnesses_reduction_fn_ = nullptr;
  }
  ResetState();
}

string Nsga2SearchAlgorithm::Serialize() {
  Nsga2SearchAlgorithmState checkpoint;
  checkpoint.set_time_nanos(current_time_->Nanos());
  for (const shared_ptr<Individual>& individual : child_population_) {
    *checkpoint.add_incoming_individuals() = *individual;
  }
  for (const shared_ptr<Individual>& individual : parent_population_) {
    *checkpoint.add_outgoing_individuals() = *individual;
  }
  checkpoint.set_next_selected_idx(next_selected_idx_);
  checkpoint.set_puts_count(puts_count_);
  return checkpoint.SerializeAsString();
}

void Nsga2SearchAlgorithm::Deserialize(const string& serialized) {
  Nsga2SearchAlgorithmState checkpoint;
  CHECK(checkpoint.ParseFromString(serialized));
  child_population_.clear();
  for (const Individual& individual : checkpoint.incoming_individuals()) {
    child_population_.emplace_back(make_shared<Individual>(individual));
  }
  parent_population_.clear();
  for (const Individual& individual : checkpoint.outgoing_individuals()) {
    parent_population_.emplace_back(make_shared<Individual>(individual));
  }
  CHECK(checkpoint.has_next_selected_idx());
  next_selected_idx_ = checkpoint.next_selected_idx();

  CHECK(checkpoint.has_puts_count());
  puts_count_ = checkpoint.puts_count();
}

void Nsga2SearchAlgorithm::Exchange(const vector<Individual>& puts,
                                    Integer num_to_get,
                                    vector<Individual>* gets,
                                    vector<Individual>* kills) {
  vector<shared_ptr<Individual>> prev_population;  // To compute `kills`.
  if (kills != nullptr) prev_population = CopyCurrentPopulation();
  vector<shared_ptr<Individual>> shared_puts;
  shared_puts.reserve(puts.size());
  for (const Individual& individual : puts) {
    shared_puts.emplace_back(make_shared<Individual>(individual));
  }
  HandlePuts(shared_puts);
  vector<shared_ptr<Individual>> shared_gets;
  HandleGets(num_to_get, &shared_gets);
  if (!shared_gets.empty()) {
    gets->clear();
    gets->reserve(shared_gets.size());
    for (const shared_ptr<Individual>& individual : shared_gets) {
      gets->push_back(*individual);
    }
  }
  if (kills != nullptr) {
    vector<shared_ptr<Individual>> shared_kills;
    SubtractCurrentPopulation(&prev_population, &shared_kills);
    for (const shared_ptr<Individual>& individual : shared_kills) {
      kills->push_back(*individual);
    }
  }
}

string Nsga2SearchAlgorithm::Report() const {
  return "Reporting not supported.";
}

void Nsga2SearchAlgorithm::SearchRestart(Integer start_nanos) {
  vector<shared_ptr<Individual>> recent_individuals;
  for (const shared_ptr<Individual>& individual : parent_population_) {
    CHECK(individual->data().has_origin_nanos());
    if (individual->data().origin_nanos() >= start_nanos) {
      recent_individuals.emplace_back(individual);
    }
  }
  for (const shared_ptr<Individual>& individual : child_population_) {
    CHECK(individual->data().has_origin_nanos());
    if (individual->data().origin_nanos() >= start_nanos) {
      recent_individuals.emplace_back(individual);
    }
  }

  // Sort recent individuals according to their age, to be able to reinsert
  // them in chronological order.
  std::sort(
      recent_individuals.begin(), recent_individuals.end(),
      [](const shared_ptr<Individual>& i1, const shared_ptr<Individual>& i2) {
        return i1->data().origin_nanos() < i2->data().origin_nanos();
      });

  ResetState();
  HandlePuts(recent_individuals);
}

void Nsga2SearchAlgorithm::ResetState() {
  child_population_.clear();
  parent_population_.clear();
  next_selected_idx_ = 0;
  puts_count_ = 0;
}

vector<shared_ptr<Individual>> Nsga2SearchAlgorithm::CopyCurrentPopulation() {
  vector<shared_ptr<Individual>> curr_population;
  curr_population.insert(curr_population.end(), child_population_.begin(),
                         child_population_.end());
  curr_population.insert(curr_population.end(), parent_population_.begin(),
                         parent_population_.end());
  return curr_population;
}

void Nsga2SearchAlgorithm::SubtractCurrentPopulation(
    vector<shared_ptr<Individual>>* from,
    vector<shared_ptr<Individual>>* diff) {
  absl::flat_hash_set<string> alive_individual_ids;
  for (const shared_ptr<Individual>& individual : child_population_) {
    alive_individual_ids.insert(individual->individual_id());
  }
  for (const shared_ptr<Individual>& individual : parent_population_) {
    alive_individual_ids.insert(individual->individual_id());
  }

  for (shared_ptr<Individual>& individual : *from) {
    if (!alive_individual_ids.contains(individual->individual_id())) {
      diff->emplace_back(individual);
    }
  }
}

void Nsga2SearchAlgorithm::HandlePuts(
    const vector<shared_ptr<Individual>>& puts) {
  for (Integer i = 0; i < puts.size(); ++i) {
    child_population_.emplace_back(puts[i]);
    puts_count_++;
    if (puts_count_ >= spec_.child_population_size()) {
      InvokeNonDominatedSort();
      child_population_.clear();
      next_selected_idx_ = 0;
      puts_count_ = 0;
    }
  }
}

namespace {

// Removes duplicated individual IDs. Assumes the individuals are sorted so that
// later individuals are preferred (i.e. parents must be before children and
// children must be in the order they were received).
void DeduplicateIndividuals(vector<shared_ptr<Candidate>>& individuals) {
  vector<shared_ptr<Candidate>> deduplicated;
  absl::flat_hash_set<string> ids_seen;
  for (auto it = individuals.rbegin(); it != individuals.rend(); ++it) {
    if (!ids_seen.contains((*it)->individual->individual_id())) {
      deduplicated.push_back(*it);
      ids_seen.insert((*it)->individual->individual_id());
    }
  }
  individuals.clear();
  for (auto it = deduplicated.rbegin(); it != deduplicated.rend(); ++it) {
    individuals.push_back(*it);
  }
}

}  // namespace

void Nsga2SearchAlgorithm::InvokeNonDominatedSort() {
  // Calculate pareto frontiers.
  vector<shared_ptr<Candidate>> candidates;
  if (spec_.use_elitism()) {
    for (const shared_ptr<Individual>& individual : parent_population_) {
      candidates.push_back(make_shared<Candidate>(individual));
    }
    for (const shared_ptr<Individual>& individual : child_population_) {
      candidates.push_back(make_shared<Candidate>(individual));
    }
  } else {
    for (const shared_ptr<Individual>& individual : child_population_) {
      candidates.push_back(make_shared<Candidate>(individual));
    }
  }

  ReduceFitnesses(candidates);

  DeduplicateIndividuals(candidates);
  vector<vector<shared_ptr<Candidate>>> frontiers =
      FastNonDominatedSort(candidates);

  if (spec_.use_crowding_distance_sorting()) {
    for (Integer i = 0; i < frontiers.size(); ++i) {
      CrowdingDistanceSort(frontiers[i]);
    }
  }

  // Update outgoing population.
  parent_population_.clear();
  for (const Nsga2SelectionStageSpec& stage_spec : spec_.stages()) {
    PerformSelectionStage(stage_spec, frontiers, parent_population_, rng_);
  }
}

void Nsga2SearchAlgorithm::CrowdingDistanceSort(
    vector<shared_ptr<Candidate>>& front) {
  const Integer num_individuals = front.size();
  if (num_individuals <= 1) return;
  const Integer num_objectives = FitnessesSize(front[0]);

  vector<double> distances(num_individuals);
  // dist is an index array, dist[i][j] represents the individual index in the
  // front array. This array is also used for per objective sorting.
  vector<vector<int>> dist(num_objectives, vector<int>(num_individuals, 0));
  for (Integer i = 0; i < num_objectives; ++i) {
    for (Integer j = 0; j < num_individuals; ++j) {
      dist[i][j] = j;
    }
  }

  for (Integer i = 0; i < num_objectives; ++i) {
    std::stable_sort(dist[i].begin(), dist[i].end(),
                     [i, &front](const int a, const int b) -> bool {
                       return Fitness(front[a], i) < Fitness(front[b], i);
                     });

    double max_value = Fitness(front[dist[i][num_individuals - 1]], i);
    double min_value = Fitness(front[dist[i][0]], i);
    for (Integer j = 0; j < num_individuals; ++j) {
      if (j == 0 || j == num_individuals - 1) {
        distances[dist[i][j]] = num_objectives;  // give a maximum value.
      } else if (max_value != min_value) {
        distances[dist[i][j]] += ((Fitness(front[dist[i][j + 1]], i) -
                                   Fitness(front[dist[i][j - 1]], i)) /
                                  (max_value - min_value));
      }
    }
  }

  // Sort front in decreasing order wrt crowding distances.
  vector<int> idx_arr(num_individuals);
  for (Integer i = 0; i < num_individuals; ++i) {
    idx_arr[i] = i;
  }
  std::stable_sort(idx_arr.begin(), idx_arr.end(),
                   [&distances](const int a, const int b) -> bool {
                     return distances[a] > distances[b];
                   });

  vector<shared_ptr<Candidate>> sorted_front(num_individuals);
  for (Integer i = 0; i < num_individuals; ++i) {
    sorted_front[i] = front[idx_arr[i]];
  }
  front.swap(sorted_front);
}

bool SatisfiesFitnessLimits(const Nsga2SelectionStageSpec& spec,
                            const shared_ptr<Candidate>& individual) {
  if (!spec.min_fitnesses().empty()) {  // Empty ==> no minimum fitnesses.
    const Integer num_fitnesses = FitnessesSize(individual);
    CHECK_EQ(spec.min_fitnesses_size(), num_fitnesses);
    for (Integer index = 0; index < num_fitnesses; ++index) {
      if (Fitness(individual, index) < spec.min_fitnesses(index)) {
        return false;
      }
    }
  }
  if (!spec.max_fitnesses().empty()) {  // Empty ==> no maximum fitnesses.
    const Integer num_fitnesses = FitnessesSize(individual);
    CHECK_EQ(spec.max_fitnesses_size(), num_fitnesses);
    for (Integer index = 0; index < num_fitnesses; ++index) {
      if (Fitness(individual, index) >= spec.max_fitnesses(index)) {
        return false;
      }
    }
  }
  return true;
}

void PerformSelectionStage(const Nsga2SelectionStageSpec& stage_spec,
                           vector<vector<shared_ptr<Candidate>>>& frontiers,
                           vector<shared_ptr<Individual>>& parent_population,
                           RNG& rng) {
  for (vector<shared_ptr<Candidate>>& frontier : frontiers) {
    vector<shared_ptr<Candidate>> losers;
    for (shared_ptr<Candidate>& individual : frontier) {
      if (parent_population.size() >= stage_spec.parent_population_size()) {
        break;
      }
      if (SatisfiesFitnessLimits(stage_spec, individual)) {
        parent_population.push_back(individual->individual);
      } else {
        losers.push_back(individual);
      }
    }
    frontier.clear();
    for (shared_ptr<Candidate>& individual : losers) {
      frontier.push_back(individual);
    }
  }
}

void Nsga2SearchAlgorithm::HandleGets(Integer num_to_get,
                                      vector<shared_ptr<Individual>>* gets) {
  if (parent_population_.empty()) return;
  if (num_to_get == -1) {
    GetFullPopulation(gets);
  } else {
    GetSelectedIndividuals(num_to_get, gets);
  }
}

void Nsga2SearchAlgorithm::GetFullPopulation(
    vector<shared_ptr<Individual>>* gets) {
  CHECK(!parent_population_.empty());
  gets->clear();
  for (const shared_ptr<Individual>& individual : parent_population_) {
    gets->emplace_back(individual);
  }
  std::sort(
      gets->begin(), gets->end(),
      [this](const shared_ptr<Individual>& i1,
             const shared_ptr<Individual>& i2) {
        const Integer num_objectives = i1->data().fitnesses_size();
        CHECK_GE(num_objectives, 1);
        CHECK_EQ(i2->data().fitnesses_size(), num_objectives);
        for (Integer obj = 0; obj < num_objectives; ++obj) {
          if (i1->data().fitnesses(obj) > i2->data().fitnesses(obj)) {
            return true;
          } else if (i1->data().fitnesses(obj) < i2->data().fitnesses(obj)) {
            return false;
          } else {
            // Continue loop to the next objective.
          }
        }
        return false;  // All objectives are the same.
      });
}

void Nsga2SearchAlgorithm::GetSelectedIndividuals(
    Integer num_to_get, vector<shared_ptr<Individual>>* gets) {
  for (Integer i = 0; i < num_to_get; ++i) {
    gets->emplace_back(parent_population_[next_selected_idx_]);
    next_selected_idx_ = (next_selected_idx_ + 1) % parent_population_.size();
  }
}

void Nsga2SearchAlgorithm::ReduceFitnesses(
    vector<shared_ptr<Candidate>>& candidates) {
  if (fitnesses_reduction_fn_ != nullptr) {
    for (shared_ptr<Candidate>& candidate : candidates) {
      candidate->reduced_fitnesses =
          fitnesses_reduction_fn_(candidate->reduced_fitnesses);
    }
  }
}

}  // namespace evolution
}  // namespace brain
