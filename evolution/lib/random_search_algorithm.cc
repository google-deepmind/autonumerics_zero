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

#include "evolution/lib/random_search_algorithm.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "evolution/lib/current_time.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/mocking.h"
#include "evolution/lib/population_server.pb.h"
#include "evolution/lib/random_search_algorithm_spec.pb.h"
#include "evolution/lib/random_search_algorithm_state.pb.h"
#include "evolution/lib/rng.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

using ::absl::flat_hash_set;
using ::std::make_shared;
using ::std::shared_ptr;
using ::std::string;
using ::std::vector;

RandomSearchAlgorithm::RandomSearchAlgorithm(
    const RandomSearchAlgorithmSpec& spec, const RNGSeed rng_seed,
    CurrentTimeInterface* mock_current_time)
    : spec_(spec),
      rng_(rng_seed),
      owned_current_time_(MakeOwnedIfNoMock<CurrentTime>(mock_current_time)),
      current_time_(GetPointerToMockOrOwned(mock_current_time,
                                            owned_current_time_.get())) {
  CHECK_GT(spec_.population_size(), 1);
}

string RandomSearchAlgorithm::Serialize() {
  RandomSearchAlgorithmState state;
  state.set_time_nanos(current_time_->Nanos());
  for (shared_ptr<Individual>& individual : population_) {
    *state.add_individuals() = *individual;
  }
  return state.SerializeAsString();
}

void RandomSearchAlgorithm::Deserialize(const string& serialized) {
  RandomSearchAlgorithmState state;
  CHECK(state.ParseFromString(serialized));
  population_.clear();
  for (const Individual& individual : state.individuals()) {
    population_.push_back(make_shared<Individual>(individual));
  }
}

void RandomSearchAlgorithm::Exchange(const vector<Individual>& puts,
                                     Integer num_to_get,
                                     vector<Individual>* gets,
                                     vector<Individual>* kills) {
  for (const Individual& individual : puts) {
    population_.push_back(make_shared<Individual>(individual));
    if (population_.size() > spec_.population_size()) {
      std::shared_ptr<Individual> removed = RemoveFromPopulation();
      if (kills != nullptr) {
        kills->push_back(*removed);
      }
    }
  }
  if (num_to_get == -1) {
    FillWithSortedWholePopulation(gets);
  } else if (spec_.has_distinct_selection()) {
    if (population_.size() >=
        spec_.distinct_selection().min_population_size()) {
      FillBySamplingWithoutReplacement(num_to_get, gets);
    } else {
      FillBySamplingWithReplacement(num_to_get, gets);
    }
  } else {
    FillBySamplingWithReplacement(num_to_get, gets);
  }
}

void RandomSearchAlgorithm::FillBySamplingWithReplacement(
    const Integer num_to_get, vector<Individual>* gets) {
  for (Integer i = 0; i < num_to_get; ++i) {
    shared_ptr<Individual> selected = Select();
    if (selected == nullptr) continue;
    gets->push_back(*selected);
  }
}

void RandomSearchAlgorithm::FillBySamplingWithoutReplacement(
    const Integer num_to_get, vector<Individual>* gets) {
  flat_hash_set<Individual*> selected_set;
  for (Integer i = 0; i < num_to_get; ++i) {
    shared_ptr<Individual> selected = Select();
    CHECK(selected != nullptr);
    while (selected_set.contains(selected.get())) {
      selected = Select();
    }
    selected_set.insert(selected.get());
    gets->push_back(*selected);
  }
}

shared_ptr<Individual> RandomSearchAlgorithm::Select() {
  if (population_.empty()) {
    return nullptr;
  }
  const Integer individual_index = rng_.UniformInteger(0, population_.size());
  shared_ptr<Individual> curr_individual = population_[individual_index];
  return curr_individual;
}

shared_ptr<Individual> RandomSearchAlgorithm::RemoveFromPopulation() {
  if (!spec_.remove_oldest()) {
    // Swap a random individual to the population front, so it will be removed.
    const Integer remove_index = rng_.UniformInteger(0, population_.size());
    population_.front().swap(population_[remove_index]);
  }
  shared_ptr<Individual> removed = population_.front();
  population_.pop_front();
  return removed;
}

void RandomSearchAlgorithm::FillWithSortedWholePopulation(
    vector<Individual>* individuals) {
  // Copy the population pointers.
  vector<shared_ptr<Individual>> population_copy;
  population_copy.reserve(population_.size());
  for (shared_ptr<Individual>& individual : population_) {
    population_copy.emplace_back(individual);
  }

  // Sort the pointers in the population copy. We don't assume the presence of
  // fitnesses. Individuals with fitnesses go first, remaining individuals are
  // sorted by their timestamp.
  std::sort(population_copy.begin(), population_copy.end(),
            [](const shared_ptr<Individual>& individual1,
               const shared_ptr<Individual>& individual2) {
              if (individual1->data().fitnesses().empty() &&
                  individual2->data().fitnesses().empty()) {
                return individual1->time_nanos() > individual2->time_nanos();
              } else if (individual1->data().fitnesses().empty()) {
                return false;
              } else if (individual2->data().fitnesses().empty()) {
                return true;
              } else {
                return individual1->data().fitnesses(0) >
                       individual2->data().fitnesses(0);
              }
            });

  // Return individuals (not pointers).
  for (shared_ptr<Individual>& individual : population_copy) {
    individuals->push_back(*individual);
  }
}

void RandomSearchAlgorithm::SearchRestart(Integer start_nanos) {
  vector<shared_ptr<Individual>> recent_individuals;
  for (shared_ptr<Individual>& individual : population_) {
    CHECK(individual->data().has_origin_nanos());
    if (individual->data().origin_nanos() >= start_nanos) {
      recent_individuals.push_back(individual);
    }
  }

  // We only keep the recent individuals.
  population_.clear();
  for (shared_ptr<Individual>& individual : recent_individuals) {
    population_.push_back(individual);
  }

  std::sort(population_.begin(), population_.end(),
            [](const shared_ptr<Individual>& individual1,
               const shared_ptr<Individual>& individual2) {
              return individual1->time_nanos() < individual2->time_nanos();
            });
}

std::string RandomSearchAlgorithm::Report() const {
  return "Reporting not supported.";
}

}  // namespace evolution
}  // namespace brain
