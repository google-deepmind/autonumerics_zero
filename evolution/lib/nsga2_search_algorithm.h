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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_NSGA2_SEARCH_ALGORITHM_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_NSGA2_SEARCH_ALGORITHM_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "evolution/lib/candidate.h"
#include "evolution/lib/current_time.h"
#include "evolution/lib/fitnesses_reduction.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/nsga2_search_algorithm_spec.pb.h"
#include "evolution/lib/population_server.pb.h"
#include "evolution/lib/rng.h"
#include "evolution/lib/search_algorithm.h"
#include "evolution/lib/search_algorithm_stats.pb.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

// NSGA-II algorithm.
// Reference paper: Deb 2002, "A fast and elitist multiobjective genetic
// algorithm: NSGA-II".
// Thread-compatible, but not thread-safe.
class Nsga2SearchAlgorithm final
    : public ThreadCompatibleSearchAlgorithmInterface {
 public:
  Nsga2SearchAlgorithm(const Nsga2SearchAlgorithmSpec& spec, RNGSeed rng_seed,
                       CurrentTimeInterface* mock_current_time = nullptr);
  Nsga2SearchAlgorithm(const Nsga2SearchAlgorithm& other) = delete;
  Nsga2SearchAlgorithm& operator=(const Nsga2SearchAlgorithm& other) = delete;

  // See base class.
  std::string Serialize() override;
  void Deserialize(const std::string& serialized) override;
  void Exchange(const std::vector<Individual>& puts, Integer num_to_get,
                std::vector<Individual>* gets,
                std::vector<Individual>* kills) override;
  std::string Report() const override;
  void SearchRestart(Integer start_nanos) override;

 private:
  // Puts the internal state of the algorithm in a zero-knowledge state.
  void ResetState();

  // Copies the current population and returns the copy.
  std::vector<std::shared_ptr<Individual>> CopyCurrentPopulation();

  // Subtracts the current population from `from` and copies the results
  // to `diff`.
  void SubtractCurrentPopulation(
      std::vector<std::shared_ptr<Individual>>* from,
      std::vector<std::shared_ptr<Individual>>* diff);

  // Put-related functions.
  // Entry point for put-related functions.
  void HandlePuts(const std::vector<std::shared_ptr<Individual>>& puts);
  // Reassembles parent_population_ (i.e. parents) and the
  // child_population_ (i.e. their children) into the new
  // outgoing_generation_ (i.e. future parents).
  void InvokeNonDominatedSort();
  // Sorts a Pareto front by crowding distance, most distant first.
  // See crowding-distance-assignment algorithm in section III B of the NSGA-II
  // paper.
  void CrowdingDistanceSort(std::vector<std::shared_ptr<Candidate>>& frontier);

  // Get-related functions.
  // Entry point to get-related functions.
  void HandleGets(Integer num_to_get,
                  std::vector<std::shared_ptr<Individual>>* gets);
  // Copies the population to the `gets` output vector. Sorts this output vector
  // first by the first objective, second by the second objective, etc.
  void GetFullPopulation(std::vector<std::shared_ptr<Individual>>* gets);
  // Copies individuals chosen by the NSGA selection process.
  void GetSelectedIndividuals(Integer num_to_get,
                              std::vector<std::shared_ptr<Individual>>* gets);
  void ReduceFitnesses(std::vector<std::shared_ptr<Candidate>>& candidates);

  const Nsga2SearchAlgorithmSpec spec_;
  // Store individuals from puts before nondominated sorting.
  std::vector<std::shared_ptr<Individual>> child_population_;
  // If use elitism, store first half individuals after nondominate sorting
  // the array: [child_population_ + previous elite_population_].
  // If not use elitism, store all individuals after nondominate sorting
  // the array: child_population_.
  std::vector<std::shared_ptr<Individual>> parent_population_;
  Integer next_selected_idx_;
  // Number of individuals got added since the latest involcation of
  // ParetoFrontierSort.
  Integer puts_count_;
  RNG rng_;
  const std::unique_ptr<CurrentTimeInterface> owned_current_time_;
  CurrentTimeInterface* current_time_;
  FitnessesReductionFn fitnesses_reduction_fn_;
};

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_NSGA2_SEARCH_ALGORITHM_H_
