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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_RANDOM_SEARCH_ALGORITHM_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_RANDOM_SEARCH_ALGORITHM_H_

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "evolution/lib/current_time.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/population_server.pb.h"
#include "evolution/lib/random_search_algorithm_spec.pb.h"
#include "evolution/lib/rng.h"
#include "evolution/lib/search_algorithm.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

// An algorithm to implement regularized evolution in the server.
// Thread-compatible, but not thread-safe.
class RandomSearchAlgorithm final
    : public ThreadCompatibleSearchAlgorithmInterface {
 public:
  RandomSearchAlgorithm(const RandomSearchAlgorithmSpec& spec, RNGSeed rng_seed,
                        CurrentTimeInterface* mock_current_time = nullptr);
  RandomSearchAlgorithm(const RandomSearchAlgorithm& other) = delete;
  RandomSearchAlgorithm& operator=(const RandomSearchAlgorithm& other) = delete;

  std::string Serialize() override;
  void Deserialize(const std::string& serialized) override;
  void Exchange(const std::vector<Individual>& puts, Integer num_to_get,
                std::vector<Individual>* gets,
                std::vector<Individual>* kills) override;
  std::string Report() const override;
  void SearchRestart(Integer start_nanos) override;

 private:
  // Selects randomly.
  std::shared_ptr<Individual> Select();

  // Removes an individual from the population and returns it.
  std::shared_ptr<Individual> RemoveFromPopulation();

  // Returns all the individuals in order of their 0th fitness (highest first)
  // as the primary key (individuals with no fitness go last) and in order of
  // their timestamp (latest first) as the secondary key.
  void FillWithSortedWholePopulation(std::vector<Individual>* individuals);

  // Return individuals by random sampling from the population.
  void FillBySamplingWithReplacement(Integer num_to_get,
                                     std::vector<Individual>* gets);
  void FillBySamplingWithoutReplacement(Integer num_to_get,
                                        std::vector<Individual>* gets);

  // Returns randomly selected individuals.
  void FillWithSelectedIndividuals(Integer num_to_get,
                                   std::vector<Individual>* individuals);

  const RandomSearchAlgorithmSpec spec_;
  std::deque<std::shared_ptr<Individual>> population_;
  RNG rng_;
  const std::unique_ptr<CurrentTimeInterface> owned_current_time_;
  CurrentTimeInterface* const current_time_;
};

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_RANDOM_SEARCH_ALGORITHM_H_
