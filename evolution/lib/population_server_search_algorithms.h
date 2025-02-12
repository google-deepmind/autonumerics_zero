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

// Function to construct algorithm instances to be run on the population server.

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_POPULATION_SERVER_SEARCH_ALGORITHMS_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_POPULATION_SERVER_SEARCH_ALGORITHMS_H_

#include <memory>
#include <vector>

#include "evolution/lib/search_algorithm.h"
#include "evolution/lib/search_algorithm_spec.pb.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

// Builds an algorithm, if no `mock_algorithm` is provided.
std::unique_ptr<ThreadSafeSearchAlgorithmInterface>
TryMakePopulationServerAlgorithmOwned(
    const SearchAlgorithmSpec& spec, RNGSeed rng_seed,
    ThreadSafeSearchAlgorithmInterface* mock_algorithm);

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_POPULATION_SERVER_SEARCH_ALGORITHMS_H_
