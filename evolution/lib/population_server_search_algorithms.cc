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

#include "evolution/lib/population_server_search_algorithms.h"

#include <memory>
#include <ostream>

#include "absl/log/log.h"
#include "evolution/lib/nsga2_search_algorithm.h"
#include "evolution/lib/random_search_algorithm.h"
#include "evolution/lib/search_algorithm.h"
#include "evolution/lib/search_algorithm_spec.pb.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

using ::std::endl;
using ::std::make_unique;
using ::std::unique_ptr;

unique_ptr<ThreadSafeSearchAlgorithmInterface>
TryMakePopulationServerAlgorithmOwned(
    const SearchAlgorithmSpec& spec, const RNGSeed rng_seed,
    ThreadSafeSearchAlgorithmInterface* mock_algorithm) {
  if (mock_algorithm == nullptr) {
    // No externally-owned object passed. Construct one.
    unique_ptr<ThreadSafeSearchAlgorithmInterface> algorithm;
    if (spec.has_nsga2()) {
      algorithm = make_unique<LockedPopulationServerAlgorithm>(
          make_unique<Nsga2SearchAlgorithm>(spec.nsga2(), rng_seed), spec);
    } else if (spec.has_random()) {
      algorithm = make_unique<LockedPopulationServerAlgorithm>(
          make_unique<RandomSearchAlgorithm>(spec.random(), rng_seed), spec);
    } else {
      LOG(FATAL) << "Unsupported population server algorithm." << endl;
    }
    return algorithm;
  } else {
    // Using externally owned object.
    return unique_ptr<ThreadSafeSearchAlgorithmInterface>();
  }
}

}  // namespace evolution
}  // namespace brain
