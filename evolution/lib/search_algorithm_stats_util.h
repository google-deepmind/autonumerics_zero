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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_SEARCH_ALGORITHM_STATS_UTIL_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_SEARCH_ALGORITHM_STATS_UTIL_H_

#include <vector>

#include "absl/types/span.h"
#include "evolution/lib/search_algorithm_stats.pb.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

// Takes a set of SearchAlgorithmStats and returns a single aggregated message.
SearchAlgorithmStats AggregateStats(
    absl::Span<const SearchAlgorithmStats> all_stats);

// Returns the experiment progress with the given index. If the experiment
// progress vector is empty, then it returns zero (based on the assumption that
// it is so early in the experiment that there is no information on
// experiment progress yet). For more info on the experiment progress vector,
// see the SearchAlgorithmStats proto and the Individual proto.
double ExperimentProgressOrZero(absl::Span<const double> experiment_progress,
                                Integer index);

// Returns the fitness with the given index. If the fitnesses vector is empty,
// then it returns the lowest possible value. For more info on the fitnesses
// vector, see the Individual proto.
double FitnessOrLowest(absl::Span<const double> fitnesses, Integer index);

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_SEARCH_ALGORITHM_STATS_UTIL_H_
