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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_NSGA2_UTIL_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_NSGA2_UTIL_H_

#include <memory>
#include <vector>

#include "evolution/lib/candidate.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/population_server.pb.h"

namespace brain {
namespace evolution {

// Sorts a population into Pareto fronts. Uses topological sort.
// See fast-non-dominated-sort implementation in section III A on the NSGA-II
// paper.
// TODO(ereal): this first overload is deprecated. Remove it when possible,
// then remove the templetized code and the overloads for `FitnessesSize` and
// `Fitness`.
std::vector<std::vector<Individual>> FastNonDominatedSort(
    const std::vector<Individual>& candidates);
std::vector<std::vector<std::shared_ptr<Candidate>>> FastNonDominatedSort(
    const std::vector<std::shared_ptr<Candidate>>& candidates);

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_NSGA2_UTIL_H_
