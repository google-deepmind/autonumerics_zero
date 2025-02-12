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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_FITNESSES_REDUCTION_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_FITNESSES_REDUCTION_H_

#include <functional>
#include <vector>

#include "evolution/lib/fitnesses_reduction_spec.pb.h"
#include "evolution/lib/individual.pb.h"

namespace brain {
namespace evolution {

using FitnessesReductionFn =
    std::function<std::vector<double>(const std::vector<double>&)>;

FitnessesReductionFn BuildFitnessesReductionFn(
    const FitnessesReductionSpec& spec);

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_FITNESSES_REDUCTION_H_
