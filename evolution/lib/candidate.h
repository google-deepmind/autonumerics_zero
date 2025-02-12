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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_CANDIDATE_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_CANDIDATE_H_

#include <memory>
#include <vector>

#include "evolution/lib/individual.pb.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

// Holds an individual and a set of reduced fitnesses. The intended use is
// for the `reduced_fitnesses` to be populated from information in the
// `individual` while the `individual` is left unchanged.
struct Candidate {
  explicit Candidate(std::shared_ptr<Individual> individual);
  std::shared_ptr<Individual> individual;
  std::vector<double> reduced_fitnesses;
};

// Overloads that allow accessing the fitnesses in the individual or in the
// candidate from templatized code.
Integer FitnessesSize(const Individual& candidate);
Integer FitnessesSize(const std::shared_ptr<Candidate>& candidate);
double Fitness(const Individual& candidate, Integer index);
double Fitness(const std::shared_ptr<Candidate>& candidate,
               Integer fitness_index);

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_CANDIDATE_H_
