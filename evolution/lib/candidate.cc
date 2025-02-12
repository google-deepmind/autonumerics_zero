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

#include "evolution/lib/candidate.h"

#include <memory>

#include "evolution/lib/individual.pb.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

using ::std::shared_ptr;

Candidate::Candidate(shared_ptr<Individual> individual) {
  this->individual = individual;
  reduced_fitnesses.insert(reduced_fitnesses.end(),
                           individual->data().fitnesses().begin(),
                           individual->data().fitnesses().end());
}

Integer FitnessesSize(const Individual& candidate) {
  return candidate.data().fitnesses_size();
}
Integer FitnessesSize(const shared_ptr<Candidate>& candidate) {
  return candidate->reduced_fitnesses.size();
}

double Fitness(const Individual& candidate, Integer index) {
  return candidate.data().fitnesses(index);
}
double Fitness(const shared_ptr<Candidate>& candidate, Integer fitness_index) {
  return candidate->reduced_fitnesses.at(fitness_index);
}

}  // namespace evolution
}  // namespace brain
