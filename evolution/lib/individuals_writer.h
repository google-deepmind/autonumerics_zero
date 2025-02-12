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

// Base class for Individual proto writers.

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_INDIVIDUALS_WRITER_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_INDIVIDUALS_WRITER_H_

#include <vector>

#include "evolution/lib/individual.pb.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace brain {
namespace evolution {

// Interface for Individual proto writers.
class IndividualsWriter {
 public:
  virtual ~IndividualsWriter() {}
  virtual void Write(const Individual& individual) = 0;
  virtual void Write(const std::vector<Individual>& individuals) = 0;
  virtual void Write(
      const ::google::protobuf::RepeatedPtrField<Individual>& individuals) = 0;
};

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_INDIVIDUALS_WRITER_H_
