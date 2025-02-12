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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_ID_GENERATOR_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_ID_GENERATOR_H_

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "evolution/lib/rng.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

// To allow mocking.
class IDGeneratorInterface {
 public:
  virtual ~IDGeneratorInterface() {}
  virtual std::string Generate() = 0;
};

class IDGenerator final : public IDGeneratorInterface {
 public:
  IDGenerator(
      RNG* rng,
      // The number of characters the ID should have.
      Integer num_chars,
      // These values will be randomly mixed into the (random) individual ID.
      // For example, the worker_id can be passed as a value. This is useful
      // to avoid collisions across workers for large workers of collections
      // (large here means that the number of workers squared is not much
      // smaller that the number of possible 32-bit random seeds).
      absl::Span<const Integer> integers_to_mix,
      absl::Span<const std::string> strings_to_mix);
  IDGenerator(const IDGenerator& other) = delete;
  IDGenerator& operator=(const IDGenerator& other) = delete;
  std::string Generate() override;

 private:
  RNG* rng_;
  const Integer num_chars_;
  std::vector<uint32_t> uint32s_to_mix_;
};

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_ID_GENERATOR_H_
