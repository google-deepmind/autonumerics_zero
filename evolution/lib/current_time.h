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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_CURRENT_TIME_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_CURRENT_TIME_H_

#include <memory>

#include "absl/time/time.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

// To allow mocking.
class CurrentTimeInterface {
 public:
  virtual ~CurrentTimeInterface() {}
  virtual Integer Nanos() const = 0;
  virtual absl::Time Now() const = 0;
};

// Thread-safe.
class CurrentTime final : public CurrentTimeInterface {
 public:
  CurrentTime();
  CurrentTime(const CurrentTime& other) = delete;
  CurrentTime& operator=(const CurrentTime& other) = delete;
  Integer Nanos() const override;
  absl::Time Now() const override;
};

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_CURRENT_TIME_H_
