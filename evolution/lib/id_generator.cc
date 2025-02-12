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

#include "evolution/lib/id_generator.h"

#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "evolution/lib/hashing.h"
#include "evolution/lib/rng.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

using ::std::string;
using ::std::vector;

// Converts a value to a uint32_t that can be used for mixing.
uint32_t AdjustForMixing(Integer value);
uint32_t AdjustForMixing(absl::string_view value);

// Assigns each value in [0, 62) with a character that can be used in a
// suffix for an individual ID.
constexpr Integer kNumChars = 62;
char IntegerToChar(Integer char_index);

IDGenerator::IDGenerator(RNG* rng, const Integer num_chars,
                         absl::Span<const Integer> integers_to_mix,
                         absl::Span<const string> strings_to_mix)
    : rng_(rng), num_chars_(num_chars) {
  for (Integer integer_to_mix : integers_to_mix) {
    uint32s_to_mix_.push_back(AdjustForMixing(integer_to_mix));
  }
  for (const string& string_to_mix : strings_to_mix) {
    uint32s_to_mix_.push_back(AdjustForMixing(string_to_mix));
  }
}

std::string IDGenerator::Generate() {
  vector<char> random_chars;
  for (int i = 0; i < num_chars_; ++i) {
    // Just a random uint32_t.
    static_assert(std::is_same<RNGSeed, uint32_t>::value,
                  "Internal error: type mismatch.");
    uint32_t random_int = rng_->UniformRNGSeed();

    // Mix requested values.
    for (uint32_t value : uint32s_to_mix_) {
      random_int = Mix(random_int, value);
    }
    char random_char = IntegerToChar(random_int % kNumChars);
    random_chars.push_back(random_char);
  }
  return string(random_chars.begin(), random_chars.end());
}

uint32_t AdjustForMixing(Integer value) {
  // Move away from special numbers. We must at least avoid -1 (which is passed
  // can be passed when this is a server or a local run instead of a worker in
  // a collection).
  value += 3;
  CHECK_GE(value, 2);
  CHECK(value < std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(value);
}

uint32_t AdjustForMixing(absl::string_view value) {
  uint64_t hash = absl::HashOf(value);
  uint64_t limit = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(hash % limit);
}

char IntegerToChar(const Integer char_index) {
  if (char_index < 26) {
    return char_index + 97;  // Maps 0-25 to 'a'-'z'.
  } else if (char_index < 52) {
    return char_index - 26 + 65;  // Maps 26-51 to 'A'-'Z'.
  } else if (char_index < 62) {
    return char_index - 52 + 48;  // Maps 52-61 to '0'-'9'.
  } else {
    LOG(FATAL) << "char_index too large." << std::endl;
  }
}

}  // namespace evolution
}  // namespace brain
