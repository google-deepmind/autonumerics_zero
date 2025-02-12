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

// Hashing-related utilities.

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_HASHING_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_HASHING_H_

#include <sched.h>

#include <cstddef>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

constexpr size_t kUnhashable = 0;

// Hash-mixes a vector of numbers. The numbers must be of a type that can be
// casted to a size_t (it must be unsigned and it must have <= 64 bits).
// Intended to be used with the RNGSeed type.
template <typename NumberT>
NumberT Mix(const std::vector<NumberT>& numbers) {
  for (const NumberT number : numbers) {
    CHECK_GE(number, static_cast<NumberT>(0.0));
  }
  return absl::HashOf(numbers);
}

// Hash-mixes two numbers. The numbers must be of a type that can be
// casted to a size_t (it must be unsigned and it must have <= 64 bits).
// Intended to be used with the RNGSeed type.
template <typename NumberT>
NumberT Mix(NumberT first, NumberT second) {
  return Mix<NumberT>({first, second});
}

// Avoid overloading to use with PyClif. See restrictions for `Mix`.
inline Integer IntegerMix(const std::vector<Integer>& numbers) {
  return Mix(numbers);
}

// Converts floating point numbers to fixed precision and combines them into
// an integer hash. The numbers must be non-negative or else this method will
// crash by design. For the hash to be effective, most of the numbers must be
// significantly larger than 1/2^48. Avoid overloading to use with PyClif.
inline size_t FloatMix(const std::vector<double>& numbers) {
  constexpr Integer precision_bits = 48;
  const double multiplier =
      static_cast<double>(static_cast<Integer>(1) << precision_bits);
  std::vector<size_t> numbers_fixed;
  numbers_fixed.reserve(numbers.size());
  for (double number_float : numbers) {
    CHECK_GE(number_float, 0.0);
    CHECK_LT(number_float, 1.0e4);  // Prevent overflow.
    const size_t number_fixed = static_cast<size_t>(number_float * multiplier);
    numbers_fixed.push_back(number_fixed);
  }
  return absl::HashOf(numbers_fixed);
}

// Deprecated. For new code, use the `Deconstructed` classes in deconstructed.h.
class DeconstructedDouble {
 public:
  explicit DeconstructedDouble(double value, size_t mantissa_bits);

  bool Hashable();

  // Returns the sign bit (1 for negative numbers, 0 otherwise).
  size_t Sign();

  // Returns the exponent exactly.
  size_t Exponent();

  // Returns the mantissa, truncated to the given number of bits. Truncation is
  // done by replacing least-significant bits with zeros.
  size_t TruncatedMantissa();

 private:
  size_t sign_;
  size_t exponent_;
  size_t mantissa_;
  bool hashable_;
};


}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_HASHING_H_
