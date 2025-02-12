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

#include "evolution/lib/hashing.h"

#include "absl/log/check.h"

namespace brain {
namespace evolution {

using ::std::vector;

constexpr size_t kSignMaskComplement =
    0b0111111111111111111111111111111111111111111111111111111111111111;

constexpr size_t kExponentMaskComplement =
    0b1000000000001111111111111111111111111111111111111111111111111111;

constexpr size_t kPositiveZero =
    0b0000000000000000000000000000000000000000000000000000000000000000;

constexpr size_t kNegativeZero =
    0b1000000000000000000000000000000000000000000000000000000000000000;

constexpr size_t kSpecialExponent = 0b11111111111;

DeconstructedDouble::DeconstructedDouble(double value,
                                         const size_t mantissa_bits) {
  size_t temp = *reinterpret_cast<size_t*>(&value);

  // Canonicalize zero.
  if (temp == kNegativeZero) {
    temp = kPositiveZero;
  }

  // Extract sign bit and zero it out.
  sign_ = temp >> 63;
  temp &= kSignMaskComplement;

  // Extract exponent bits and zero them out.
  exponent_ = temp >> 52;
  temp &= kExponentMaskComplement;

  // Extract mantissa and truncate.
  CHECK_LE(mantissa_bits, 52);
  mantissa_ = temp >> (52 - mantissa_bits);

  // Handle special values.
  if (exponent_ == kSpecialExponent) {  // Infinities and NaNs.
    hashable_ = false;
  } else {
    hashable_ = true;
  }
}

bool DeconstructedDouble::Hashable() { return hashable_; }

size_t DeconstructedDouble::Sign() {
  CHECK(hashable_);
  return sign_;
}

size_t DeconstructedDouble::Exponent() {
  CHECK(hashable_);
  return exponent_;
}

size_t DeconstructedDouble::TruncatedMantissa() {
  CHECK(hashable_);
  return mantissa_;
}

}  // namespace evolution
}  // namespace brain
