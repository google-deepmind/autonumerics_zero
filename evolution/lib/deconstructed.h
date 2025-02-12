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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_DECONSTRUCTED_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_DECONSTRUCTED_H_

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"

namespace brain {
namespace evolution {

// Ensures necessary conventions are satisfied. If any of these crash, the bit
// manipulations in this library may not be valid.
void CheckFloat64Assumptions();
void CheckFloat32Assumptions();
void CheckUInt64Assumptions();
void CheckUInt32Assumptions();

// A class to deconstruct the binary representation of a `double` according to
// the IEEE float64 standard.
class DeconstructedFloat64 {
 public:
  explicit DeconstructedFloat64(double value);

  // Builds from an exact representation. Subnormals not supported. To use from
  // Python, clif BuildExactDeconstructedFloat64.
  DeconstructedFloat64(
      // A signed string with 52 bits. In addition, there can be a "-" prefix to
      // indicate a negative number.
      absl::string_view signed_binary_significand,
      // The exponent. Must be in [-1022, 1023].
      intmax_t signed_decimal_exponent);

  DeconstructedFloat64(const DeconstructedFloat64& other);
  DeconstructedFloat64& operator=(const DeconstructedFloat64& other);

  uintmax_t GetSign() const;
  uintmax_t GetSignificand() const;
  uintmax_t GetExponent() const;

  // The signed exponent. Crashes if this deconstructed float represents zero.
  intmax_t GetSignedDecimalExponent() const;

  bool IsZero() const;
  bool IsNormal() const;

  // The inverse of the constructor.
  double ToDouble() const;

  // Returns an exact mixed representation of the number as a string of the
  // form: B * 2^(D), where B is a signed binary number and D is a signed
  // decimal exponent. B's sign is always explicit and all significand bits
  // are displayed, so as to keep lists aligned. Examples:
  // +1.001001010 x 2^(-30), -1.01 x 2^(5), +0 x 2^(0).
  std::string AsMixedString() const;

 private:
  uintmax_t sign_;
  uintmax_t significand_;
  uintmax_t exponent_;
};

// A class to deconstruct the binary representation of a `float` according to
// the IEEE float32 standard.
class DeconstructedFloat32 {
 public:
  explicit DeconstructedFloat32(float value);

  // Builds from an exact representation. Subnormals not supported. To use from
  // Python, clif BuildExactDeconstructedFloat64.
  DeconstructedFloat32(
      // A signed string with 23 bits. In addition, there can be a "-" prefix to
      // indicate a negative number.
      absl::string_view signed_binary_significand,
      // The exponent. Must be in [-126, 127].
      intmax_t signed_decimal_exponent);

  DeconstructedFloat32(const DeconstructedFloat32& other);
  DeconstructedFloat32& operator=(const DeconstructedFloat32& other);

  uintmax_t GetSign() const;
  uintmax_t GetSignificand() const;
  uintmax_t GetExponent() const;

  // The signed exponent. Crashes if this deconstructed float represents zero.
  intmax_t GetSignedDecimalExponent() const;

  bool IsZero() const;
  bool IsNormal() const;

  // The inverse of the constructor.
  float ToFloat() const;

  // Returns an exact mixed representation of the number as a string of the
  // form: B * 2^(D), where B is a signed binary number and D is a signed
  // decimal exponent. B's sign is always explicit and all significand bits
  // are displayed, so as to keep lists aligned. Examples:
  // +1.001001010 x 2^(-30), -1.01 x 2^(5), +0 x 2^(0).
  std::string AsMixedString() const;

 private:
  uintmax_t sign_;
  uintmax_t significand_;
  uintmax_t exponent_;
};

// A class to deconstruct the binary representation of a `float` according to
// the Google's bfloat16 standard.
class DeconstructedBFloat16 {
 public:
  explicit DeconstructedBFloat16(float value);

  // Builds from an exact representation. Subnormals not supported. To use from
  // Python, clif BuildExactDeconstructedBFloat16.
  DeconstructedBFloat16(
      // A signed string with 7 bits. In addition, there can be a "-" prefix to
      // indicate a negative number.
      absl::string_view signed_binary_significand,
      // The exponent. Must be in [-126, 127].
      intmax_t signed_decimal_exponent);

  DeconstructedBFloat16(const DeconstructedBFloat16& other);
  DeconstructedBFloat16& operator=(const DeconstructedBFloat16& other);

  uintmax_t GetSign() const;
  uintmax_t GetSignificand() const;
  uintmax_t GetExponent() const;

  // The signed exponent. Crashes if this deconstructed float represents zero.
  intmax_t GetSignedDecimalExponent() const;

  bool IsZero() const;
  bool IsNormal() const;

  // The inverse of the constructor.
  float ToFloat() const;

  // Returns an exact mixed representation of the number as a string of the
  // form: B * 2^(D), where B is a signed binary number and D is a signed
  // decimal exponent. B's sign is always explicit and all significand bits
  // are displayed, so as to keep lists aligned. Examples:
  // +1.0010010 x 2^(-30), -1.0100000 x 2^(5), +0 x 2^(0).
  std::string AsMixedString() const;

 private:
  uintmax_t sign_;
  uintmax_t significand_;
  uintmax_t exponent_;
};

// A rough approximation of 1 ULP for the given number `x`. `x` is interpreted
// as a float64 (we expect x to be a float64 or float80 casted from a float64).
// Subnormals are not supported and will cause a crash.
template <typename F>
F RoughFloat64Ulp(F x) {
  static_assert(
      std::is_same<F, double>::value || std::is_same<F, long double>::value,
      "Wrong type.");
  if (x == 0.0) {
    // 1074 = 1022 (offset for subnormals) + 52 (number of significand bits).
    return pow(2.0, -1074);
  } else if (std::isnan(x)) {
    return std::numeric_limits<F>::quiet_NaN();
  } else if (std::isinf(x)) {
    return std::numeric_limits<F>::infinity();
  } else {
    DeconstructedFloat64 z(x);
    if (z.IsNormal()) {
      intmax_t e = z.GetSignedDecimalExponent();
      e -= 52;  // For the 52 bits in the float64 significand.
      return pow(2.0, static_cast<F>(e));
    } else {
      LOG(FATAL) << "Subnormals not supported yet.";
    }
  }
}

// A rough approximation of 1 ULP for the given number `x`. `x` is interpreted
// as a float32 (we expect x to be a float32 or a float64/80 casted from a
// float32). Subnormals are not supported and will cause a crash.
template <typename F>
F RoughFloat32Ulp(F x) {
  static_assert(std::is_same<F, float>::value ||
                    std::is_same<F, double>::value ||
                    std::is_same<F, long double>::value,
                "Wrong type.");
  if (x == 0.0) {
    // 149 = 126 (offset for subnormals) + 23 (number of significand bits).
    return pow(2.0, -149);
  } else if (std::isnan(x)) {
    return std::numeric_limits<F>::quiet_NaN();
  } else if (std::isinf(x)) {
    return std::numeric_limits<F>::infinity();
  } else {
    float y = static_cast<float>(x);
    DeconstructedFloat32 z(y);
    if (z.IsNormal()) {
      intmax_t e = z.GetSignedDecimalExponent();
      e -= 23;  // For the 23 bits in the float32 significand.
      return pow(2.0, static_cast<F>(e));
    } else {
      LOG(FATAL) << "Subnormals not supported yet.";
    }
  }
}

// A rough approximation of 1 ULP for the given number `x`. `x` is interpreted
// as a bfloat16 (we expect x just came from casting a jnp.bfloat16 to a
// float64 or a float80). Subnormals are not supported and will cause a crash.
template <typename F>
F RoughBfloat16Ulp(F x) {
  static_assert(std::is_same<F, float>::value ||
                    std::is_same<F, double>::value ||
                    std::is_same<F, long double>::value,
                "Wrong type.");
  if (x == 0.0) {
    // 133 = 126 (offset for subnormals) + 7 (number of significand bits).
    return pow(2.0, -133);
  } else if (std::isnan(x)) {
    return std::numeric_limits<F>::quiet_NaN();
  } else if (std::isinf(x)) {
    return std::numeric_limits<F>::infinity();
  } else {
    DeconstructedFloat64 z(x);
    if (z.IsNormal()) {
      intmax_t e = z.GetSignedDecimalExponent();
      e -= 7;  // For the 7 bits in the bfloat16 significand.
      return pow(2.0, static_cast<F>(e));
    } else {
      LOG(FATAL) << "Subnormals not supported yet.";
    }
  }
}

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_DECONSTRUCTED_H_
