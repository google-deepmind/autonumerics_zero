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

#include "evolution/lib/deconstructed.h"

#include <bitset>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

using ::std::bitset;
using std::is_same;
using std::numeric_limits;
using ::std::string;
using ::std::stringstream;

// Must be a `uint64_t` because it must have the same size as the `double` type.
constexpr uint64_t kFloat64SignMaskComplement =
    0b0111111111111111111111111111111111111111111111111111111111111111UL;

// Must be a `uint64_t` because it must have the same size as the `double` type.
constexpr uint64_t kFloat64ExponentMaskComplement =
    0b1000000000001111111111111111111111111111111111111111111111111111UL;

// Must be a `uint32_t` because it must have the same size as the `float` type.
constexpr uint32_t kFloat32SignMaskComplement =
    0b01111111111111111111111111111111U;

// Must be a `uint32_t` because it must have the same size as the `float` type.
constexpr uint32_t kFloat32ExponentMaskComplement =
    0b10000000011111111111111111111111U;

constexpr uint32_t kBFloat16SignificandMaskCheck =
    0b00000000000000001111111111111111U;

void CheckFloat64Assumptions() {
  CHECK_EQ(numeric_limits<double>::lowest(), -1.7976931348623157e308);
  CHECK_EQ(numeric_limits<double>::denorm_min(), 4.9406564584124654e-324);
  CHECK_EQ(numeric_limits<double>::min(), 2.2250738585072014e-308);
  CHECK_EQ(numeric_limits<double>::max(), 1.7976931348623157e308);
}

void CheckFloat32Assumptions() {
  CHECK_EQ(numeric_limits<float>::lowest(), -3.4028234664e38F);
  CHECK_EQ(numeric_limits<float>::denorm_min(), 1.4012984643e-45F);
  CHECK_EQ(numeric_limits<float>::min(), 1.1754943508e-38F);
  CHECK_EQ(numeric_limits<float>::max(), 3.4028234664e38F);
}

void CheckUInt64Assumptions() {
  // We use uint64_t for bit manipulations of the float64 type.
  CHECK_EQ(numeric_limits<uint64_t>::min(), 0);
  CHECK_EQ(
      numeric_limits<uint64_t>::max(),
      0b1111111111111111111111111111111111111111111111111111111111111111UL);

  // In this library, we use the UL suffix to specify uint64_t literals.
  CHECK((is_same<unsigned long, uint64_t>::value));  // NOLINT
}

void CheckUInt32Assumptions() {
  // We use uint32_t for bit manipulations of the float32 type.
  CHECK_EQ(numeric_limits<uint32_t>::min(), 0);
  CHECK_EQ(numeric_limits<uint32_t>::max(),
           0b11111111111111111111111111111111UL);

  // In this library, we use the U suffix to specify uint32_t literals.
  CHECK((is_same<unsigned int, uint32_t>::value));  // NOLINT
}

DeconstructedFloat64::DeconstructedFloat64(double value) {
  // Must be a `uint64_t` because it must have the same size as a `double`.
  uint64_t temp = absl::bit_cast<uint64_t>(value);

  // Extract sign bit and zero it out.
  sign_ = static_cast<uintmax_t>(temp >> 63);
  temp &= kFloat64SignMaskComplement;

  // Extract exponent bits and zero them out.
  exponent_ = static_cast<uintmax_t>(temp >> 52);
  temp &= kFloat64ExponentMaskComplement;

  // Extract significand.
  significand_ = static_cast<uintmax_t>(temp);

  CHECK(exponent_ != 0b11111111111ULL) << "Infinity and nan are not supported.";
}

DeconstructedFloat64::DeconstructedFloat64(
    absl::string_view signed_binary_significand,
    const intmax_t signed_decimal_exponent) {
  // Get the sign.
  Integer significand_start = 0;
  if (signed_binary_significand[0] == '-') {
    sign_ = 1ULL;
    significand_start = 1;
  } else {
    sign_ = 0;
  }

  // Get the significand.
  CHECK_EQ(signed_binary_significand.size(), significand_start + 52)
      << "Wrong number of bits in the significand.";
  bitset<52> significand_bits;
  for (Integer bit = 0; bit < 52; ++bit) {
    if (signed_binary_significand[significand_start + bit] == '0') {
      significand_bits[52 - bit - 1] = false;  // 0.
    } else if (signed_binary_significand[significand_start + bit] == '1') {
      significand_bits[52 - bit - 1] = true;  // 1.
    } else {
      LOG(FATAL) << "Invalid significand string.";
    }
  }
  significand_ = significand_bits.to_ullong();

  // Get the exponent.
  CHECK(signed_decimal_exponent >= -1022 && signed_decimal_exponent <= 1023)
      << "Exponent out of bounds.";
  exponent_ = static_cast<uintmax_t>(signed_decimal_exponent + 1023);
}

DeconstructedFloat64::DeconstructedFloat64(const DeconstructedFloat64& other) {
  sign_ = other.sign_;
  significand_ = other.significand_;
  exponent_ = other.exponent_;
}

DeconstructedFloat64& DeconstructedFloat64::operator=(
    const DeconstructedFloat64& other) {
  sign_ = other.sign_;
  significand_ = other.significand_;
  exponent_ = other.exponent_;
  return *this;
}

uintmax_t DeconstructedFloat64::GetSign() const { return sign_; }

uintmax_t DeconstructedFloat64::GetSignificand() const { return significand_; }

uintmax_t DeconstructedFloat64::GetExponent() const { return exponent_; }

intmax_t DeconstructedFloat64::GetSignedDecimalExponent() const {
  CHECK(!IsZero());
  intmax_t signed_exponent = static_cast<intmax_t>(exponent_);
  signed_exponent -= 1023;
  return signed_exponent;
}

bool DeconstructedFloat64::IsZero() const {
  return exponent_ == 0 && significand_ == 0;
}

bool DeconstructedFloat64::IsNormal() const {
  return exponent_ != 0 || significand_ == 0;
}

double DeconstructedFloat64::ToDouble() const {
  uint64_t temp = (sign_ << 63) | significand_ | (exponent_ << 52);
  double result = absl::bit_cast<double>(temp);
  return result;
}

string DeconstructedFloat64::AsMixedString() const {
  std::stringstream stream;
  if (sign_ == 1ULL) {
    stream << "-";
  } else {
    stream << "+";
  }
  if (IsZero()) {
    stream << "0.0000000000000000000000000000000000000000000000000000 x 2^(0)";
  } else {
    if (IsNormal()) {
      stream << "1.";
      bitset<52> significand_bits(significand_);
      stream << significand_bits;
    } else {
      LOG(FATAL) << "Subnormals not supported yet.";
    }
    stream << " x 2^(";
    const intmax_t signed_exponent = GetSignedDecimalExponent();
    stream << signed_exponent << ")";
  }
  return stream.str();
}

DeconstructedFloat32::DeconstructedFloat32(float value) {
  // Must be a `uint32_t` because it must have the same size as a `float`.
  uint32_t temp = absl::bit_cast<uint32_t>(value);

  // Extract sign bit and zero it out.
  sign_ = static_cast<uintmax_t>(temp >> 31);
  temp &= kFloat32SignMaskComplement;

  // Extract exponent bits and zero them out.
  exponent_ = static_cast<uintmax_t>(temp >> 23);
  temp &= kFloat32ExponentMaskComplement;

  // Extract significand.
  significand_ = static_cast<uintmax_t>(temp);

  CHECK(exponent_ != 0b11111111ULL) << "Infinity and nan are not supported.";
}

DeconstructedFloat32::DeconstructedFloat32(
    absl::string_view signed_binary_significand,
    const intmax_t signed_decimal_exponent) {
  // Get the sign.
  Integer significand_start = 0;
  if (signed_binary_significand[0] == '-') {
    sign_ = 1ULL;
    significand_start = 1;
  } else {
    sign_ = 0;
    significand_start = 0;
  }

  // Get the significand.
  CHECK_EQ(signed_binary_significand.size(), significand_start + 23)
      << "Wrong number of bits in the significand.";
  bitset<52> significand_bits;
  for (Integer bit = 0; bit < 23; ++bit) {
    if (signed_binary_significand[significand_start + bit] == '0') {
      significand_bits[23 - bit - 1] = false;  // 0.
    } else if (signed_binary_significand[significand_start + bit] == '1') {
      significand_bits[23 - bit - 1] = true;  // 1.
    } else {
      LOG(FATAL) << "Invalid significand string.";
    }
  }
  significand_ = significand_bits.to_ullong();

  // Get the exponent.
  CHECK(signed_decimal_exponent >= -126 && signed_decimal_exponent <= 127)
      << "Exponent out of bounds.";
  exponent_ = static_cast<uintmax_t>(signed_decimal_exponent + 127);
}

DeconstructedFloat32::DeconstructedFloat32(const DeconstructedFloat32& other) {
  sign_ = other.sign_;
  significand_ = other.significand_;
  exponent_ = other.exponent_;
}

DeconstructedFloat32& DeconstructedFloat32::operator=(
    const DeconstructedFloat32& other) {
  sign_ = other.sign_;
  significand_ = other.significand_;
  exponent_ = other.exponent_;
  return *this;
}

uintmax_t DeconstructedFloat32::GetSign() const { return sign_; }

uintmax_t DeconstructedFloat32::GetSignificand() const { return significand_; }

uintmax_t DeconstructedFloat32::GetExponent() const { return exponent_; }

intmax_t DeconstructedFloat32::GetSignedDecimalExponent() const {
  CHECK(!IsZero());
  intmax_t signed_exponent = static_cast<intmax_t>(exponent_);
  signed_exponent -= 127;
  return signed_exponent;
}

bool DeconstructedFloat32::IsZero() const {
  return exponent_ == 0 && significand_ == 0;
}

bool DeconstructedFloat32::IsNormal() const {
  return exponent_ != 0 || significand_ == 0;
}

float DeconstructedFloat32::ToFloat() const {
  uint32_t temp = (sign_ << 31) | significand_ | (exponent_ << 23);
  float result = absl::bit_cast<float>(temp);
  return result;
}

string DeconstructedFloat32::AsMixedString() const {
  std::stringstream stream;
  if (sign_ == 1ULL) {
    stream << "-";
  } else {
    stream << "+";
  }
  if (IsZero()) {
    stream << "0.00000000000000000000000 x 2^(0)";
  } else {
    if (IsNormal()) {
      stream << "1.";
      bitset<23> significand_bits(significand_);
      stream << significand_bits;
    } else {
      LOG(FATAL) << "Subnormals not supported yet.";
    }
    stream << " x 2^(";
    const intmax_t signed_exponent = GetSignedDecimalExponent();
    stream << signed_exponent << ")";
  }
  return stream.str();
}

DeconstructedBFloat16::DeconstructedBFloat16(float value) {
  // Must be a `uint32_t` because it must have the same size as a `float`.
  uint32_t temp = absl::bit_cast<uint32_t>(value);

  // Extract sign bit and zero it out.
  sign_ = static_cast<uintmax_t>(temp >> 31);
  temp &= kFloat32SignMaskComplement;

  // Extract exponent bits and zero them out.
  exponent_ = static_cast<uintmax_t>(temp >> 23);
  temp &= kFloat32ExponentMaskComplement;

  // Extract significand for bfloat16 by truncating the float32 significand to
  // 7 bits.
  CHECK((temp & kBFloat16SignificandMaskCheck) == 0)
      << "Input significand's last 16 bits non zero.";
  significand_ = static_cast<uintmax_t>(temp) >> 16;

  CHECK(exponent_ != 0b11111111ULL) << "Infinity and nan are not supported.";
}

DeconstructedBFloat16::DeconstructedBFloat16(
    absl::string_view signed_binary_significand,
    const intmax_t signed_decimal_exponent) {
  // Get the sign.
  Integer significand_start = 0;
  if (signed_binary_significand[0] == '-') {
    sign_ = 1ULL;
    significand_start = 1;
  } else {
    sign_ = 0;
    significand_start = 0;
  }

  // Get the significand.
  CHECK_EQ(signed_binary_significand.size(), significand_start + 7)
      << "Wrong number of bits in the significand.";
  bitset<52> significand_bits;
  for (Integer bit = 0; bit < 7; ++bit) {
    if (signed_binary_significand[significand_start + bit] == '0') {
      significand_bits[7 - bit - 1] = false;  // 0.
    } else if (signed_binary_significand[significand_start + bit] == '1') {
      significand_bits[7 - bit - 1] = true;  // 1.
    } else {
      LOG(FATAL) << "Invalid significand string.";
    }
  }
  significand_ = significand_bits.to_ullong();

  // Get the exponent.
  CHECK(signed_decimal_exponent >= -126 && signed_decimal_exponent <= 127)
      << "Exponent out of bounds.";
  exponent_ = static_cast<uintmax_t>(signed_decimal_exponent + 127);
}

DeconstructedBFloat16::DeconstructedBFloat16(
    const DeconstructedBFloat16& other) {
  sign_ = other.sign_;
  significand_ = other.significand_;
  exponent_ = other.exponent_;
}

DeconstructedBFloat16& DeconstructedBFloat16::operator=(
    const DeconstructedBFloat16& other) {
  sign_ = other.sign_;
  significand_ = other.significand_;
  exponent_ = other.exponent_;
  return *this;
}

uintmax_t DeconstructedBFloat16::GetSign() const { return sign_; }

uintmax_t DeconstructedBFloat16::GetSignificand() const { return significand_; }

uintmax_t DeconstructedBFloat16::GetExponent() const { return exponent_; }

intmax_t DeconstructedBFloat16::GetSignedDecimalExponent() const {
  CHECK(!IsZero());
  intmax_t signed_exponent = static_cast<intmax_t>(exponent_);
  signed_exponent -= 127;
  return signed_exponent;
}

bool DeconstructedBFloat16::IsZero() const {
  return exponent_ == 0 && significand_ == 0;
}

bool DeconstructedBFloat16::IsNormal() const {
  return exponent_ != 0 || significand_ == 0;
}

float DeconstructedBFloat16::ToFloat() const {
  uint32_t temp = (sign_ << 31) | (significand_ << 16) | (exponent_ << 23);
  float result = absl::bit_cast<float>(temp);
  return result;
}

string DeconstructedBFloat16::AsMixedString() const {
  std::stringstream stream;
  if (sign_ == 1ULL) {
    stream << "-";
  } else {
    stream << "+";
  }
  if (IsZero()) {
    stream << "0.0000000 x 2^(0)";
  } else {
    if (IsNormal()) {
      stream << "1.";
      bitset<7> significand_bits(significand_);
      stream << significand_bits;
    } else {
      LOG(FATAL) << "Subnormals not supported yet.";
    }
    stream << " x 2^(";
    const intmax_t signed_exponent = GetSignedDecimalExponent();
    stream << signed_exponent << ")";
  }
  return stream.str();
}

}  // namespace evolution
}  // namespace brain
