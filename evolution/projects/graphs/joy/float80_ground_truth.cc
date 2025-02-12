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

#include "evolution/projects/graphs/joy/float80_ground_truth.h"

#include <cmath>
#include <limits>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "boost/math/special_functions/airy.hpp"
#include "boost/math/special_functions/bessel.hpp"
#include "evolution/lib/deconstructed.h"
#include "evolution/lib/types.h"
#include "evolution/projects/graphs/joy/data.pb.h"

namespace brain {
namespace evolution {
namespace graphs {
namespace joy {

using ::boost::math::airy_ai;
using ::std::abs;
using ::std::exp2;

Float80GroundTruth::Float80GroundTruth() : ulp_bounds_list_() {
  CHECK_EQ(std::numeric_limits<Float80>::digits, 64);
}
Float80GroundTruth::Float80GroundTruth(const std::vector<UlpBounds>& ulp_bounds)
    : ulp_bounds_list_(ulp_bounds) {
  CHECK_EQ(std::numeric_limits<Float80>::digits, 64);
}

Float80GroundTruth::~Float80GroundTruth() = default;

double Float80GroundTruth::Label(double x) {
  const Float80 x80 = static_cast<Float80>(x);
  const Float80 label = Func(x80);
  return static_cast<double>(label);
}
Float80 get_ulp_reference_point_for_input(
    Float80 input, absl::Span<const UlpBounds> ulp_bounds_list) {
  for (const UlpBounds& ulp_bounds : ulp_bounds_list) {
    if (ulp_bounds.is_in_bounds(input)) {
      return ulp_bounds.get_ulp_reference_point();
    }
  }
  LOG(FATAL) << "Unknown ULP value for input.";
}

double get_ulp_reference_point_for_input_as_double(
    double input, const std::vector<UlpBounds>& ulp_bounds_list) {
  return static_cast<double>(get_ulp_reference_point_for_input(
      static_cast<Float80>(input), ulp_bounds_list));
}

UlpBounds UlpBounds::UlpBoundsFromDoubles(double min_input, double max_input,
                                          double min_output,
                                          double max_output) {
  return UlpBounds(static_cast<Float80>(min_input),
                   static_cast<Float80>(min_output),
                   static_cast<Float80>(max_output),
                   static_cast<Float80>((min_output + max_output) / 2));
}

double Float80GroundTruth::SignedRelativeError(double x, double y,
                                               double epsilon,
                                               FloatDTypeSpec ulp_dtype_spec) {
  if (!std::isfinite(y)) {  // Catch infinities and NaN.
    // Assume the label is not infinite, so this should give infinite error.
    return std::numeric_limits<double>::infinity();
  }
  const Float80 x80 = static_cast<Float80>(x);
  const Float80 y80 = static_cast<Float80>(y);
  const Float80 epsilon80 = static_cast<Float80>(epsilon);
  const Float80 err80 =
      SignedRelativeErrorImpl(x80, y80, epsilon80, ulp_dtype_spec);
  return static_cast<double>(err80);
}

double Float80GroundTruth::SignedRelativeErrorWithUlpAt(
    double x, double y, double epsilon, FloatDTypeSpec ulp_dtype_spec,
    double ulp_at) {
  if (!std::isfinite(y)) {  // Catch infinities and NaN.
    // Assume the label is not infinite, so this should give infinite error.
    return std::numeric_limits<double>::infinity();
  }
  const Float80 x80 = static_cast<Float80>(x);
  const Float80 y80 = static_cast<Float80>(y);
  const Float80 epsilon80 = static_cast<Float80>(epsilon);
  Float80 ulp_at_f80 = static_cast<Float80>(ulp_at);
  const Float80 err80 = SignedRelativeErrorImpl(
      x80, y80, epsilon80, ulp_dtype_spec, nullptr, &ulp_at_f80);
  return static_cast<double>(err80);
}

Float80 Float80GroundTruth::SignedRelativeErrorImpl(
    Float80 x, Float80 y, Float80 epsilon, FloatDTypeSpec ulp_dtype_spec,
    Float80* override_ulp, Float80* override_ulp_with_value_at) {
  const Float80 label = Func(x);
  const Float80 numerator = label - y;
  Float80 ulp_reference_point = std::numeric_limits<Float80>::quiet_NaN();
  CHECK(override_ulp == nullptr || override_ulp_with_value_at == nullptr);
  CHECK(override_ulp_with_value_at == nullptr || ulp_dtype_spec != NODTYPE);
  if (override_ulp_with_value_at != nullptr) {
    ulp_reference_point = *override_ulp_with_value_at;
  } else if (ulp_bounds_list_.empty()) {
    ulp_reference_point = label;
  } else {
    ulp_reference_point =
        get_ulp_reference_point_for_input(x, ulp_bounds_list_);
  }
  if (numerator == 0.0L) return 0.0;
  Float80 denominator = std::numeric_limits<Float80>::quiet_NaN();
  if (override_ulp == nullptr) {
    if (ulp_dtype_spec == NODTYPE) {
      denominator = abs(label);
    } else if (ulp_dtype_spec == BFLOAT16) {
      denominator = RoughBfloat16Ulp(ulp_reference_point);
    } else if (ulp_dtype_spec == FLOAT32) {
      denominator = RoughFloat32Ulp(ulp_reference_point);
    } else if (ulp_dtype_spec == FLOAT64) {
      denominator = RoughFloat64Ulp(ulp_reference_point);
    } else {
      LOG(FATAL) << "Unsupported dtype.";
    }
  } else {
    denominator = *override_ulp;
  }
  denominator += epsilon;
  CHECK(denominator != 0.0L) << "Division by zero in relative error.";
  const Float80 error = numerator / denominator;
  return error;
}

Float80 Exp2Float80GroundTruth::Func(Float80 x) { return exp2(x); }

Float80 ExpeFloat80GroundTruth::Func(Float80 x) { return exp(x); }

Float80 Log2Float80GroundTruth::Func(Float80 x) { return log2(x); }

Float80 LogeFloat80GroundTruth::Func(Float80 x) { return log(x); }

Float80 ErfFloat80GroundTruth::Func(Float80 x) { return erf(x); }

Float80 ErfcFloat80GroundTruth::Func(Float80 x) { return erfc(x); }

WavyFloat80GroundTruth::WavyFloat80GroundTruth(const int num_periods)
    : num_periods_(static_cast<Float80>(num_periods)) {}

Float80 WavyFloat80GroundTruth::Func(Float80 x) {
  return 1.0L + boost::math::airy_ai(-num_periods_ * x);
}

AiryFloat80GroundTruth::AiryFloat80GroundTruth(const int input_scaling)
    : input_scaling_(static_cast<Float80>(input_scaling)) {}

Float80 AiryFloat80GroundTruth::Func(Float80 x) {
  return boost::math::airy_ai(-input_scaling_ * x);
}

Float80 BesselFloat80GroundTruth::Func(Float80 x) {
  return boost::math::cyl_bessel_i(0.5, x);
}

// Taylor coefficients for 2^x to roughly float80 precision.
// Computed in 2023-07-06.roughly_float80_coefficients_for_exp2.nb
// (https://drive.google.com/file/d/1CIykrd06oM9BGGRBQDzkc8e0dfuIgLcL/view?usp=drive_link)
// then converted to hex with:
//   Float80 v = <computed_value_to_100_sig_figs>L;
//   std::cout << std::hexfloat << v << std::endl;
constexpr Integer kExpCoeffsSize = 21;
constexpr Float80 kExp2Coeffs[] = {
    0x8p-3L,  // Coefficient of x^0.
    0xb.17217f7d1cf79acp-4L,
    0xf.5fdeffc162c7543p-6L,
    0xe.35846b82505fc5ap-8L,
    0x9.d955b7dd273b94ep-10L,
    0xa.ec3ff3c53398884p-13L,
    0xa.184897c363c3b7ap-16L,
    0xf.fe5fe2c45863436p-20L,
    0xb.160111d2e411fecp-23L,
    0xd.a929e9caf3e1ed2p-27L,
    0xf.267a8ac5c764fb8p-31L,
    0xf.465639a8dd92608p-35L,
    0xe.1deb287e14c2f16p-39L,
    0xc.0b0c98b3687cb14p-43L,
    0x9.8a4b26ac3c54bap-47L,
    0xe.1b7421d82010f34p-52L,
    0x9.c744d73cfc59c92p-56L,
    0xc.c2225a0e12d3eabp-61L,
    0xf.b8bb5eda1b4aebap-66L,
    0x9.2d3f65c1ae326d7p-70L,
    0xa.2d6625a8289ac27p-75L};  // Coefficient of x^20.

Float80 Exp2VettedGroundTruth::Func(Float80 x) {
  // Compute 2^x using Horner's scheme.
  Integer i = kExpCoeffsSize - 1;
  Float80 r = kExp2Coeffs[i];
  while (--i >= 0) {
    r *= x;
    r += kExp2Coeffs[i];
  }
  return r;
}

}  // namespace joy
}  // namespace graphs
}  // namespace evolution
}  // namespace brain
