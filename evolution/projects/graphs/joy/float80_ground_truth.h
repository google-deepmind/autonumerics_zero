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

// Ground truth functions that compute in float80.

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_PROJECTS_GRAPHS_JOY_FLOAT80_GROUND_TRUTH_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_PROJECTS_GRAPHS_JOY_FLOAT80_GROUND_TRUTH_H_

#include <vector>

#include "absl/types/span.h"
#include "evolution/projects/graphs/joy/data.pb.h"

namespace brain {
namespace evolution {
namespace graphs {
namespace joy {

typedef long double Float80;

struct UlpBounds {
 public:
  constexpr UlpBounds(Float80 min_input, Float80 max_input, Float80 min_output,
                      Float80 max_output)
      : min_input(min_input),
        max_input(max_input),
        ulp_reference_point((min_output + max_output) / 2) {}

  static UlpBounds UlpBoundsFromDoubles(double min_input, double max_input,
                                        double min_output, double max_output);
  Float80 get_ulp_reference_point() const { return ulp_reference_point; }
  double get_ulp_reference_point_as_double() const {
    return static_cast<double>(get_ulp_reference_point());
  }
  bool is_in_bounds(Float80 input) const {
    return min_input <= input && input < max_input;
  }

 private:
  Float80 min_input;
  Float80 max_input;
  Float80 ulp_reference_point;
};

Float80 get_ulp_reference_point_for_input(
    Float80 input, absl::Span<const UlpBounds> ulp_bounds_list);

double get_ulp_reference_point_for_input_as_double(
    double input, const std::vector<UlpBounds>& ulp_bounds_list);

class Float80GroundTruth {
 public:
  Float80GroundTruth();
  virtual ~Float80GroundTruth();

  // Returns the label, after doing internal computations in float80 and
  // rounding to float64 at the end.
  double Label(double x);

  // Returns the signed error, after doing internal computations in float80 and
  // rounding to float64 at the end. `epsilon` is a constant to avoid
  // dividing by zero when calculating the relative error.
  double SignedRelativeError(double x, double y, double epsilon,
                             FloatDTypeSpec ulp_dtype_spec);
  double SignedRelativeErrorWithUlpAt(double x, double y, double epsilon,
                                      FloatDTypeSpec ulp_dtype_spec,
                                      double ulp_at);

  // Computes the signed relative error in float80. Does not check for
  // infinities.
  Float80 SignedRelativeErrorImpl(
      Float80 x, Float80 y, Float80 epsilon, FloatDTypeSpec ulp_dtype_spec,
      // Cannot be specified with `override_ulp_with_value_at`.
      Float80* override_ulp = nullptr,
      // Cannot be specified with `override_ulp`.
      Float80* override_ulp_with_value_at = nullptr);

 protected:
  explicit Float80GroundTruth(const std::vector<UlpBounds>& ulp_bounds);
  const std::vector<UlpBounds> ulp_bounds_list_;

 private:
  virtual Float80 Func(Float80 x) = 0;
};

class Exp2Float80GroundTruth : public Float80GroundTruth {
 private:
  Float80 Func(Float80 x) override;
};

class ExpeFloat80GroundTruth : public Float80GroundTruth {
 private:
  Float80 Func(Float80 x) override;
};

class Log2Float80GroundTruth : public Float80GroundTruth {
 private:
  Float80 Func(Float80 x) override;
};

class LogeFloat80GroundTruth : public Float80GroundTruth {
 private:
  Float80 Func(Float80 x) override;
};

class ErfFloat80GroundTruth : public Float80GroundTruth {
 private:
  Float80 Func(Float80 x) override;
};

class ErfcFloat80GroundTruth : public Float80GroundTruth {
 private:
  Float80 Func(Float80 x) override;
};

class WavyFloat80GroundTruth : public Float80GroundTruth {
 public:
  explicit WavyFloat80GroundTruth(int num_periods);

 private:
  Float80 Func(Float80 x) override;

  const Float80 num_periods_;
};

class AiryFloat80GroundTruth : public Float80GroundTruth {
 public:
  explicit AiryFloat80GroundTruth(int input_scaling);

 private:
  Float80 Func(Float80 x) override;

  const Float80 input_scaling_;
};

// The modified Bessel function of the first kind of order 1/2.
class BesselFloat80GroundTruth : public Float80GroundTruth {
 private:
  Float80 Func(Float80 x) override;
};

// Calculates the exponential using a Taylor series with vetted float80
// coefficients.
class Exp2VettedGroundTruth : public Float80GroundTruth {
 public:
  Exp2VettedGroundTruth()
      : Float80GroundTruth(std::vector<UlpBounds>{
            UlpBounds(/*min_input=*/0.0L, /*max_input=*/1.0L,
                      /*min_output=*/1.0L, /*max_output=*/2.0L),
            UlpBounds(/*min_input=*/1.0L, /*max_input=*/2.0L,
                      /*min_output=*/2.0L, /*max_output=*/4.0L)}) {}

 private:
  Float80 Func(Float80 x) override;
};

}  // namespace joy
}  // namespace graphs
}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_PROJECTS_GRAPHS_JOY_FLOAT80_GROUND_TRUTH_H_
