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

#include "evolution/lib/fitnesses_reduction.h"

#include <math.h>

#include <functional>
#include <limits>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "evolution/lib/fitnesses_reduction_spec.pb.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

using ::std::function;
using ::std::vector;

FitnessesReductionFn BuildFitnessesReductionFn(
    const FitnessesReductionSpec& spec) {
  if (spec.has_fitness0_only()) {
    return [](const vector<double>& fitnesses) {
      CHECK_GT(fitnesses.size(), 0);
      return vector<double>{fitnesses[0]};
    };
  } else if (spec.has_fitness0_then_fitness1()) {
    const double target = spec.fitness0_then_fitness1().fitness0_target();
    const bool negate_and_invert =
        spec.fitness0_then_fitness1().negate_and_invert_fitness1();
    return [target, negate_and_invert](const vector<double>& fitnesses) {
      CHECK_EQ(fitnesses.size(), 2);
      const double fitness0 = fitnesses[0];
      if (fitness0 < target) {
        return vector<double>{fitness0};
      } else {
        double fitness1 = fitnesses[1];
        if (negate_and_invert) {
          fitness1 = -1.0 / fitness1;
        }
        return vector<double>{target + fitness1};
      }
    };
  } else if (spec.has_subsequence_fitnesses_reduction()) {
    const SubsequenceFitnessesReduction& method_spec =
        spec.subsequence_fitnesses_reduction();
    const vector<Integer> indexes(method_spec.indexes().begin(),
                                  method_spec.indexes().end());
    return [indexes](const vector<double>& fitnesses) {
      vector<double> reduced_fitnesses;
      reduced_fitnesses.reserve(indexes.size());
      for (const Integer index : indexes) {
        reduced_fitnesses.push_back(fitnesses.at(index));
      }
      return reduced_fitnesses;
    };
  } else if (spec.has_one_dim_circuit_fitnesses_reduction()) {
    return [](const vector<double>& fitnesses) {
      CHECK_EQ(fitnesses.size(), 3);
      const double correctness = fitnesses[0];
      if (correctness < 1.0) {
        return vector<double>{correctness};
      } else {
        const double num_gates = -fitnesses[1];
        const double cpl = -fitnesses[2];
        return vector<double>{1.0 + 1.0 / (sqrt(num_gates) + cpl)};
      }
    };
  } else if (spec.has_diverse_correctness_circuit_fitnesses_reduction()) {
    const DiverseCorrectnessCircuitFitnessesReduction& method_spec =
        spec.diverse_correctness_circuit_fitnesses_reduction();
    const double min_theta = method_spec.has_min_theta()
                                 ? method_spec.min_theta()
                                 : -std::numeric_limits<double>::infinity();
    const double max_theta = method_spec.has_max_theta()
                                 ? method_spec.max_theta()
                                 : std::numeric_limits<double>::infinity();
    return [min_theta, max_theta](const vector<double>& fitnesses) {
      CHECK_EQ(fitnesses.size(), 3);
      const double correctness = fitnesses[0];
      const double num_gates = -fitnesses[1];
      const double cpl = -fitnesses[2];
      if (num_gates == 0.0 || cpl == 0.0) {
        // Special case to handle pathological circuits (no gates with
        // positive cost or no path from input to output).
        // Return two zeros to indicate a "worst-possible" circuit.
        return vector<double>{0.0, 0.0};
      }
      double theta = atan(cpl / sqrt(num_gates));
      if (theta < min_theta) {
        theta = min_theta;
      }
      if (theta > max_theta) {
        theta = max_theta;
      }
      return vector<double>{correctness * cos(theta), correctness * sin(theta)};
    };
  } else if (spec.has_shape_circuit_fitnesses_reduction()) {
    const ShapeCircuitFitnessesReduction& method_spec =
        spec.shape_circuit_fitnesses_reduction();
    CHECK(method_spec.has_max_num_gates() && method_spec.has_max_cpl());
    const double max_num_gates = method_spec.max_num_gates();
    const double max_cpl = method_spec.max_cpl();
    const double min_num_gates =
        method_spec.has_min_num_gates() ? method_spec.min_num_gates() : 0.0;
    const double min_cpl =
        method_spec.has_min_cpl() ? method_spec.min_cpl() : 0.0;
    return [max_num_gates, max_cpl, min_num_gates,
            min_cpl](const vector<double>& fitnesses) {
      CHECK_EQ(fitnesses.size(), 3);
      const double correctness = fitnesses[0];
      if (correctness < 1.0) {  // Incorrect circuit.
        return vector<double>{-max_num_gates, -max_cpl};
      } else {  // Correct circuit.
        double num_gates = -fitnesses[1];
        double cpl = -fitnesses[2];
        if (num_gates < min_num_gates) num_gates = min_num_gates;
        if (num_gates > max_num_gates) num_gates = max_num_gates;
        if (cpl < min_cpl) cpl = min_cpl;
        if (cpl > max_cpl) cpl = max_cpl;
        return vector<double>{-sqrt(num_gates), -cpl};
      }
    };
  } else {
    LOG(FATAL) << "Unsupported fitnesses reduction: " << spec;
  }
}

}  // namespace evolution
}  // namespace brain
