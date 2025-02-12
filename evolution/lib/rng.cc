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

#include "evolution/lib/rng.h"

#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/discrete_distribution.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "evolution/lib/hashing.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

using ::absl::GetCurrentTimeNanos;
using ::absl::Nanoseconds;
using ::absl::SleepFor;
using ::std::ifstream;
using ::std::make_unique;
using ::std::numeric_limits;
using ::std::string;

RNG::RNG(const RNGSeed rng_seed)
    : bit_gen_(make_unique<absl::BitGen>(std::seed_seq({rng_seed}))) {}

RNG::RNG(RNG&& other) : bit_gen_(std::move(other.bit_gen_)) {}

RNG& RNG::operator=(RNG&& other) {
  bit_gen_ = std::move(other.bit_gen_);
  return *this;
}

void RNG::Reset(const RNGSeed rng_seed) {
  delete bit_gen_.release();
  bit_gen_ = std::make_unique<absl::BitGen>(std::seed_seq({rng_seed}));
}

RNG RNG::Spawn() { return RNG(UniformRNGSeed()); }

RNGSeed RNG::UniformRNGSeed() {
  return absl::Uniform<RNGSeed>(absl::IntervalOpen, *bit_gen_,
                                std::numeric_limits<RNGSeed>::min(),
                                std::numeric_limits<RNGSeed>::max());
}

bool RNG::UniformBool() { return ::absl::Bernoulli(*bit_gen_, 0.5); }

Integer RNG::UniformInteger(const Integer low, const Integer high) {
  return ::absl::Uniform<Integer>(*bit_gen_, low, high);
}

double RNG::UniformDouble(const double low, const double high) {
  return ::absl::Uniform<double>(*bit_gen_, low, high);
}

float RNG::UniformFloat(const float low, const float high) {
  return ::absl::Uniform<float>(*bit_gen_, low, high);
}

double RNG::UniformProbability() {
  return ::absl::Uniform<double>(*bit_gen_, 0.0, 1.0);
}

double RNG::GaussianDouble(const double mean, const double stdev) {
  return ::absl::Gaussian<double>(*bit_gen_, mean, stdev);
}

Integer RNG::DiscreteInteger(absl::discrete_distribution<Integer> dis) {
  return dis(*bit_gen_);
}

std::vector<double> RNG::Dirichlet(absl::Span<const double> concentrations) {
  CHECK_GT(concentrations.size(), 0);
  double remaining_concentration_sum = 0.0;
  for (const double concentration : concentrations) {
    CHECK_GT(concentration, 0);
    remaining_concentration_sum += concentration;
  }
  std::vector<double> sample;
  double remaining_weight = 1.0;
  for (int i = 0; i < concentrations.size() - 1; ++i) {
    remaining_concentration_sum -= concentrations[i];
    double marginal = ::absl::Beta<double>(*bit_gen_, concentrations[i],
                                           remaining_concentration_sum);
    double value = remaining_weight * marginal;
    sample.push_back(value);
    remaining_weight -= value;
  }
  sample.push_back(remaining_weight);
  return sample;
}

string RNG::UniformString(const size_t size) {
  string random_string;
  for (size_t i = 0; i < size; ++i) {
    char random_char;
    const Integer char_index = UniformInteger(0, 64);
    if (char_index < 26) {
      random_char = char_index + 97;  // Maps 0-25 to 'a'-'z'.
    } else if (char_index < 52) {
      random_char = char_index - 26 + 65;  // Maps 26-51 to 'A'-'Z'.
    } else if (char_index < 62) {
      random_char = char_index - 52 + 48;  // Maps 52-61 to '0'-'9'.
    } else if (char_index == 62) {
      random_char = '_';
    } else if (char_index == 63) {
      random_char = '~';
    } else {
      LOG(FATAL) << "Code should not get here." << std::endl;
    }
    random_string.push_back(random_char);
  }
  return random_string;
}

string RNG::UniformFileName(const size_t size) {
  CHECK_GT(size, 0);
  string random_string;
  char random_char;

  // First char.
  Integer char_index = UniformInteger(0, 52);
  if (char_index < 26) {
    random_char = char_index + 97;  // Maps 0-25 to 'a'-'z'.
  } else if (char_index < 52) {
    random_char = char_index - 26 + 65;  // Maps 26-51 to 'A'-'Z'.
  } else {
    LOG(FATAL) << "Code should not get here." << std::endl;
  }
  random_string.push_back(random_char);

  // Remaining chars.
  for (size_t i = 1; i < size; ++i) {
    char_index = UniformInteger(0, 62);
    if (char_index < 26) {
      random_char = char_index + 97;  // Maps 0-25 to 'a'-'z'.
    } else if (char_index < 52) {
      random_char = char_index - 26 + 65;  // Maps 26-51 to 'A'-'Z'.
    } else if (char_index < 62) {
      random_char = char_index - 52 + 48;  // Maps 52-61 to '0'-'9'.
    } else {
      LOG(FATAL) << "Code should not get here." << std::endl;
    }
    random_string.push_back(random_char);
  }
  return random_string;
}

absl::BitGen& RNG::BitGen() { return *bit_gen_; }

namespace rng_internal {

bool FillRNGSeedFromURandom(RNGSeed* rng_seed) {
  ifstream stream("/dev/urandom", std::ios::in | std::ios::binary);
  if (stream) {
    stream.read(reinterpret_cast<char*>(rng_seed), sizeof(*rng_seed));
    if (stream) {
      return true;
    }
  }
  return false;
}

bool FillRNGSeedFromTime(RNGSeed* rng_seed) {
  RNGSeed rng_seed_baseline = static_cast<RNGSeed>(
      GetCurrentTimeNanos() % numeric_limits<RNGSeed>::max());
  SleepFor(Nanoseconds(2));
  RNGSeed rng_seed_new = static_cast<RNGSeed>(GetCurrentTimeNanos() %
                                              numeric_limits<RNGSeed>::max());
  if (rng_seed_new != rng_seed_baseline) {
    *rng_seed = rng_seed_new;
    return true;
  }
  return false;
}

}  // namespace rng_internal

RNGSeed GenerateRNGSeed() {
  RNGSeed time_rng_seed;
  rng_internal::FillRNGSeedFromTime(&time_rng_seed);
  LOG(INFO) << "FillRNGSeedFromTime with value: " << time_rng_seed;

  RNGSeed urandom_rng_seed;
  if (rng_internal::FillRNGSeedFromURandom(&urandom_rng_seed)) {
    LOG(INFO) << "FillRNGSeedFromURandom with value: " << urandom_rng_seed;
    // If there is urandom, return a mix of it with the time seed.
    return Mix(urandom_rng_seed, time_rng_seed);
  } else {
    // If there is no urandom, return just the time seed.
    return time_rng_seed;
  }
}

RNGSeed UseOrGenerateRNGSeed(const Integer rng_seed) {
  if (rng_seed <= 0) {
    return GenerateRNGSeed();
  } else {
    return static_cast<RNGSeed>(rng_seed);
  }
}

}  // namespace evolution
}  // namespace brain
