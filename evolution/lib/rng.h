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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_RNG_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_RNG_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/random/discrete_distribution.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

class RNGInterface {
 public:
  virtual ~RNGInterface() {}

  // Returns a uniform integer between low (incl) and high (excl).
  virtual Integer UniformInteger(Integer low, Integer high) = 0;

  // Returns a double drawn from a Gaussian distribution.
  virtual double GaussianDouble(double mean, double stdev) = 0;
};

// Thread-compatible, but not thread-safe.
class RNG : public RNGInterface {
 public:
  explicit RNG(RNGSeed rng_seed);

  RNG(const RNG& other) = delete;
  RNG& operator=(const RNG& other) = delete;

  RNG(RNG&& other);
  RNG& operator=(RNG&& other);

  // Resets the generator with a new random seed.
  void Reset(RNGSeed rng_seed);

  // Returns a new, independent, and deterministically seeded generator.
  RNG Spawn();

  // A seed for another RNG.
  RNGSeed UniformRNGSeed();

  // Returns a uniform bool.
  bool UniformBool();

  // See base class.
  Integer UniformInteger(Integer low, Integer high) override;

  // Returns a uniform double between low (incl) and high (excl).
  double UniformDouble(double low, double high);

  // Returns a uniform double between low (incl) and high (excl).
  float UniformFloat(float low, float high);

  // Returns a uniform double between 0.0 and 1.0.
  double UniformProbability();

  // Returns a double drawn from a Gaussian distribution.
  double GaussianDouble(double mean, double stdev) override;

  // Returns an integer generated from an absl::discrete_distribution.
  Integer DiscreteInteger(absl::discrete_distribution<Integer> dis);

  // Samples a vector that sums to one from the given Dirichlet distribution.
  // The concentrations determine the distribution size, and should be positive.
  std::vector<double> Dirichlet(absl::Span<const double> concentrations);

  // Returns a string with characters that are indepdently uniformly sampled
  // from the 64 characters 'a'-'z', 'A'-'Z', '0'-'9', '_' and '~'.
  std::string UniformString(size_t size);

  // Returns a string where the first character is uniformly sampled from
  //   'a'-'z', 'A'-'Z'
  // and the remaining characters are indepdently uniformly sampled from
  //   'a'-'z', 'A'-'Z', '0'-'9'.
  std::string UniformFileName(size_t size);

  template <class ElementT>
  void Shuffle(std::vector<ElementT>* elements);

  template <class ElementT>
  ElementT Choice(const std::vector<ElementT>& elements);

  // Returns a reference to the random source. Can be used as a URBG for
  // functions in the standard library that require one.
  absl::BitGen& BitGen();

 private:
  std::unique_ptr<absl::BitGen> bit_gen_;
};

template <class ElementT>
void RNG::Shuffle(std::vector<ElementT>* elements) {
  std::shuffle(elements->begin(), elements->end(), *bit_gen_);
}

template <class ElementT>
ElementT RNG::Choice(const std::vector<ElementT>& elements) {
  return elements[UniformInteger(0, elements.size())];
}

namespace rng_internal {
bool FillRNGSeedFromURandom(RNGSeed* rng_seed);
bool FillRNGSeedFromTime(RNGSeed* rng_seed);
}  // namespace rng_internal

// Generate a random seed using current time.
RNGSeed GenerateRNGSeed();

// Generates an RNG seed if rng_seed <= 0, else returns rng_seed.
RNGSeed UseOrGenerateRNGSeed(const Integer rng_seed);

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_RNG_H_
