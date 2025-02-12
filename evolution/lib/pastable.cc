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

#include "evolution/lib/pastable.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "absl/log/check.h"

namespace brain {
namespace evolution {

using ::std::istringstream;
using ::std::string;
using ::std::stringstream;

constexpr std::array<uint8_t, 16> kToHex = {48, 49, 50, 51, 52, 53,  54,  55,
                                            56, 57, 97, 98, 99, 100, 101, 102};

constexpr std::array<uint8_t, 103> kToBinary = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 10,  11,  12,  13,  14,  15};

string BinaryToPastable(const string& binary) {
  const uint8_t* binary_start = reinterpret_cast<const uint8_t*>(binary.data());
  const uint8_t* binary_end = binary_start + binary.size();
  stringstream stream;
  for (const uint8_t* binary = binary_start; binary != binary_end; ++binary) {
    stream << static_cast<char>(kToHex[(*binary) & 15]);  // First 4 bits of 8.
    stream << static_cast<char>(kToHex[((*binary) & 240) >> 4]);  // Last 4.
  }
  string pastable = stream.str();
  CHECK(pastable.size() == 2 * binary.size());
  return pastable;
}

string PastableToBinary(const string& pastable) {
  istringstream stream(pastable);
  CHECK_EQ(pastable.size() % 2, 0);
  const size_t binary_size = pastable.size() / 2;
  string binary_string(binary_size, 'a');
  uint8_t* binary_start = reinterpret_cast<uint8_t*>(&binary_string[0]);
  uint8_t* binary_end = binary_start + binary_size;
  for (uint8_t* binary = binary_start; binary != binary_end; ++binary) {
    uint8_t first_bits = static_cast<uint8_t>(stream.get());
    uint8_t last_bits = static_cast<uint8_t>(stream.get());
    (*binary) = kToBinary[first_bits] + (kToBinary[last_bits] << 4);
  }
  return binary_string;
}

}  // namespace evolution
}  // namespace brain
