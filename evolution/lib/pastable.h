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

// Utility to convert between byte strings and pastable strings.

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_PASTABLE_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_PASTABLE_H_

#include <string>

namespace brain {
namespace evolution {

// Converts a binary string to/from one that can be cut-and-pasted.
std::string BinaryToPastable(const std::string& binary);
std::string PastableToBinary(const std::string& pastable);

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_PASTABLE_H_
