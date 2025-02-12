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

// Classes to print during debugging.

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_PRINTING_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_PRINTING_H_

#include <sched.h>

#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <vector>


namespace brain {
namespace evolution {

// Print and Flush can be used to print to stdout for debugging purposes.
// Usage:
// Print() << "my_variable = " << my_variable << stuff << "etc." << Flush();

class Flush {};

class FlushAndContinue {};

class Print {
 public:
  Print() {}

  template <typename PrintedT>
  Print& operator<<(const PrintedT& component) {
    stream_ << component;
    return *this;
  }

  template <>
  Print& operator<< <Flush>(const Flush& component) {
    std::cout << stream_.str() << std::endl;
    return *this;
  }

  template <>
  Print& operator<< <FlushAndContinue>(const FlushAndContinue& component) {
    std::cout << stream_.str();
    return *this;
  }

  template <>
  Print& operator<< <std::vector<double>>(
      const std::vector<double>& component) {
    stream_ << "{";
    for (size_t i = 0; i < component.size(); ++i) {
      stream_ << std::setprecision(20) << component[i];
      if (i != component.size() - 1) {
        stream_ << ", ";
      }
    }
    stream_ << "}";
    return *this;
  }

 private:
  std::ostringstream stream_;
};

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_PRINTING_H_
