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

#include <pybind11/pybind11.h>

#include <string>

PYBIND11_MODULE(pastable, m) {
  m.def("BinaryToPastable", [](std::string binary) {
    ::brain::evolution::BinaryToPastable(binary);
  });
  m.def("PastableToBinary", [](std::string pastable) {
    ::brain::evolution::PastableToBinary(pastable);
  });
}
