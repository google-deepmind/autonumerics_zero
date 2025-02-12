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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace brain {
namespace evolution {
PYBIND11_MODULE(deconstructed, m) {
  pybind11::class_<DeconstructedBFloat16>(m, "DeconstructedBFloat16")
      .def(pybind11::init(
               [](float value) { return DeconstructedBFloat16(value); }),
           pybind11::arg("value"));
  m.def("RoughBfloat16Ulp",
        [](float x) { return ::brain::evolution::RoughBfloat16Ulp(x); });
}
}  // namespace evolution
}  // namespace brain
