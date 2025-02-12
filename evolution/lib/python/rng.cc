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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace brain {
namespace evolution {
PYBIND11_MODULE(rng, m) {
  m.def("GenerateRNGSeed", &GenerateRNGSeed);
  pybind11::class_<RNG>(m, "RNG")
      .def(
          pybind11::init([](const uint32_t rng_seed) { return RNG(rng_seed); }))
      .def("Reset", &RNG::Reset)
      .def("Spawn", &RNG::Spawn)
      .def("UniformRNGSeed", &RNG::UniformRNGSeed)
      .def("UniformBool", &RNG::UniformBool)
      .def("UniformInteger", &RNG::UniformInteger)
      .def("UniformDouble", &RNG::UniformDouble)
      .def("UniformProbability", &RNG::UniformProbability)
      .def("GaussianDouble", &RNG::GaussianDouble)
      .def("UniformString", &RNG::UniformString)
      .def("UniformFileName", &RNG::UniformFileName);
  // m.def()
}
}  // namespace evolution
}  // namespace brain
