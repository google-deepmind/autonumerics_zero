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

#include "evolution/projects/graphs/joy/float80_ground_truth.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind11_protobuf/native_proto_caster.h"

namespace brain {
namespace evolution {
namespace graphs {
namespace joy {

PYBIND11_MODULE(float80_ground_truth, m) {
  pybind11_protobuf::ImportNativeProtoCasters();
  pybind11::class_<Exp2Float80GroundTruth>(m, "Exp2Float80GroundTruth")
      .def(pybind11::init<>())
      .def("label", &Exp2Float80GroundTruth::Label)
      .def("signed_relative_error",
           &Exp2Float80GroundTruth::SignedRelativeError)
      .def("signed_relative_error_with_ulp_at",
           &Exp2Float80GroundTruth::SignedRelativeErrorWithUlpAt);

  pybind11::class_<ExpeFloat80GroundTruth>(m, "ExpeFloat80GroundTruth")
      .def(pybind11::init<>())
      .def("label", &ExpeFloat80GroundTruth::Label)
      .def("signed_relative_error",
           &ExpeFloat80GroundTruth::SignedRelativeError)
      .def("signed_relative_error_with_ulp_at",
           &ExpeFloat80GroundTruth::SignedRelativeErrorWithUlpAt);

  pybind11::class_<Log2Float80GroundTruth>(m, "Log2Float80GroundTruth")
      .def(pybind11::init<>())
      .def("label", &Log2Float80GroundTruth::Label)
      .def("signed_relative_error",
           &Log2Float80GroundTruth::SignedRelativeError)
      .def("signed_relative_error_with_ulp_at",
           &Log2Float80GroundTruth::SignedRelativeErrorWithUlpAt);

  pybind11::class_<LogeFloat80GroundTruth>(m, "LogeFloat80GroundTruth")
      .def(pybind11::init<>())
      .def("label", &LogeFloat80GroundTruth::Label)
      .def("signed_relative_error",
           &LogeFloat80GroundTruth::SignedRelativeError)
      .def("signed_relative_error_with_ulp_at",
           &LogeFloat80GroundTruth::SignedRelativeErrorWithUlpAt);

  pybind11::class_<ErfFloat80GroundTruth>(m, "ErfFloat80GroundTruth")
      .def(pybind11::init<>())
      .def("label", &ErfFloat80GroundTruth::Label)
      .def("signed_relative_error", &ErfFloat80GroundTruth::SignedRelativeError)
      .def("signed_relative_error_with_ulp_at",
           &ErfFloat80GroundTruth::SignedRelativeErrorWithUlpAt);

  pybind11::class_<ErfcFloat80GroundTruth>(m, "ErfcFloat80GroundTruth")
      .def(pybind11::init<>())
      .def("label", &ErfcFloat80GroundTruth::Label)
      .def("signed_relative_error",
           &ErfcFloat80GroundTruth::SignedRelativeError)
      .def("signed_relative_error_with_ulp_at",
           &ErfcFloat80GroundTruth::SignedRelativeErrorWithUlpAt);

  pybind11::class_<WavyFloat80GroundTruth>(m, "WavyFloat80GroundTruth")
      .def(pybind11::init<int>())
      .def("label", &WavyFloat80GroundTruth::Label)
      .def("signed_relative_error",
           &WavyFloat80GroundTruth::SignedRelativeError)
      .def("signed_relative_error_with_ulp_at",
           &WavyFloat80GroundTruth::SignedRelativeErrorWithUlpAt);

  pybind11::class_<AiryFloat80GroundTruth>(m, "AiryFloat80GroundTruth")
      .def(pybind11::init<int>())
      .def("label", &AiryFloat80GroundTruth::Label)
      .def("signed_relative_error",
           &AiryFloat80GroundTruth::SignedRelativeError)
      .def("signed_relative_error_with_ulp_at",
           &AiryFloat80GroundTruth::SignedRelativeErrorWithUlpAt);

  pybind11::class_<BesselFloat80GroundTruth>(m, "BesselFloat80GroundTruth")
      .def(pybind11::init<>())
      .def("label", &BesselFloat80GroundTruth::Label)
      .def("signed_relative_error",
           &BesselFloat80GroundTruth::SignedRelativeError)
      .def("signed_relative_error_with_ulp_at",
           &BesselFloat80GroundTruth::SignedRelativeErrorWithUlpAt);

  pybind11::class_<Exp2VettedGroundTruth>(m, "Exp2VettedGroundTruth")
      .def(pybind11::init<>())
      .def("label", &Exp2VettedGroundTruth::Label)
      .def("signed_relative_error", &Exp2VettedGroundTruth::SignedRelativeError)
      .def("signed_relative_error_with_ulp_at",
           &Exp2VettedGroundTruth::SignedRelativeErrorWithUlpAt);

  pybind11::class_<UlpBounds>(m, "UlpBounds")
      .def("get_ulp_reference_point", &UlpBounds::get_ulp_reference_point)
      .def_static("Create_UlpBounds", &UlpBounds::UlpBoundsFromDoubles);

  m.def("get_ulp_reference_point_for_input",
        &get_ulp_reference_point_for_input);
}

}  // namespace joy
}  // namespace graphs
}  // namespace evolution
}  // namespace brain
