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

#include "evolution/lib/population_client_manager.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "evolution/lib/individual.pb.h"
#include "evolution/lib/population_client_manager_spec.pb.h"
#include "evolution/lib/search_algorithm_stats.pb.h"
#include "evolution/lib/types.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace brain {
namespace evolution {
PYBIND11_MODULE(population_client_manager, m) {
  pybind11_protobuf::ImportNativeProtoCasters();
  pybind11::class_<PopulationClientManager>(m, "PopulationClientManager")
      .def(pybind11::init([](const PopulationClientManagerSpec& spec,
                             Integer worker_id, RNGSeed rng_seed) {
             return std::make_unique<PopulationClientManager>(spec, worker_id,
                                                              rng_seed);
           }),
           pybind11::arg("spec"), pybind11::arg("worker_id"),
           pybind11::arg("rng_seed"))
      .def("PingClients", &PopulationClientManager::PingClients)
      .def(
          "Communicate",
          static_cast<std::pair<std::vector<Individual>, SearchAlgorithmStats> (
              PopulationClientManager::*)(const std::vector<IndividualData>&,
                                          Integer)>(
              &PopulationClientManager::Communicate))
      .def("NumClients", &PopulationClientManager::NumClients);
}
}  // namespace evolution
}  // namespace brain
