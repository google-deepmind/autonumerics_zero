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

#include "evolution/lib/population_client.h"

#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "evolution/lib/current_time.h"
#include "evolution/lib/id_generator.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/individuals_writer.h"
#include "evolution/lib/mocking.h"
#include "evolution/lib/population_client_spec.pb.h"
#include "evolution/lib/population_server.h"
#include "evolution/lib/population_server.pb.h"
#include "evolution/lib/population_server_spec.pb.h"
#include "evolution/lib/search_algorithm_stats.pb.h"
#include "evolution/lib/server.pb.h"
#include "evolution/lib/types.h"

ABSL_FLAG(bool, log_population_client_rpc_retries, true,
          "Whether to print timing information.");
ABSL_FLAG(bool, log_population_client_rpc_timing, true,
          "Whether to print timing information.");

namespace brain {
namespace evolution {


using ::std::make_unique;
using ::std::pair;
using ::std::string;
using ::std::unique_ptr;
using ::std::vector;

constexpr Integer kIndividualIDLength = 32;  // Must match population.sdl.

PopulationClientInterface::~PopulationClientInterface() = default;

std::vector<Individual> PopulationClientInterface::Communicate(
    const std::vector<IndividualData>& uploads, const Integer num_to_download,
    SearchAlgorithmStats* stats) {
  return CommunicateImpl(uploads, num_to_download, stats);
}

pair<vector<Individual>, SearchAlgorithmStats>
PopulationClientInterface::Communicate(const vector<IndividualData>& uploads,
                                       const Integer num_to_download) {
  SearchAlgorithmStats stats;
  vector<Individual> downloads =
      CommunicateImpl(uploads, num_to_download, &stats);
  return std::make_pair(std::move(downloads), std::move(stats));
}

bool PopulationClientInterface::Ping() { return PingImpl(); }

std::unique_ptr<IndividualsWriter> MakeIndividualsWriter(
    const PopulationClientSpec& spec, Integer worker_id,
    IndividualsWriter* mock);

RpclessPopulationClient::RpclessPopulationClient(
    const PopulationClientSpec& client_spec,
    const PopulationServerSpec& server_spec, RNGSeed server_rng_seed,
    RNGSeed client_rng_seed, Integer worker_id,
    PopulationServer* mock_population_server,
    IDGeneratorInterface* mock_individual_id_generator,
    CurrentTimeInterface* mock_current_time)
    : client_spec_(client_spec),
      rng_(client_rng_seed),
      owned_server_(MakeOwnedIfNoMock<PopulationServer>(
          mock_population_server, server_spec, server_rng_seed,
          /*mock_algo*/ nullptr, worker_id)),
      server_(
          GetPointerToMockOrOwned(mock_population_server, owned_server_.get())),
      worker_id_(worker_id),
      owned_individual_id_generator_(MakeOwnedIfNoMock<IDGenerator>(
          mock_individual_id_generator, &rng_, kIndividualIDLength,
          /*integers_to_mix=*/
          vector<Integer>{worker_id, client_spec.worker_collection_id()},
          /*strings_to_mix=*/
          vector<string>{string(client_spec.worker_collection_name())})),
      individual_id_generator_(GetPointerToMockOrOwned(
          mock_individual_id_generator, owned_individual_id_generator_.get())),
      owned_current_time_(MakeOwnedIfNoMock<CurrentTime>(mock_current_time)),
      current_time_(GetPointerToMockOrOwned(mock_current_time,
                                            owned_current_time_.get())) {}

bool RpclessPopulationClient::PingImpl() { return true; }

vector<Individual> RpclessPopulationClient::CommunicateImpl(
    const vector<IndividualData>& uploads, const Integer num_to_download,
    SearchAlgorithmStats* stats) {
  unique_ptr<CommunicateResponse> response;
  CommunicateRequest request;

  request.set_num_to_download(num_to_download);
  if (stats != nullptr) {
    request.set_get_stats(true);
  }
  response = make_unique<CommunicateResponse>();
  for (const IndividualData& data : uploads) {
    *request.add_uploads() =
        MakeIndividual(data,
                       individual_id_generator_->Generate(),  // individual_id
                       current_time_->Nanos(), worker_id_, client_spec_,
                       client_spec_.tmp_dir());
  }
  server_->Communicate(&request, response.get());
  vector<Individual> downloads;
  for (const Individual& individual : response->downloads()) {
    downloads.push_back(individual);
  }
  if (stats != nullptr) {
    *stats = response->stats();
  }
  return downloads;
}

Individual MakeIndividual(const IndividualData& data,
                          absl::string_view individual_id,
                          const Integer time_nanos, const Integer worker_id,
                          const PopulationClientSpec& client_spec,
                          const string& tmp_dir) {
  Individual individual;
  *individual.mutable_data() = data;
  individual.set_individual_id(individual_id);
  individual.set_time_nanos(time_nanos);
  individual.set_worker_id(worker_id);
  if (client_spec.has_worker_collection_id()) {
    individual.set_worker_collection_id(client_spec.worker_collection_id());
  }
  if (client_spec.has_worker_collection_name()) {
    individual.set_worker_collection_name(client_spec.worker_collection_name());
  }
  return individual;
}

std::unique_ptr<IndividualsWriter> MakeIndividualsWriter(
    const PopulationClientSpec& spec, const Integer worker_id,
    IndividualsWriter* mock) {
  return std::unique_ptr<IndividualsWriter>();
}

}  // namespace evolution
}  // namespace brain
