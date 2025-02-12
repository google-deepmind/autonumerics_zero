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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_POPULATION_CLIENT_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_POPULATION_CLIENT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "evolution/lib/current_time.h"
#include "evolution/lib/id_generator.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/individuals_writer.h"
#include "evolution/lib/population_client_spec.pb.h"
#include "evolution/lib/population_server.h"
#include "evolution/lib/population_server_spec.pb.h"
#include "evolution/lib/rng.h"
#include "evolution/lib/search_algorithm_stats.pb.h"
#include "evolution/lib/types.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace brain {
namespace evolution {

// Base class to allow for mocking.
class PopulationClientInterface {
 public:
  virtual ~PopulationClientInterface();

  // Communicates with the server. Returns individuals downloaded.
  std::vector<Individual> Communicate(
      // Individuals to upload to the server.
      const std::vector<IndividualData>& uploads,
      // Number of individuals to download from the server.
      Integer num_to_download,
      // If not nullptr, will be filled with stats returned by the server.
      SearchAlgorithmStats* stats);

  // Alternative overload for PyClif. See above for arguments and return values.
  std::pair<std::vector<Individual>, SearchAlgorithmStats> Communicate(
      const std::vector<IndividualData>& uploads, Integer num_to_download);

  // Pings the server, returning whether the server responded.
  bool Ping();

 private:
  // See `Communicate` for arguments and return values. The `Communicate` forms
  // delegate both to this method (instead of to one of them) to avoid confusing
  // PyClif for the derived classes.
  virtual std::vector<Individual> CommunicateImpl(
      const std::vector<IndividualData>& uploads, Integer num_to_download,
      SearchAlgorithmStats* stats) = 0;

  // Pings the server, returning whether the server responded.
  virtual bool PingImpl() = 0;
};

// Client to communicate with a rpcless population.
class RpclessPopulationClient : public PopulationClientInterface {
 public:
  explicit RpclessPopulationClient(
      const PopulationClientSpec& client_spec,
      const PopulationServerSpec& server_spec, RNGSeed server_rng_seed,
      RNGSeed client_rng_seed, Integer worker_id,
      PopulationServer* mock_population_server = nullptr,
      IDGeneratorInterface* mock_individual_id_generator = nullptr,
      CurrentTimeInterface* mock_current_time = nullptr);

  RpclessPopulationClient(const RpclessPopulationClient& other) = delete;
  RpclessPopulationClient& operator=(const RpclessPopulationClient& other) =
      delete;

 private:
  // See base class.
  std::vector<Individual> CommunicateImpl(
      const std::vector<IndividualData>& uploads, Integer num_to_download,
      SearchAlgorithmStats* stats) override;

  // See base class.
  bool PingImpl() override;

  void WriteToSpanner(const ::google::protobuf::RepeatedPtrField<Individual>& individuals);

  const PopulationClientSpec client_spec_;
  RNG rng_;
  // Owned server that we directly interact with using method calls.
  const std::unique_ptr<PopulationServer> owned_server_;
  PopulationServer* const server_;
  const Integer worker_id_;
  const std::unique_ptr<IDGeneratorInterface> owned_individual_id_generator_;
  IDGeneratorInterface* const individual_id_generator_;
  const std::unique_ptr<CurrentTimeInterface> owned_current_time_;
  CurrentTimeInterface* const current_time_;
  const std::unique_ptr<IndividualsWriter> owned_individuals_writer_;
};

Individual MakeIndividual(const IndividualData& data,
                          absl::string_view individual_id, Integer time_nanos,
                          Integer worker_id,
                          const PopulationClientSpec& client_spec,
                          const std::string& tmp_dir);
void CheckAssociatedFile(const std::string& filepath,
                         const std::string& tmp_dir);

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_POPULATION_CLIENT_H_
