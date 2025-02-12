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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_POPULATION_CLIENT_MANAGER_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_POPULATION_CLIENT_MANAGER_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/random/discrete_distribution.h"
#include "absl/time/time.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/population_client.h"
#include "evolution/lib/population_client_manager_spec.pb.h"
#include "evolution/lib/rng.h"
#include "evolution/lib/search_algorithm_stats.pb.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

struct RandomClient {
  PopulationClientInterface* client;
  bool is_remote;
  Integer client_num;
};

// Base class to allow for mocking.
class PopulationClientManagerInterface {
 public:
  virtual ~PopulationClientManagerInterface() {}

  // Communicates with a random client. If the manager contains remote clients,
  // the `stats` return parameter will be an estimate of the overall experiment
  // progress; otherwise, it will be the exact local server's stats.
  virtual std::vector<Individual> Communicate(
      const std::vector<IndividualData>& uploads, const Integer num_to_download,
      SearchAlgorithmStats* stats) = 0;

  // Alternative overload for PyClif.
  virtual std::pair<std::vector<Individual>, SearchAlgorithmStats> Communicate(
      const std::vector<IndividualData>& uploads,
      const Integer num_to_download) = 0;

  // Pings all clients, returning the number of successful pings.
  virtual Integer PingClients() = 0;

  // Returns a pointer to a random client according to the distribution, along
  // with a boolean indicating if the client is remote or not.
  virtual RandomClient GetRandomClient() = 0;

  // Returns a reference to the vector containing all clients. Useful for
  // performing operations explicitly against all clients.
  virtual const std::vector<std::unique_ptr<PopulationClientInterface>>&
  GetAllClients() = 0;
  virtual std::vector<PopulationClientInterface*> GetAllRemoteClients() = 0;
  virtual std::vector<PopulationClientInterface*> GetAllLocalClients() = 0;

  // Returns the number of clients.
  virtual Integer NumClients() = 0;

  // Returns the local progress that this manager has accumulated.
  virtual const std::vector<double> GetProgressDelta() = 0;
};

// PopulationClientManager constructs a collection of PopulationClients,
// building local servers or communicating to remote ones, as required.
// It also handles the logic of picking one to communicate with based on a
// discrete distribution defined by the specified weights. Thread-compatible.
class PopulationClientManager : public PopulationClientManagerInterface {
  // Type definition for a functor that constructs and returns a mock client.
  typedef std::function<std::unique_ptr<PopulationClientInterface>()>
      CreateMockClientFn;

 public:
  PopulationClientManager(
      const PopulationClientManagerSpec& spec, Integer worker_id,
      RNGSeed rng_seed,
      // Pass nullptr unless in tests. We don't use a default arg because it
      // confuses PyClif.
      CreateMockClientFn create_mock_local_client,
      CreateMockClientFn create_mock_remote_client = nullptr);

  // Overload without optional args (helpful for PyClif, to avoid cliffing
  // CreateMockClientFn, etc.). This is also the form used in C++ outside of
  // tests, as a side effect. It just delegates to the constructor above.
  PopulationClientManager(const PopulationClientManagerSpec& spec,
                          Integer worker_id, RNGSeed rng_seed);

  ~PopulationClientManager() override;

  PopulationClientManager(const PopulationClientManager& other) = delete;
  PopulationClientManager& operator=(const PopulationClientManager& other) =
      delete;

  // See base class.
  std::vector<Individual> Communicate(
      const std::vector<IndividualData>& uploads, const Integer num_to_download,
      SearchAlgorithmStats* stats) override;

  // See base class.
  std::pair<std::vector<Individual>, SearchAlgorithmStats> Communicate(
      const std::vector<IndividualData>& uploads,
      const Integer num_to_download) override;

  // See base class.
  Integer PingClients() override;

  // See base class.
  RandomClient GetRandomClient() override;

  // See base class.
  const std::vector<std::unique_ptr<PopulationClientInterface>>& GetAllClients()
      override;
  std::vector<PopulationClientInterface*> GetAllRemoteClients() override;
  std::vector<PopulationClientInterface*> GetAllLocalClients() override;

  // See base class.
  Integer NumClients() override;

  // See base class.
  const std::vector<double> GetProgressDelta() override;

 private:
  // Returns the manager's best approximation of the overall experiment stats.
  // This will be an underestimate, where the margin of error is determined by
  // how recently each remote server was contacted.
  SearchAlgorithmStats GetExperimentStats();

  // Returns a client that performs actions as function calls rather than RPCs.
  std::unique_ptr<PopulationClientInterface> CreateRpclessClient(
      const PopulationClientManagerElement& element);

  // Keyed by server ID.
  std::unordered_map<std::string, std::string> remote_addresses_;

  // A vector containing pointers to all PopulationClients. The first
  // `num_remote_clients_` of them are remote; the remainder are local.
  std::vector<std::unique_ptr<PopulationClientInterface>> clients_;

  // The weights of each of the clients in `clients_` and the corresponding
  // normalized distribution.
  std::vector<double> weights_;
  absl::discrete_distribution<Integer> distribution_;

  // The minimum period between requests through each client. If empty, the
  // client will be contacted as often as needed.
  std::vector<absl::Duration> target_rpc_periods_;

  // The time of the last RPC through each client. Must have the same length as
  // `target_rpc_periods_`.
  std::vector<absl::Time> latest_rpc_times_;

  // The running count of local server progress since the last time the manager
  // communicated with a remote server.
  std::vector<double> progress_delta_;

  // The last-reported stats objects for each remote server in `clients_`, where
  // the i-th entry in this vector corresponds to the i-th server in `clients_`.
  std::vector<SearchAlgorithmStats> remote_stats_;

  // The number of remote clients. If only a single server is passed in at
  // initialization, this value will be 1 regardless of whether that server is
  // local or remote, so that GetExperimentStats() operates the same in a local
  // experiment as it does in a remote one.
  Integer num_remote_clients_;

  // The ID of the worker this manager is running on, if applicable.
  Integer worker_id_;

  RNG rng_;
};

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_POPULATION_CLIENT_MANAGER_H_
