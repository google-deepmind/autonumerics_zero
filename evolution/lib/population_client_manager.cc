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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/discrete_distribution.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/population_client.h"
#include "evolution/lib/population_client_manager_spec.pb.h"
#include "evolution/lib/printing.h"
#include "evolution/lib/rng.h"
#include "evolution/lib/search_algorithm_stats.pb.h"
#include "evolution/lib/search_algorithm_stats_util.h"
#include "evolution/lib/types.h"

ABSL_FLAG(bool, log_which_client, false,
          "Whether to print which client was chosen for communcation.");

namespace brain {
namespace evolution {

using ::absl::Duration;
using ::absl::GetFlag;
using ::absl::Now;
using ::absl::Seconds;
using ::absl::StrCat;
using ::absl::Time;
using ::absl::ZeroDuration;
using ::brain::evolution::PopulationClientManagerElement;
using ::std::make_unique;
using ::std::move;
using ::std::pair;
using ::std::string;
using ::std::unique_ptr;
using ::std::unordered_map;
using ::std::vector;

constexpr Duration kMaxClientChooseTime = Seconds(1);

void SetUpRateThrottling(const vector<double>& target_rpc_period_secs_list,
                         RNG& rng, vector<Duration>& target_rpc_periods,
                         vector<Time>& latest_rpc_times) {
  target_rpc_periods.clear();
  latest_rpc_times.clear();
  bool has_throttled_client = false;
  bool has_non_throttled_client = false;
  const Time now = Now();
  for (const double target_rpc_period_secs : target_rpc_period_secs_list) {
    const Duration target_rpc_period = Seconds(target_rpc_period_secs);
    Time latest_rpc_time = now;
    if (target_rpc_period > ZeroDuration()) {
      has_throttled_client = true;
      // So the initial RPC is spread out like the rest.
      latest_rpc_time -=
          Seconds(rng.UniformDouble(0.0, target_rpc_period_secs));
    } else {
      has_non_throttled_client = true;
    }
    target_rpc_periods.push_back(target_rpc_period);
    latest_rpc_times.push_back(latest_rpc_time);
  }
  if (has_throttled_client) {  // Throttling is enabled.
    if (!has_non_throttled_client) {
      LOG(FATAL) << "At least one client must not be throttled.";
    }
  } else {  // Throttling is disabled.
    target_rpc_periods.clear();
    latest_rpc_times.clear();
  }
}

PopulationClientManager::PopulationClientManager(
    const PopulationClientManagerSpec& spec, Integer worker_id,
    RNGSeed rng_seed, CreateMockClientFn create_mock_local_client,
    CreateMockClientFn create_mock_remote_client)
    : num_remote_clients_(0), worker_id_(worker_id), rng_(rng_seed) {
  Print() << "pop client manager seed: " << rng_seed << Flush();
  vector<double> target_rpc_period_secs_list;
  for (const auto& element : spec.elements()) {
    unique_ptr<PopulationClientInterface> client;
    if (create_mock_local_client != nullptr) {
      client = create_mock_local_client();
    } else {
      client = CreateRpclessClient(element);
    }
    // Local clients get pushed to the back.
    clients_.push_back(move(client));
    weights_.push_back(element.weight());
    target_rpc_period_secs_list.push_back(element.target_rpc_period_secs());
  }

  SetUpRateThrottling(target_rpc_period_secs_list, rng_, target_rpc_periods_,
                      latest_rpc_times_);

  // Initializing a manager with only a single server indicates the experiment
  // is being tested locally. Treat the single local server as a remote server
  // to retain the benefit of experiment stats being aggregated correctly.
  if (spec.elements_size() == 1) {
    num_remote_clients_ = 1;
  }

  // Initialize remote stats vector with the correct size.
  remote_stats_ = vector<SearchAlgorithmStats>(num_remote_clients_);

  distribution_ =
      absl::discrete_distribution<Integer>(weights_.cbegin(), weights_.cend());
}

PopulationClientManager::PopulationClientManager(
    const PopulationClientManagerSpec& spec, Integer worker_id,
    RNGSeed rng_seed)
    : PopulationClientManager(spec, worker_id, rng_seed, nullptr) {}

vector<Individual> PopulationClientManager::Communicate(
    const vector<IndividualData>& uploads, const Integer num_to_download,
    SearchAlgorithmStats* stats) {
  RandomClient random_client = GetRandomClient();

  // If no uploads, just communicate normally and exit.
  if (uploads.empty()) {
    return random_client.client->Communicate(uploads, num_to_download, stats);
  }

  // Otherwise, accumulate the progress of all individuals.
  for (auto& individual : uploads) {
    if (progress_delta_.empty()) {
      progress_delta_.assign(individual.individual_progress().begin(),
                             individual.individual_progress().end());
    } else {
      CHECK_EQ(individual.individual_progress_size(), progress_delta_.size());
      for (Integer i = 0; i < progress_delta_.size(); ++i) {
        progress_delta_[i] += individual.individual_progress(i);
      }
    }
  }

  vector<IndividualData> puts(uploads);

  // If communicating with a remote server, communicate and send progress, then
  // zero out progress delta.
  if (random_client.is_remote) {
    // If communicating with a remote server, set `worker_progress` on the first
    // individual so the server can accumulate the progress.
    for (auto& progress : progress_delta_) {
      puts[0].add_worker_progress(progress);
    }

    // Reset deltas before communicating.
    for (auto& delta : progress_delta_) {
      delta = 0.0;
    }
  }

  // Even if the user passes in stats = nullptr, we still want the server to
  // respond with its stats, so possibly substitute in our own object.
  SearchAlgorithmStats response_stats;

  auto result =
      random_client.client->Communicate(puts, num_to_download, &response_stats);

  // If we communicated with a remote server, grab the server's reported stats
  // to update the approximation of the overall experiment stats.
  if (random_client.is_remote) {
    remote_stats_[random_client.client_num] = response_stats;
  }

  // Intercept `stats` and replace it with the overall experiment stats.
  if (stats != nullptr) {
    *stats = GetExperimentStats();
  }

  return result;
}

// Join all local servers when the destructor is run.
PopulationClientManager::~PopulationClientManager() {}

pair<vector<Individual>, SearchAlgorithmStats>
PopulationClientManager::Communicate(const vector<IndividualData>& uploads,
                                     const Integer num_to_download) {
  SearchAlgorithmStats stats;
  vector<Individual> downloads = Communicate(uploads, num_to_download, &stats);
  return std::make_pair(std::move(downloads), std::move(stats));
}

Integer PopulationClientManager::PingClients() {
  Integer successes = 0;
  for (auto& client : clients_) {
    if (client->Ping()) {
      successes += 1;
    }
  }
  return successes;
}

RandomClient PopulationClientManager::GetRandomClient() {
  Integer idx = -1;
  const Time start = Now();
  while (true) {
    idx = rng_.DiscreteInteger(distribution_);
    CHECK_GE(idx, 0);
    CHECK_LT(idx, clients_.size());
    if (target_rpc_periods_.empty() ||
        target_rpc_periods_[idx] == ZeroDuration()) {
      break;
    }
    const Time now = Now();
    if (now - latest_rpc_times_[idx] >= target_rpc_periods_[idx]) {
      latest_rpc_times_[idx] = now;
      break;
    }
    CHECK_LT(now - start, kMaxClientChooseTime)
        << "Took to long to choose an RPC client.";
  }
  if (GetFlag(FLAGS_log_which_client)) {
    LOG(INFO) << StrCat("Communicating with client #", idx);
  }
  return {clients_[idx].get(), /*is_remote=*/idx < num_remote_clients_,
          /*client_num=*/idx};
}

const std::vector<std::unique_ptr<PopulationClientInterface>>&
PopulationClientManager::GetAllClients() {
  return clients_;
}

std::vector<PopulationClientInterface*>
PopulationClientManager::GetAllRemoteClients() {
  std::vector<PopulationClientInterface*> remote_clients;
  remote_clients.reserve(num_remote_clients_);
  for (Integer index = 0; index < num_remote_clients_; ++index) {
    remote_clients.push_back(clients_[index].get());
  }
  return remote_clients;
}

std::vector<PopulationClientInterface*>
PopulationClientManager::GetAllLocalClients() {
  std::vector<PopulationClientInterface*> local_clients;
  for (Integer index = num_remote_clients_; index < clients_.size(); ++index) {
    local_clients.push_back(clients_[index].get());
  }
  return local_clients;
}

// See base class.
Integer PopulationClientManager::NumClients() { return clients_.size(); }

// See base class.
const std::vector<double> PopulationClientManager::GetProgressDelta() {
  return progress_delta_;
}


SearchAlgorithmStats PopulationClientManager::GetExperimentStats() {
  return AggregateStats(remote_stats_);
}

unique_ptr<PopulationClientInterface>
PopulationClientManager::CreateRpclessClient(
    const PopulationClientManagerElement& element) {
  if (!element.has_local_server_spec()) {
    LOG(FATAL) << "Trying to create a rpcless client without "
                  "local_server_spec";
  }
  RNG rng(rng_.UniformRNGSeed());
  return make_unique<RpclessPopulationClient>(
      element.client_spec(), element.local_server_spec(), rng.UniformRNGSeed(),
      rng.UniformRNGSeed(), worker_id_);
}

}  // namespace evolution
}  // namespace brain
