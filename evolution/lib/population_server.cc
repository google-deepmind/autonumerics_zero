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

#include "evolution/lib/population_server.h"

#include <algorithm>
#include <csignal>
#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "evolution/lib/current_time.h"
#include "evolution/lib/id_generator.h"
#include "evolution/lib/individual.pb.h"
#include "evolution/lib/mocking.h"
#include "evolution/lib/population_server.pb.h"
#include "evolution/lib/population_server_search_algorithms.h"
#include "evolution/lib/population_server_spec.pb.h"
#include "evolution/lib/printing.h"
#include "evolution/lib/rng.h"
#include "evolution/lib/search_algorithm.h"
#include "evolution/lib/search_algorithm_stats.pb.h"
#include "evolution/lib/server.pb.h"
#include "evolution/lib/snapshot.pb.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

using ::absl::Duration;
using ::absl::Nanoseconds;
using ::absl::SleepFor;
using ::std::cout;
using ::std::endl;
using ::std::function;
using ::std::make_unique;
using ::std::string;
using ::std::unique_ptr;
using ::std::vector;

constexpr Integer kNanosPerSecond = 1000000000;
constexpr Integer kSnapshotIDLength = 32;  // Must match population.sdl.

void VerboseMonitor(PopulationServer* population_server) {
  population_server->lock_.AssertHeld();
  SearchAlgorithmStats stats;
  vector<Individual> kills;
  population_server->algorithm_->Exchange({}, 0, nullptr, &stats, &kills);
  Print() << population_server->algorithm_->Report() << Flush();
  CHECK(kills.empty());
  Print() << "Monitoring stats: " << absl::StrCat(stats) << Flush();
  if (population_server->num_pending_requests_ >
      (population_server->spec_.max_pending_requests() / 10) * 9) {
    LOG(WARNING) << "Server currently has "
                 << population_server->num_pending_requests_
                 << " pending requests, which is near capacity.";
  }
}

void ShutDown(PopulationServer* population_server) {
  population_server->lock_.AssertHeld();
  SearchAlgorithmStats stats;
  const string server_state = population_server->Serialize(&stats);
  CHECK_EQ(raise(SIGTERM), 0);
}

function<void()> MonitorThreadLoop(PopulationServer* population_server,
                                   const Integer max_inactive_secs,
                                   const Integer monitor_every_secs) {
  const Integer max_inactive_nanos = max_inactive_secs * kNanosPerSecond;
  const Duration monitor_every =
      Nanoseconds(monitor_every_secs * kNanosPerSecond);
  return [population_server, max_inactive_secs, max_inactive_nanos,
          monitor_every] {
    if (max_inactive_nanos > 0) {
      while (true) {
        population_server->lock_.Lock();
        if (population_server->spec_.verbose_monitor()) {
          VerboseMonitor(population_server);
        }
        if (population_server->current_time_->Nanos() -
                    population_server->last_request_nanos_ >
                max_inactive_nanos &&
            population_server->last_request_nanos_ !=
                population_server->initial_nanos_) {
          cout << "Server has been inactive for " << max_inactive_secs
               << " seconds. Shutting down." << endl;
          ShutDown(population_server);
          population_server->lock_.Unlock();
          break;
        }
        population_server->lock_.Unlock();
        SleepFor(monitor_every);
      }
    }
  };
}

PopulationServer::PopulationServer(
    const PopulationServerSpec& spec, const RNGSeed rng_seed,
    ThreadSafeSearchAlgorithmInterface* mock_algorithm, const Integer worker_id,
    CurrentTimeInterface* mock_current_time,
    IDGeneratorInterface* mock_snapshot_id_generator)
    : spec_(spec),
      rng_(rng_seed),
      worker_id_(worker_id),
      owned_algorithm_(TryMakePopulationServerAlgorithmOwned(
          spec_.algorithm(), rng_.UniformRNGSeed(), mock_algorithm)),
      algorithm_(
          GetPointerToMockOrOwned(mock_algorithm, owned_algorithm_.get())),
      num_pending_requests_(0),
      owned_current_time_(MakeOwnedIfNoMock<CurrentTime>(mock_current_time)),
      current_time_(GetPointerToMockOrOwned(mock_current_time,
                                            owned_current_time_.get())),
      last_request_nanos_(current_time_->Nanos()),
      initial_nanos_(last_request_nanos_),
      owned_snapshot_id_generator_(MakeOwnedIfNoMock<IDGenerator>(
          mock_snapshot_id_generator, &rng_, kSnapshotIDLength,
          /*integers_to_mix=*/vector<Integer>{worker_id},
          /*strings_to_mix=*/vector<string>{string(spec.server_id())})),
      snapshot_id_generator_(GetPointerToMockOrOwned(
          mock_snapshot_id_generator, owned_snapshot_id_generator_.get())) {}

PopulationServer::~PopulationServer() {
  if (monitor_thread_ != nullptr) {
    monitor_thread_->join();
  }
}

void PopulationServer::StartMonitorThread() {
  monitor_thread_ = make_unique<std::thread>(MonitorThreadLoop(
      this, spec_.max_inactive_secs(), spec_.monitor_every_secs()));
}

void PopulationServer::Communicate(const CommunicateRequest* request,
                                   CommunicateResponse* response) {
  last_request_nanos_ = current_time_->Nanos();

  // Communicate with algorithm.
  const auto uploads =
      vector<Individual>(request->uploads().begin(), request->uploads().end());
  vector<Individual> downloads;
  unique_ptr<SearchAlgorithmStats> stats;
  if (request->get_stats()) {
    stats = make_unique<SearchAlgorithmStats>();
  }
  vector<Individual> kills;
  const ExchangeStatus status = algorithm_->Exchange(
      uploads, request->num_to_download(), &downloads, stats.get(), &kills);
  if (status == ExchangeStatus::OK) {
    response->set_status(ServerEnums::SUCCEEDED);
  } else if (status == ExchangeStatus::REJECTED) {
    response->set_status(ServerEnums::REJECTED);
  } else {
    LOG(FATAL) << "Algorithm responded with unknown status.";
  }
  for (const Individual& individual : downloads) {
    *response->add_downloads() = individual;
  }
  if (request->get_stats()) {
    CHECK(stats != nullptr);
    *response->mutable_stats() = *stats;
  }

  absl::MutexLock lock(&lock_);
}

string PopulationServer::Serialize(SearchAlgorithmStats* stats) {
  return algorithm_->Serialize(stats);
}

void PopulationServer::Deserialize(const string& serialized) {
  algorithm_->Deserialize(serialized);
}

Snapshot PopulationServer::TakeSnapshot() {
  Snapshot snapshot;
  CHECK(!spec_.server_id().empty());
  snapshot.set_server_id(spec_.server_id());
  snapshot.set_snapshot_id(snapshot_id_generator_->Generate());
  vector<Individual> population;
  SearchAlgorithmStats stats;
  vector<Individual> kills;
  algorithm_->Exchange({}, -1, &population, &stats, &kills);
  CHECK(kills.empty());
  *snapshot.mutable_stats() = stats;

  std::vector<double> max_fitnesses;
  std::vector<double> min_fitnesses;
  std::vector<double> total_fitnesses;
  // For each Individual.data.hashes[] dimension, the count of the number of
  // hash occurrence in the population.
  std::vector<absl::flat_hash_map<std::string, Integer>> hash_tallies;
  for (const Individual& individual : population) {
    *snapshot.add_individuals() = individual;
  }

  if (population.empty()) {
    return snapshot;
  }
  // The population is non-empty. Take population stats.
  for (int individual_id = 0; individual_id < population.size();
       individual_id++) {
    const Individual& individual = population[individual_id];
    if (individual_id == 0) {
      // First individual.
      for (double fitness : individual.data().fitnesses()) {
        max_fitnesses.push_back(fitness);
        min_fitnesses.push_back(fitness);
        total_fitnesses.push_back(fitness);
      }
      for (int hash_dim = 0; hash_dim < individual.data().hashes_size();
           hash_dim++) {
        hash_tallies.push_back({});
        hash_tallies[hash_dim][individual.data().hashes(hash_dim)] = 1;
      }
      continue;
    }

    for (Integer fitness_dim = 0;
         fitness_dim < individual.data().fitnesses_size(); fitness_dim++) {
      double fitness = individual.data().fitnesses(fitness_dim);
      max_fitnesses[fitness_dim] =
          std::max(max_fitnesses[fitness_dim], fitness);
      min_fitnesses[fitness_dim] =
          std::min(min_fitnesses[fitness_dim], fitness);
      total_fitnesses[fitness_dim] += fitness;
    }
    for (int hash_dim = 0; hash_dim < individual.data().hashes_size();
         hash_dim++) {
      hash_tallies[hash_dim][individual.data().hashes(hash_dim)]++;
    }
  }

  for (double max_fitness : max_fitnesses) {
    snapshot.mutable_current_population_stats()->add_max_fitnesses(max_fitness);
  }
  for (double min_fitness : min_fitnesses) {
    snapshot.mutable_current_population_stats()->add_min_fitnesses(min_fitness);
  }
  for (double total_fitness : total_fitnesses) {
    snapshot.mutable_current_population_stats()->add_mean_fitnesses(
        total_fitness / population.size());
  }
  for (int hash_dim = 0; hash_dim < hash_tallies.size(); hash_dim++) {
    auto& hash_tally = hash_tallies[hash_dim];
    const std::string most_frequent_hash =
        std::max_element(
            hash_tally.begin(), hash_tally.end(),
            [](const auto& x, const auto& y) { return x.second < y.second; })
            ->first;
    snapshot.mutable_current_population_stats()->add_mode_hash(
        most_frequent_hash);
    snapshot.mutable_current_population_stats()->add_mode_count(
        hash_tally[most_frequent_hash]);
  }
  return snapshot;
}

}  // namespace evolution
}  // namespace brain
