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

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_POPULATION_SERVER_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_POPULATION_SERVER_H_

#include <functional>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include "absl/base/thread_annotations.h"
#include "absl/flags/flag.h"
#include "absl/synchronization/mutex.h"
#include "evolution/lib/current_time.h"
#include "evolution/lib/id_generator.h"
#include "evolution/lib/population_server.pb.h"
#include "evolution/lib/population_server_spec.pb.h"
#include "evolution/lib/rng.h"
#include "evolution/lib/search_algorithm.h"
#include "evolution/lib/search_algorithm_stats.pb.h"
#include "evolution/lib/snapshot.pb.h"
#include "evolution/lib/types.h"

extern absl::Flag<bool> FLAGS_crogs_dna_hack;

namespace brain {
namespace evolution {
// Implementation of the server. Keep thread safe.
class PopulationServer {
 public:
  PopulationServer(const PopulationServerSpec& spec, RNGSeed rng_seed,
                   ThreadSafeSearchAlgorithmInterface* mock_algorithm,
                   const Integer worker_id = -1,
                   CurrentTimeInterface* mock_current_time = nullptr,
                   IDGeneratorInterface* mock_snapshot_id_generator = nullptr);
  ~PopulationServer();
  PopulationServer(const PopulationServer& other) = delete;
  PopulationServer& operator=(const PopulationServer& other) = delete;
  void StartMonitorThread();
  void Communicate(const CommunicateRequest* request,
                   CommunicateResponse* response) ABSL_LOCKS_EXCLUDED(lock_);

 private:
  friend void ShutDown(PopulationServer*);
  friend void VerboseMonitor(PopulationServer*);
  friend std::function<void()> MonitorThreadLoop(PopulationServer*,
                                                 const Integer, const Integer);

  // The stats are filled in-sync with the serialized state. stats can be
  // nullptr.
  std::string Serialize(SearchAlgorithmStats* stats = nullptr)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  void Deserialize(const std::string& serialized)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  Snapshot TakeSnapshot() ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  const PopulationServerSpec spec_;

  // This Mutex is only used to lock rare events. It should not be used to
  // lock `algorithm_`.
  absl::Mutex lock_;

  RNG rng_ ABSL_GUARDED_BY(lock_);

  // For local servers only. worker_id should be set to -1 for remote
  // server.
  const Integer worker_id_;
  const std::unique_ptr<ThreadSafeSearchAlgorithmInterface> owned_algorithm_;
  ThreadSafeSearchAlgorithmInterface* const algorithm_;
  AtomicInteger num_pending_requests_;

  // A thread to print regular updates and shut down the server when
  // inactive.
  std::unique_ptr<std::thread> monitor_thread_;

  const std::unique_ptr<CurrentTimeInterface> owned_current_time_;
  CurrentTimeInterface* const current_time_;
  // Timestamp of the last request received by the server.
  AtomicInteger last_request_nanos_;
  // Timestamp of when the server is initialized.
  const Integer initial_nanos_;

  const std::unique_ptr<IDGeneratorInterface> owned_snapshot_id_generator_;
  IDGeneratorInterface* const snapshot_id_generator_;
};

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_POPULATION_SERVER_H_
