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

#include "evolution/lib/nsga2_util.h"

#include <memory>
#include <queue>
#include <vector>

#include "absl/log/check.h"
#include "evolution/lib/candidate.h"
#include "evolution/lib/individual.pb.h"

namespace brain {
namespace evolution {

using ::std::queue;
using ::std::shared_ptr;
using ::std::vector;

namespace {

// Returns true if individual1 Pareto-dominates individual2.
template <class IndividualT>
bool ParetoDominates(const IndividualT& candidate1,
                     const IndividualT& candidate2) {
  bool dominates = false;
  const int num_fitnesses = FitnessesSize(candidate1);
  CHECK_EQ(FitnessesSize(candidate2), num_fitnesses);
  for (int i = 0; i < num_fitnesses; ++i) {
    if (Fitness(candidate1, i) < Fitness(candidate2, i)) {
      return false;
    } else if (Fitness(candidate1, i) > Fitness(candidate2, i)) {
      dominates = true;
    }
  }
  return dominates;
}

}  // namespace

template <class IndividualT>
vector<vector<IndividualT>> FastNonDominatedSortImpl(
    const vector<IndividualT>& candidates) {
  vector<vector<int>> dependency_graph(candidates.size(), vector<int>());
  vector<int> indegree(candidates.size(), 0);
  queue<int> queue;
  for (int i = 0; i < candidates.size(); ++i) {
    for (int j = 0; j < candidates.size(); ++j) {
      if (ParetoDominates(candidates[i], candidates[j])) {
        dependency_graph[i].push_back(j);
      } else if (ParetoDominates(candidates[j], candidates[i])) {
        indegree[i]++;
      }
    }
    if (!indegree[i]) {
      queue.push(i);
    }
  }

  vector<vector<IndividualT>> fronts;
  while (!queue.empty()) {
    int len = queue.size();
    fronts.push_back(vector<IndividualT>());
    vector<IndividualT>& front = fronts.back();
    while (len-- > 0) {
      int parent = queue.front();
      queue.pop();
      front.push_back(candidates[parent]);
      for (const int child : dependency_graph[parent]) {
        indegree[child]--;
        if (indegree[child] == 0) {
          queue.push(child);
        }
      }
    }
  }
  return fronts;
}

vector<vector<Individual>> FastNonDominatedSort(
    const vector<Individual>& candidates) {
  return FastNonDominatedSortImpl(candidates);
}

vector<vector<shared_ptr<Candidate>>> FastNonDominatedSort(
    const vector<shared_ptr<Candidate>>& candidates) {
  return FastNonDominatedSortImpl(candidates);
}

}  // namespace evolution
}  // namespace brain
