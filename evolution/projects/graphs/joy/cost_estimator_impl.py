# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Interface for cost estimators."""

from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import interpretation

FinalizedFn = interpretation.FinalizedFn
LearnableParams = graph_lib.LearnableParams


class CostEstimatorImpl(object):
  """Interface for all cost estimator implementations."""

  def estimate(
      self, graph: graph_lib.Graph, learnable_params: LearnableParams
  ) -> float:
    """Measures the cost of a graph."""
    raise NotImplementedError("Must be implemented by subclass.")

  def estimate_finalized_fn(self, finalized_fn: FinalizedFn) -> float:
    """Measures the relative/normalized cost of a finalized function."""
    raise NotImplementedError("Must be implemented by subclass.")
