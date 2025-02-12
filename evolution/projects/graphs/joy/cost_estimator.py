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

"""Cost estimator."""

from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import cost_estimator_spec_pb2
from evolution.projects.graphs.joy import interpretation
from evolution.projects.graphs.joy import num_constants_cost_estimator
from evolution.projects.graphs.joy import num_interactions_cost_estimator
from evolution.projects.graphs.joy import num_transforms_cost_estimator
from evolution.projects.graphs.joy import wall_clock_cost_estimator

FinalizedFn = interpretation.FinalizedFn
LearnableParams = graph_lib.LearnableParams


class CostEstimator(object):
  """A class to estimate the cost of a graph or function."""

  def __init__(self, spec: cost_estimator_spec_pb2.CostEstimatorSpec):
    if spec.HasField("num_transforms"):
      self.cost_estimator_impl = (
          num_transforms_cost_estimator.NumTransformsCostEstimator(
              spec.num_transforms
          )
      )
    elif spec.HasField("num_interactions"):
      self.cost_estimator_impl = (
          num_interactions_cost_estimator.NumInteractionsCostEstimator(
              spec.num_interactions
          )
      )
    elif spec.HasField("num_constants"):
      self.cost_estimator_impl = (
          num_constants_cost_estimator.NumConstantsCostEstimator(
              spec.num_constants
          )
      )
    elif spec.HasField("wall_clock"):
      self.cost_estimator_impl = (
          wall_clock_cost_estimator.WallClockCostEstimator(spec.wall_clock)
      )
    else:
      raise NotImplementedError("Unknown cost estimator.")

  def estimate(
      self, graph: graph_lib.Graph, learnable_params: LearnableParams
  ) -> float:
    """Measures the cost of a graph."""
    return self.cost_estimator_impl.estimate(
        graph=graph, learnable_params=learnable_params
    )

  def estimate_finalized_fn(self, finalized_fn: FinalizedFn) -> float:
    """Measures the relative/normalized cost of a finalized function."""
    return self.cost_estimator_impl.estimate_finalized_fn(
        finalized_fn=finalized_fn
    )
