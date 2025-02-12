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

"""A cost estimator that counts some of the ops, with given constraints."""

from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import cost_estimator_impl
from evolution.projects.graphs.joy import num_constants_cost_estimator_spec_pb2


LearnableParams = graph_lib.LearnableParams
NumConstantsCostEstimatorSpec = (
    num_constants_cost_estimator_spec_pb2.NumConstantsCostEstimatorSpec
)


class NumConstantsCostEstimator(cost_estimator_impl.CostEstimatorImpl):
  """Counts the numer of constants. See NumConstantsCostEstimatorSpec."""

  def __init__(self, spec: NumConstantsCostEstimatorSpec):
    self._spec = spec
    if self._spec.min_num_constants <= 0.0:
      raise ValueError("min_num_constants must be positive.")

  def estimate(
      self, graph: graph_lib.Graph, learnable_params: LearnableParams
  ) -> float:
    del learnable_params

    graph = graph.clone()
    graph.prune()

    num_constants = 0.0
    for vertex_id, vertex in graph.vertices.items():
      if vertex_id not in graph.required_inputs and not vertex.op.in_types:
        num_constants += 1.0

    # Apply minimum.
    num_constants = max(num_constants, self._spec.min_num_constants)

    # Return *speed*, not cost.
    return 1.0 / num_constants
