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
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import num_interactions_cost_estimator_spec_pb2


JnpPreciseFloat = data_lib.JnpPreciseFloat
LearnableParams = graph_lib.LearnableParams
NumInteractionsCostEstimatorSpec = (
    num_interactions_cost_estimator_spec_pb2.NumInteractionsCostEstimatorSpec
)


class NumInteractionsCostEstimator(cost_estimator_impl.CostEstimatorImpl):
  """Counts the numer of interactions. See NumInteractionsCostEstimatorSpec."""

  def __init__(self, spec: NumInteractionsCostEstimatorSpec):
    self._spec = spec
    if self._spec.min_weighted_interactions <= 0.0:
      raise ValueError("min_weighted_interactions must be positive.")
    if self._spec.op_weights:
      self._op_weights = {}
      for op_weight in self._spec.op_weights:
        if op_weight.weight < 0.0:
          raise ValueError("Op weight must be non-negative.")
        self._op_weights[op_weight.op_id] = op_weight.weight
    else:
      self._op_weights = None

  def estimate(
      self, graph: graph_lib.Graph, learnable_params: LearnableParams
  ) -> float:
    del learnable_params

    graph = graph.clone()
    graph.prune()

    # Get interacting vertices, excluding required inputs and outputs.
    relevant_vertex_ids = graph.interacting_vertex_ids()
    relevant_vertex_ids.difference_update(graph.required_inputs)
    relevant_vertex_ids.difference_update(graph.required_outputs)

    # Calculate the total cost based on the relevant vertices' ops.
    weighted_interactions = 0.0
    for vertex_id in relevant_vertex_ids:
      if self._op_weights is None:
        weighted_interactions += 1.0
      else:
        op_id = graph.vertices[vertex_id].op.op_id
        if op_id not in self._op_weights:
          raise ValueError("Unknown weight for op ID %s" % op_id)
        weighted_interactions += self._op_weights[op_id]

    # Apply minimum.
    weighted_interactions = max(
        weighted_interactions, self._spec.min_weighted_interactions
    )

    # Return *speed*, not cost.
    return 1.0 / weighted_interactions
