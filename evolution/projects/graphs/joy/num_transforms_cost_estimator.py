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

"""A simple cost estimator that counts some of the ops."""

from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import op as op_lib
from evolution.projects.graphs.joy import conversions
from evolution.projects.graphs.joy import cost_estimator_impl
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import num_transforms_cost_estimator_spec_pb2


JnpPreciseFloat = data_lib.JnpPreciseFloat
LearnableParams = graph_lib.LearnableParams
NumTransformsCostEstimatorSpec = (
    num_transforms_cost_estimator_spec_pb2.NumTransformsCostEstimatorSpec
)


class NumTransformsCostEstimator(cost_estimator_impl.CostEstimatorImpl):
  """Counts the numer of transforms. See NumTransformsCostEstimatorSpec."""

  def __init__(self, spec: NumTransformsCostEstimatorSpec):
    self._spec = spec

  def estimate(
      self, graph: graph_lib.Graph, learnable_params: LearnableParams
  ) -> float:
    del learnable_params
    clone = graph.clone()
    clone.prune()
    num_transforms = 0
    for vertex in clone.vertices.values():
      if not isinstance(vertex.op, op_lib.InputOp) and not isinstance(
          vertex.op, op_lib.OutputOp
      ):
        num_transforms += 1
    if (
        self._spec.HasField("min_num_transforms")
        and num_transforms < self._spec.min_num_transforms
    ):
      num_transforms = self._spec.min_num_transforms
    return float(conversions.flip_and_squash(JnpPreciseFloat(num_transforms)))
