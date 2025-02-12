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

"""A class to fingerprint vertices."""

from jax import numpy as jnp
from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import dataset as dataset_lib
from evolution.projects.graphs.joy import eval_metadata as eval_metadata_lib
from evolution.projects.graphs.joy import fingerprinter_spec_pb2
from evolution.projects.graphs.joy import interpretation

ExecutionOutputs = eval_metadata_lib.ExecutionOutputs
JnpPreciseFloat = data_lib.JnpPreciseFloat
LearnableParams = graph_lib.LearnableParams


class Fingerprinter(object):
  """Fingerprints the vertices of a graph."""

  def __init__(self, spec: fingerprinter_spec_pb2.FingerprinterSpec):
    self._spec = spec
    if not self._spec.HasField("dataset"):
      raise ValueError("Missing dataset.")
    self._dataset = dataset_lib.build(self._spec.dataset)

  def fingerprint(
      self, graph: graph_lib.Graph, learnable_params: graph_lib.LearnableParams
  ) -> ExecutionOutputs:
    """Performs the fingerprinting.

    Args:
      graph: the graph to validate.
      learnable_params: the parameters to use with the graph.

    Returns:
      The fingerprints for all the vertices.

    Raises:
      RuntimeError: if there is a type mismatch.
    """
    data_lib.check_learnable_params_dtype(
        learnable_params, dtype=JnpPreciseFloat
    )
    inputs = self._dataset.inputs()
    execution_outputs = interpretation.execute_graph(
        graph=graph,
        learnable_params=learnable_params,
        inputs=inputs,
        dtype=self._dataset.inputs_dtype,
        contributing_only=False,
    )

    # Transform the execution outputs to the correct format (a jnp array for
    # each vertex).
    fingerprints = {}
    inputs_shape = inputs[0].shape
    for vertex_id, execution_output in execution_outputs.items():
      if isinstance(execution_output, jnp.ndarray):
        # If a constant never interacts with the inputs, it won't be broadcasted
        # to the correct shape. We fix that corner case here.
        broadcasted_outputs = jnp.broadcast_to(execution_output, inputs_shape)

        fingerprints[vertex_id] = broadcasted_outputs
      else:
        raise NotImplementedError(
            "No support for fingerprinting outputs of type %s."
            % str(type(execution_output))
        )

    for vertex_output in fingerprints.values():
      if vertex_output.dtype != self._dataset.inputs_dtype:
        raise RuntimeError("Found fingerprint with unexpected dtype.")

    return fingerprints
