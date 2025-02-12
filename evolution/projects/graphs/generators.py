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

r"""Generators.

These are objects that produce a random graph, respecting types and the DAG
structure.
"""

from typing import Any, Optional

import numpy as np

from evolution.projects.graphs import generators_spec_pb2
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import topology


class Generator(object):
  """Base class for objects that generate graphs."""

  def generate(self) -> graph_lib.Graph:
    """Generates one graph."""
    raise NotImplementedError("Must be implemented by the subclass.")

  def unprune(self, graph: graph_lib.Graph):
    """Generates additional vertices to increase the size of a valid graph.

    This method acts on an input graph that is valid (i.e. has the required
    inputs and outputs, does not have dangling edges or vertex inputs, etc.)
    but does not have the target number of vertices that this generator is
    configured for. This method will add enough non-contributing vertices to
    bring it to the desired size.

    Args:
      graph: the input graph (see above). It will be modified in place.
    """
    raise NotImplementedError("Must be implemented by the subclass.")


def build_standard_generator(
    spec: generators_spec_pb2.GeneratorSpec, **kwargs
) -> Generator:
  """Builds the generator indicated by the spec.

  Args:
    spec: the spec describing the generator to build.
    **kwargs: other kwargs.

  Returns:
    The generator built.
  """
  if spec.HasField("random_generator"):
    return RandomGenerator(spec=spec.random_generator, **kwargs)
  else:
    raise NotImplementedError("Unsupported generator.")


class RandomGenerator(Generator):
  """A simple class that generates random graphs.

  See the documentation in the RandomGeneratorSpec proto for details.
  """

  def __init__(
      self,
      spec: generators_spec_pb2.RandomGeneratorSpec,
      op_init_params: Any,
      rng: np.random.RandomState,
      **kwargs
  ):
    """Constructs the instance.

    Args:
      spec: the specification for the instance.
      op_init_params: parameters to pass to the op initialization. Can be any
        container with all the relevant parameters (typically a proto or a
        dict). It is passed to the __init__ method of the ops as-is and it is
        never read elsewhere. Can be None if none of the ops needs it.
      rng: a seed for the random number generator.
      **kwargs: discarded or passed to downstream generators (if any).
    """
    del kwargs
    self._spec = spec
    self._op_init_params = op_init_params
    self._rng = rng

    self._input_vertex_ops = []
    if not spec.required_input_vertex_op_ids:
      raise ValueError("Input vertices are required.")
    for input_vertex_op_id in spec.required_input_vertex_op_ids:
      input_vertex_op = graph_lib.Op.build_op(
          input_vertex_op_id, op_init_params=op_init_params
      )
      if input_vertex_op.in_types:
        raise ValueError("Invalid input vertex op.")
      self._input_vertex_ops.append(input_vertex_op)

    self._output_vertex_ops = []
    if not spec.required_output_vertex_op_ids:
      raise ValueError("Output vertices are required.")
    for output_vertex_op_id in spec.required_output_vertex_op_ids:
      output_vertex_op = graph_lib.Op.build_op(
          output_vertex_op_id, op_init_params=op_init_params
      )
      if output_vertex_op.out_type is not None:
        raise ValueError("Invalid output vertex op.")
      self._output_vertex_ops.append(output_vertex_op)

    self._allowed_ops = []
    for allowed_op_id in spec.allowed_op_ids:
      allowed_op = graph_lib.Op.build_op(
          allowed_op_id, op_init_params=op_init_params
      )
      if not allowed_op.in_types and allowed_op.out_type is None:
        raise ValueError("Invalid allowed op.")
      self._allowed_ops.append(allowed_op)

    if self._spec.num_vertices < len(self._input_vertex_ops) + len(
        self._output_vertex_ops
    ):
      raise ValueError("Too few vertices to build.")

    if self._spec.max_attempts == 0 or self._spec.max_attempts < -1:
      raise ValueError("Invalid number of attempts.")

    # If set, always generates this graph.
    self._single_generated_graph: Optional[graph_lib.Graph] = None

    self._op_ids_need_different_inputs = set(
        self._spec.op_ids_need_different_inputs
    )

  def generate(self) -> graph_lib.Graph:
    """See base class."""
    if self._single_generated_graph:
      return self._single_generated_graph.clone()
    graph = None
    if self._spec.single_graph:
      seed_value = 1
      self._rng.seed(seed_value)
    for _ in range(self._spec.max_attempts):
      graph = self.try_generate()
      if graph is not None:
        break
    if graph is None:
      raise RuntimeError("Max number of generation attempts reached.")
    if self._spec.single_graph:
      self._single_generated_graph = graph
    return graph

  def unprune(self, graph: graph_lib.Graph):
    """See base class."""
    graph.validate()

    # The vertex IDs of the vertices that need to be created.
    additional_vertex_ids = set()
    num_additional_vertices = self._spec.num_vertices - len(graph.vertices)
    if num_additional_vertices < 0:
      raise ValueError("Graph is too large.")
    current_index = 0
    while len(additional_vertex_ids) < num_additional_vertices:
      candidate_vertex_id = "v%s" % current_index
      if candidate_vertex_id not in graph.vertices:
        additional_vertex_ids.add(candidate_vertex_id)
      current_index += 1

    if additional_vertex_ids:
      additional_vertex_ids = sorted(list(additional_vertex_ids))
      topology.insert_vertex_recursive(
          allowed_ops=self._allowed_ops,
          vertex_ids=additional_vertex_ids,
          graph=graph,
          rng=self._rng,
          op_init_params=self._op_init_params,
          op_ids_need_different_inputs=self._op_ids_need_different_inputs,
      )

  def try_generate(self) -> Optional[graph_lib.Graph]:
    """A single attempt to generate a new graph."""
    graph = graph_lib.Graph()

    # Generate input vertices.
    for op in self._input_vertex_ops:
      vertex_id = "v%d" % len(graph.vertices)
      evolvable_params = None
      if op.has_evolvable_params:
        evolvable_params = op.create_evolvable_params(self._rng)
      graph.insert_vertex(
          vertex_id=vertex_id,
          op=op,
          evolvable_params=evolvable_params,
          mark_required_input=True,
          mark_required_output=False,
      )

    # Generate intermediate vertices.
    start_vertex_id = len(self._input_vertex_ops)
    end_vertex_id = self._spec.num_vertices - len(self._output_vertex_ops)
    intermediate_vertex_ids = [
        "v%s" % i for i in range(start_vertex_id, end_vertex_id)
    ]
    topology.insert_vertex_recursive(
        allowed_ops=self._allowed_ops,
        vertex_ids=intermediate_vertex_ids,
        graph=graph,
        rng=self._rng,
        op_init_params=self._op_init_params,
        op_ids_need_different_inputs=self._op_ids_need_different_inputs,
    )

    # Generate output vertices.
    for candidate_op in self._output_vertex_ops:
      vertex_id = "v%d" % len(graph.vertices)
      if not topology.try_insert_new_connected(
          candidate_op=candidate_op,
          vertex_id=vertex_id,
          mark_required_input=False,
          mark_required_output=True,
          graph=graph,
          rng=self._rng,
          op_init_params=self._op_init_params,
          need_different_inputs=(
              candidate_op.id() in self._op_ids_need_different_inputs
          ),
      ):
        return None  # Failed to connect to outputs.

    if self._spec.connected_only:
      graph.prune()

    if self._spec.exact_size and len(graph.vertices) != self._spec.num_vertices:
      return None

    return graph
