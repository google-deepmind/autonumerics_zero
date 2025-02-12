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

"""Custom mutators for AutoNumerics."""

from typing import Any, Callable, List, Optional

import numpy as np

from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.lib.python import rng as rng_lib
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import mutators
from evolution.projects.graphs import mutators_spec_pb2
from evolution.projects.graphs import op as op_lib
from evolution.projects.graphs import topology
from evolution.projects.graphs.joy import custom_mutators_spec_pb2
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import eval_metadata as eval_metadata_lib

JnpPreciseFloat = data_lib.JnpPreciseFloat
EvalMetadata = eval_metadata_lib.EvalMetadata

RANDOM_VERTEX_ID_SIZE = 5


def build_custom_mutator(
    spec: mutators_spec_pb2.MutatorSpec, **kwargs
) -> mutators.Mutator:
  """Builds the mutator indicated by the spec."""
  ext_spec = spec.Extensions[custom_mutators_spec_pb2.CustomMutatorSpec.ext]
  if ext_spec.HasField("insert_vertex_mutator"):
    return InsertVertexMutator(spec=ext_spec.insert_vertex_mutator, **kwargs)
  elif ext_spec.HasField("remove_vertex_mutator"):
    return RemoveVertexMutator(spec=ext_spec.remove_vertex_mutator, **kwargs)
  elif ext_spec.HasField("size_dependent_mutator"):
    return SizeDependentMutator(spec=ext_spec.size_dependent_mutator, **kwargs)
  elif ext_spec.HasField("vertex_id_mutator"):
    return VertexIdMutator(spec=ext_spec.vertex_id_mutator, **kwargs)
  else:
    return mutators.build_standard_mutator(
        spec=spec, build_mutator_fn=build_custom_mutator, **kwargs
    )


class InsertVertexMutator(mutators.Mutator):
  """A mutator that inserts a connected vertex with minimal damage to the graph.

  See the documentation in the `InsertVertexMutatorSpec` proto for more details.
  """

  def __init__(
      self,
      spec: custom_mutators_spec_pb2.InsertVertexMutatorSpec,
      op_init_params: Any,
      rng: np.random.RandomState,
      **kwargs
  ):
    """Constructs the instance.

    Args:
      spec: the specification for the instance.
      op_init_params: the op init params.
      rng: a random number generator.
      **kwargs: discarded or passed to downstream mutators (if any).
    """
    super().__init__()
    self._spec = spec
    self._op_init_params = op_init_params
    self._np_rng = rng
    self._rng = rng_lib.RNG(self._np_rng.randint(low=0, high=2**31))
    del kwargs

    # Build allowed ops.
    if not self._spec.allowed_op_ids:
      raise ValueError("Missing allowed_op_ids.")
    self._allowed_ops = []
    for allowed_op_id in self._spec.allowed_op_ids:
      allowed_op = graph_lib.Op.build_op(
          allowed_op_id, op_init_params=self._op_init_params
      )
      if not allowed_op.in_types and allowed_op.out_type is None:
        raise ValueError("Invalid allowed op.")
      self._allowed_ops.append(allowed_op)

    if not self._spec.HasField("max_num_vertices"):
      raise ValueError("Missing max_num_vertices.")

    # Build side ops.
    self._allowed_side_nonary_ops = []
    for allowed_op_id in self._spec.allowed_side_nonary_op_ids:
      allowed_op = graph_lib.Op.build_op(
          allowed_op_id, op_init_params=self._op_init_params
      )
      if allowed_op.in_types:
        raise ValueError("Op is not nonary.")
      if allowed_op.out_type is None:
        raise ValueError("Nonary op has no outputs, making it invalid.")
      self._allowed_side_nonary_ops.append(allowed_op)

  def mutate(
      self,
      parent: graph_lib.Graph,
      child_eval_metadata: Optional[EvalMetadata] = None,
      **kwargs
  ) -> graph_lib.Graph:
    """See base class."""
    # See InsertVertexMutatorSpec proto for definition of E and V.

    if child_eval_metadata is not None:
      child_eval_metadata.mutation_id = self.mutation_id
      _check_no_vertex_ids_in_metadata(child_eval_metadata)

    if not topology.is_connected_graph(parent):
      raise RuntimeError("Found disconnected parent.")
    if self._is_too_large(parent):
      return parent.clone()
    child = parent.clone()
    del parent

    removed_edge = self._remove_edge(graph=child)  # Edge E.
    new_vertex = _insert_new_vertex(  # Vertex V.
        graph=child, allowed_ops=self._allowed_ops, rng=self._np_rng
    )
    # Must connect the output first, so that cycles can be avoided while
    # connecting the inputs afterward.
    self._connect_new_vertex_output(  # From V to V_out.
        new_vertex=new_vertex, removed_edge=removed_edge, graph=child
    )
    self._connect_new_vertex_inputs(  # From V_in to V.
        new_vertex=new_vertex, removed_edge=removed_edge, graph=child
    )

    if self._spec.random_new_ids:
      # Rename new vertex to something random.
      new_random_vertex_id = _new_random_vertex_id(child, self._rng)
      child.rename_vertex(new_vertex.vertex_id, new_random_vertex_id)

    child.prune()

    if self._spec.log_mutation_id:
      printing.print_now("Executed mutation %s." % str(self.mutation_id))
    return child

  def _is_too_large(self, graph: graph_lib.Graph) -> bool:
    """Whether the graph is already at the maximum specified size."""
    return len(graph.vertices) >= self._spec.max_num_vertices

  def _remove_edge(self, graph: graph_lib.Graph) -> graph_lib.Edge:
    """Removes a random edge from the graph. Returns the edge removed."""
    edge_id = self._np_rng.choice(list(graph.edges.keys()))
    edge = graph.edges[edge_id]
    graph.remove_edge(edge_id)
    return edge

  def _connect_new_vertex_output(
      self,
      new_vertex: graph_lib.Vertex,
      removed_edge: graph_lib.Edge,
      graph: graph_lib.Graph,
  ):
    graph.insert_edge(
        src_vertex_id=new_vertex.vertex_id,
        dest_vertex_id=removed_edge.dest_vertex_id,
        dest_vertex_in_index=removed_edge.dest_vertex_in_index,
    )

  def _connect_new_vertex_inputs(
      self,
      new_vertex: graph_lib.Vertex,
      removed_edge: graph_lib.Edge,
      graph: graph_lib.Graph,
  ):
    # The possible indexes of inputs to V.
    new_vertex_in_indexes = list(range(len(new_vertex.op.in_types)))

    # If no inputs, nothing to connect.
    if not new_vertex_in_indexes:
      return

    # Pick an input at random, then connect to it from V_in.
    new_vertex_in_index_for_original_src = int(
        self._np_rng.choice(new_vertex_in_indexes)
    )
    new_vertex_in_indexes.remove(new_vertex_in_index_for_original_src)
    graph.insert_edge(
        src_vertex_id=removed_edge.src_vertex_id,
        dest_vertex_id=new_vertex.vertex_id,
        dest_vertex_in_index=new_vertex_in_index_for_original_src,
    )

    # Connect other inputs randomly.
    if self._allowed_side_nonary_ops:
      self._connect_new_vertex_inputs_from_side_vertices(
          new_vertex=new_vertex,
          new_vertex_in_indexes=new_vertex_in_indexes,
          graph=graph,
      )
    else:
      self._connect_new_vertex_inputs_from_random_vertices(
          new_vertex=new_vertex,
          new_vertex_in_indexes=new_vertex_in_indexes,
          graph=graph,
      )

  def _connect_new_vertex_inputs_from_random_vertices(
      self, new_vertex, new_vertex_in_indexes, graph
  ):
    input_vertex_candidates = _input_vertex_candidates(
        to_vertex_id=new_vertex.vertex_id, graph=graph
    )
    for dest_index in new_vertex_in_indexes:
      input_vertex_id = self._np_rng.choice(input_vertex_candidates)
      graph.insert_edge(
          src_vertex_id=input_vertex_id,
          dest_vertex_id=new_vertex.vertex_id,
          dest_vertex_in_index=dest_index,
      )

  def _connect_new_vertex_inputs_from_side_vertices(
      self, new_vertex, new_vertex_in_indexes, graph
  ):
    for dest_index in new_vertex_in_indexes:
      new_side_vertex = _insert_new_vertex(
          graph=graph,
          allowed_ops=self._allowed_side_nonary_ops,
          rng=self._np_rng,
      )
      graph.insert_edge(
          src_vertex_id=new_side_vertex.vertex_id,
          dest_vertex_id=new_vertex.vertex_id,
          dest_vertex_in_index=dest_index,
      )


class RemoveVertexMutator(mutators.Mutator):
  """A mutator that inserts a connected vertex with minimal damage to the graph.

  See the documentation in the `RemoveVertexMutatorSpec` proto for more details.
  """

  def __init__(
      self,
      spec: custom_mutators_spec_pb2.RemoveVertexMutatorSpec,
      rng: np.random.RandomState,
      **kwargs
  ):
    """Constructs the instance.

    Args:
      spec: the specification for the instance.
      rng: a random number generator.
      **kwargs: discarded or passed to downstream mutators (if any).
    """
    super().__init__()
    self._spec = spec
    self._rng = rng
    if not self._spec.HasField("min_num_vertices"):
      raise ValueError("Missing min_num_vertices.")

  def mutate(
      self,
      parent: graph_lib.Graph,
      child_eval_metadata: Optional[EvalMetadata] = None,
      **kwargs
  ) -> graph_lib.Graph:
    """See base class."""
    # See RemoveVertexMutatorSpec proto for definition of V, V', S_in and S_out.

    if child_eval_metadata is not None:
      child_eval_metadata.mutation_id = self.mutation_id

    if not topology.is_connected_graph(parent):
      raise RuntimeError("Found disconnected parent.")
    if self._is_too_small(parent):
      return parent.clone()
    child = parent.clone()
    del parent

    vertex_id_to_remove = _sample_vertex_to_remove(child, self._rng)
    vertex_to_remove = child.vertices[vertex_id_to_remove]

    # This is vertex V'.
    new_src_vertex_id = self._sample_new_src_vertex(
        vertex_id_to_remove=vertex_id_to_remove, graph=child
    )

    # Remove vertex V.
    child.remove_vertex(
        vertex_id=vertex_id_to_remove,
        remove_in_edges=True,
        remove_out_edges=False,
    )

    # Reconnect dangling inputs for vertices in S_out.
    new_src_type = child.vertices[new_src_vertex_id].op.out_type
    for out_edge_id in vertex_to_remove.out_edge_ids:
      if child.edges[out_edge_id].dest_type != new_src_type:
        raise RuntimeError(
            "Incompatible types. RemoveVertexMutator does not yet support "
            "type matching."
        )
      child.connect_src_vertex(
          src_vertex_id=new_src_vertex_id, edge_id=out_edge_id
      )
    child.prune()

    if self._spec.log_mutation_id:
      printing.print_now("Executed mutation %s." % str(self.mutation_id))
    return child

  def _is_too_small(self, graph: graph_lib.Graph) -> bool:
    """Whether the graph is already at the minimum specified size."""
    return len(graph.vertices) <= self._spec.min_num_vertices

  def _sample_new_src_vertex(
      self, vertex_id_to_remove: str, graph: graph_lib.Graph
  ) -> str:
    """Picks vertex V', the replacement of V for re-connection purposes."""
    vertex_to_remove = graph.vertices[vertex_id_to_remove]
    if vertex_to_remove.in_edge_ids:
      # If vertex has inputs, return one of them. I.e. one in S_in.
      new_src_vertex_id = self._rng.choice(
          vertex_to_remove.in_edges
      ).src_vertex_id
    else:
      # If vertex does not have inputs, pick a random vertex in the graph.
      candidate_vertex_ids = set(graph.vertices)
      candidate_vertex_ids -= set(graph.required_outputs)
      forward_vertices_cache = {}
      # Avoid self.
      if vertex_id_to_remove in candidate_vertex_ids:
        candidate_vertex_ids.remove(vertex_id_to_remove)
      for out_edge in vertex_to_remove.out_edges:
        # Avoid loops.
        if out_edge.dest_vertex_id in candidate_vertex_ids:
          candidate_vertex_ids.remove(out_edge.dest_vertex_id)
        # Avoid cycles.
        forward_vertices_ids = topology.forward_vertices_ids(
            start_vertex=out_edge.dest_vertex_id,
            graph=graph,
            cache=forward_vertices_cache,
        )
        candidate_vertex_ids -= set(forward_vertices_ids)
      assert candidate_vertex_ids
      new_src_vertex_id = self._rng.choice(sorted(list(candidate_vertex_ids)))
    return new_src_vertex_id


def _sample_vertex_to_remove(
    graph: graph_lib.Graph, rng: np.random.RandomState
) -> str:
  """Randomly picks a vertex to remove."""
  candidate_vertex_ids = set(graph.vertices)
  candidate_vertex_ids -= set(graph.required_inputs)
  candidate_vertex_ids -= set(graph.required_outputs)
  # Sorting is for determinism.
  candidate_vertex_ids = list(sorted(candidate_vertex_ids))
  assert candidate_vertex_ids
  vertex_id = rng.choice(candidate_vertex_ids)  # Vertex to remove.
  return vertex_id


def _insert_new_vertex(
    graph: graph_lib.Graph,
    allowed_ops: List[op_lib.Op],
    rng: np.random.RandomState,
) -> graph_lib.Vertex:
  """Inserts a new vertex into the graph with a random op."""
  op = rng.choice(allowed_ops)
  evolvable_params = None
  if op.has_evolvable_params:
    evolvable_params = op.create_evolvable_params(rng=rng)
  vertex_id = graph.new_vertex_id()
  graph.insert_vertex(
      vertex_id=vertex_id,
      op=op,
      evolvable_params=evolvable_params,
      mark_required_input=False,
      mark_required_output=False,
  )
  return graph.vertices[vertex_id]


def _input_vertex_candidates(
    to_vertex_id: str, graph: graph_lib.Graph
) -> List[str]:
  """Finds vertices in the given graph that could be inputs to the given vertex.

  The returned vertices won't form loops or cycles.

  Args:
    to_vertex_id: the ID of the vertex to which the candidates will connect. It
      is important that this vertex's outputs are already connected.
    graph: the graph in question.

  Returns:
    The vertex IDs of the candidates.
  """
  candidates = set(graph.vertices)
  candidates -= set(graph.required_outputs)

  # Avoid self-loop.
  candidates.remove(to_vertex_id)

  # Avoid cycles.
  forward_vertices_ids = topology.forward_vertices_ids(
      start_vertex=to_vertex_id, graph=graph
  )
  candidates -= set(forward_vertices_ids)

  assert candidates
  return sorted(list(candidates))


class SizeDependentMutator(mutators.Mutator):
  """A mutator that decides what to do based on parent(s)' size.

  See the documentation in the `SizeDependentMutatorSpec` proto for more
  details.
  """

  def __init__(
      self, spec: custom_mutators_spec_pb2.SizeDependentMutatorSpec, **kwargs
  ):
    """Constructs the instance.

    Args:
      spec: the specification for the instance.
      **kwargs: discarded or passed to downstream mutators (if any).
    """
    self._spec = spec

    if not self._spec.rules:
      raise ValueError("Missing rules.")
    self._rule_condition_fns = []
    self._rule_mutators = []
    for rule_spec in self._spec.rules:
      if not rule_spec.HasField("condition"):
        raise ValueError("Missing rule condition")
      self._rule_condition_fns.append(
          self._build_rule_condition_fn(rule_spec.condition)
      )
      if not rule_spec.HasField("mutator"):
        raise ValueError("Missing rule mutator")
      self._rule_mutators.append(
          build_custom_mutator(spec=rule_spec.mutator, **kwargs)
      )

    if not self._spec.HasField("fallback_mutator"):
      raise ValueError("Missing fallback_mutator.")
    self._fallback_mutator = build_custom_mutator(
        spec=self._spec.fallback_mutator, **kwargs
    )

  def mutate(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      parent: graph_lib.Graph,
      parent1: Optional[graph_lib.Graph],
      child_eval_metadata: Optional[EvalMetadata] = None,
      **kwargs
  ) -> graph_lib.Graph:
    """See base class."""
    if child_eval_metadata is not None:
      child_eval_metadata.mutation_id = self.mutation_id

    parent0 = parent
    del parent
    for rule_condition_fn, rule_mutator in zip(
        self._rule_condition_fns, self._rule_mutators
    ):
      if rule_condition_fn(parent0, parent1):
        return rule_mutator.mutate(parent=parent0, parent1=parent1, **kwargs)
    return self._fallback_mutator.mutate(
        parent=parent0, parent1=parent1, **kwargs
    )

  def _build_rule_condition_fn(
      self, condition_spec: custom_mutators_spec_pb2.SizeDependentCondition
  ) -> Callable[[graph_lib.Graph, Optional[graph_lib.Graph]], bool]:
    def rule_condition_fn(parent0, parent1) -> bool:
      if (
          condition_spec.HasField("first_min_size")
          and len(parent0.vertices) < condition_spec.first_min_size
      ):
        return False
      if (
          condition_spec.HasField("first_max_size")
          and len(parent0.vertices) >= condition_spec.first_max_size
      ):
        return False
      if (
          condition_spec.HasField("second_min_size")
          and len(parent1.vertices) < condition_spec.second_min_size
      ):
        return False
      if (
          condition_spec.HasField("second_max_size")
          and len(parent1.vertices) >= condition_spec.second_max_size
      ):
        return False
      if condition_spec.HasField("total_min_size") or condition_spec.HasField(
          "total_max_size"
      ):
        if parent1 is None and condition_spec.allow_single_parent:
          # A mutator can reuse parent0 as parent1
          effective_parent1 = parent0
        else:
          effective_parent1 = parent1
        num_required_inputs = len(parent0.required_inputs)
        assert len(effective_parent1.required_inputs) == num_required_inputs
        num_required_outputs = len(parent0.required_outputs)
        assert len(effective_parent1.required_outputs) == num_required_outputs
        total_size = (
            # Sum all vertices.
            len(parent0.vertices)
            + len(effective_parent1.vertices)
            -
            # Avoid double-counting required inputs and outputs.
            num_required_inputs
            - num_required_outputs
        )
        if (
            condition_spec.HasField("total_min_size")
            and total_size < condition_spec.total_min_size
        ):
          return False
        if (
            condition_spec.HasField("total_max_size")
            and total_size >= condition_spec.total_max_size
        ):
          return False
      return True

    return rule_condition_fn


class VertexIdMutator(mutators.Mutator):
  """A mutator that randomizes a vertex ID.

  See the documentation in the `VertexIdMutatorSpec` proto for more details.
  """

  def __init__(
      self,
      spec: custom_mutators_spec_pb2.VertexIdMutatorSpec,
      rng: np.random.RandomState,
      **kwargs
  ):
    """Constructs the instance.

    Args:
      spec: the specification for the instance.
      rng: a random number generator.
      **kwargs: discarded or passed to downstream mutators (if any).
    """
    del kwargs
    self._spec = spec
    self._np_rng = rng
    self._rng = rng_lib.RNG(self._np_rng.randint(low=0, high=2**31))

  def mutate(
      self,
      parent: graph_lib.Graph,
      child_eval_metadata: Optional[EvalMetadata] = None,
      **kwargs
  ) -> graph_lib.Graph:
    """See base class."""
    del kwargs
    child = parent.clone()
    del parent
    if child_eval_metadata is not None:
      child_eval_metadata.mutation_id = self.mutation_id
      _check_no_vertex_ids_in_metadata(child_eval_metadata)

    # Pick random vertex.
    old_vertex_id = self._np_rng.choice(sorted(list(child.vertices.keys())))

    new_vertex_id = _new_random_vertex_id(child, self._rng)
    child.rename_vertex(old_vertex_id, new_vertex_id)
    return child


def _new_random_vertex_id(graph: graph_lib.Graph, rng: rng_lib.RNG) -> str:
  while True:
    candidate = rng.UniformFileName(RANDOM_VERTEX_ID_SIZE)
    if candidate not in graph.vertices:
      return candidate


def _check_no_vertex_ids_in_metadata(eval_metadata: EvalMetadata):
  """Makes sure there are no vertex IDs in the metadata."""
  if eval_metadata is not None:
    if eval_metadata.learnable_params:
      raise RuntimeError(
          "Expected not to find leranable params yet. "
          "Are you not using the typical evolver?"
      )
    if eval_metadata.fingerprints:
      raise RuntimeError("Expected not to find fingerprints.")
