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

r"""Mutators.

These are objects that randomly modify an existing graph, respecting types and
the DAG structure.
"""

import re
import time
from typing import Optional, Set

import numpy as np

from evolution.lib import search_algorithm_stats_pb2
from evolution.lib.python import pastable
from evolution.lib.python import printing
from evolution.lib.python import timed_measurement
from evolution.projects.graphs import generators
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import hasher_interface
from evolution.projects.graphs import mutators_spec_pb2
from evolution.projects.graphs import topology


class Mutator(object):
  """Base class for objects that mutate graphs."""

  @classmethod
  def id(cls) -> str:
    """Returns an id for this class.

    Note this returns the short class name without package names.
    """
    return cls.__name__

  @property
  def mutation_id(self) -> str:
    """Returns the ID for this op."""
    return self.__class__.id()

  def mutate(self, parent: graph_lib.Graph, **kwargs) -> graph_lib.Graph:
    """Returns a mutated copy of the parent(s) or their recombination."""
    raise NotImplementedError("Must be implemented by the subclass.")

  def set_impact(self, impact_amount: float):
    """Sets the impact of future mutations.

    This method may not be supported by all mutators. It is needed only in
    some contexts, such as when the given mutators is wrapped inside a
    multi_impact_mutator in the experiment config.

    Args:
      impact_amount: the desired impact amount. The meaning of this value is
        mutator-dependent.
    """
    raise NotImplementedError(
        "If required, must be implemented by the subclass. Please check "
        "your config; e.g. did you mean to wrap in a multi_impact_mutator?"
    )


def build_standard_mutator(
    spec: mutators_spec_pb2.MutatorSpec,
    build_generator_fn=None,
    build_mutator_fn=None,
    **kwargs,
) -> Mutator:
  """Builds the mutator indicated by the spec.

  Args:
    spec: the spec describing the mutator to build.
    build_generator_fn: the function to call when building generators
      recursively. If `None`, then `build_standard_generator` function is used
      in the recursive calls. Set this when using custom generators (in that
      case, you have to write your own `build_custom_generator` function, which
      will be in charge of building the custom generators; if also these
      generators are used, build_custom_generator can delegate their building to
      this function, but requires setting
      build_generator_fn=build_custom_generator for correct recursion--see
      joy/custom_generators.py for an example).
    build_mutator_fn: the function to call when building mutators recursively.
      If `None`, then `build_standard_mutator` function is used in the recursive
      calls. Set this when using custom mutators (in that case, you have to
      write your own `build_custom_mutator` function, which will be in charge of
      building the custom mutators; if also these mutators are used,
      build_custom_mutator can delegate their building to this function, but
      requires setting build_mutator_fn=build_custom_mutator for correct
      recursion--see joy/custom_mutators.py for an example).
    **kwargs: other kwargs.

  Returns:
    The mutator built.
  """
  build_generator_fn = build_generator_fn or generators.build_standard_generator
  build_mutator_fn = build_mutator_fn or build_standard_mutator
  if spec.HasField("random_choice_mutator"):
    return RandomChoiceMutator(
        spec=spec.random_choice_mutator,
        build_generator_fn=build_generator_fn,
        build_mutator_fn=build_mutator_fn,
        **kwargs,
    )
  elif spec.HasField("edge_mutator"):
    return EdgeMutator(
        spec=spec.edge_mutator,
        build_generator_fn=build_generator_fn,
        build_mutator_fn=build_mutator_fn,
        **kwargs,
    )
  elif spec.HasField("retry_until_functional_change_mutator"):
    return RetryUntilFunctionalChangeMutator(
        spec=spec.retry_until_functional_change_mutator,
        build_generator_fn=build_generator_fn,
        build_mutator_fn=build_mutator_fn,
        **kwargs,
    )
  elif spec.HasField("fail_mutator"):
    return FailMutator(
        spec=spec.fail_mutator,
        build_generator_fn=build_generator_fn,
        build_mutator_fn=build_mutator_fn,
        **kwargs,
    )
  else:
    raise NotImplementedError(
        "Unsupported mutator. MutatorSpec:\n%s" % str(spec)
    )


class RandomChoiceMutator(Mutator):
  """A meta-mutator that randomly applies a mutator from a list.

  See the documentation in the RandomChoiceMutatorSpec proto for more details.
  """

  def __init__(
      self,
      spec: mutators_spec_pb2.RandomChoiceMutatorSpec,
      rng: np.random.RandomState,
      build_mutator_fn,
      **kwargs,
  ):
    """Constructs the instance.

    Args:
      spec: the specification for the instance.
      rng: a random number generator.
      build_mutator_fn: a function to allow recursive building of mutators based
        on their spec. See `build_standard_mutator`.
      **kwargs: discarded or passed to downstream mutators (if any).
    """
    self._spec = spec
    self._rng = rng
    assert self._spec.elements
    self._dependent_mutators = []
    weights = []
    for element in self._spec.elements:
      self._dependent_mutators.append(
          build_mutator_fn(spec=element.mutator, rng=rng, **kwargs)
      )
      if element.weight <= 0.0:
        raise ValueError("Found non-positive weight.")
      weights.append(element.weight)
    weights = np.array(weights)
    self._probabilities = np.divide(weights, sum(weights))

  def mutate(self, parent: graph_lib.Graph, **kwargs) -> graph_lib.Graph:
    """See base class."""
    dependent_mutator = self._rng.choice(
        self._dependent_mutators, p=self._probabilities
    )
    return dependent_mutator.mutate(parent, **kwargs)


class EdgeMutator(Mutator):
  """A simple class that randomly mutates a graph.

  See the documentation in the EdgeMutatorSpec proto for more details.
  """

  def __init__(
      self,
      spec: mutators_spec_pb2.EdgeMutatorSpec,
      rng: np.random.RandomState,
      **kwargs,
  ):
    """Constructs the instance.

    Args:
      spec: the specification for the instance.
      rng: a random number generator.
      **kwargs: discarded or passed to downstream mutators (if any).
    """
    self._frozen_edge_groups = []
    self._spec = spec
    self._rng = rng
    del kwargs
    self._num_edges_to_replace = None
    if self._spec.HasField("num_edges_to_replace"):
      self._num_edges_to_replace = self._spec.num_edges_to_replace
      if self._num_edges_to_replace <= 0:
        raise ValueError("The number of edges to replace must be positive.")
    for frozen_edge_group in self._spec.frozen_edge_groups:
      if not (
          frozen_edge_group.HasField("dest_vertex_id_re")
          or frozen_edge_group.HasField("src_vertex_id_re")
      ):
        raise ValueError(
            "Either dest_vertex_id_re or src_vertex_id_re must be set in edge"
            " groups."
        )
      src_vertex_id_re = ".*"
      dest_vertex_id_re = ".*"
      dest_vertex_in_index = -1
      if frozen_edge_group.HasField("src_vertex_id_re"):
        src_vertex_id_re = frozen_edge_group.src_vertex_id_re
      if frozen_edge_group.HasField("dest_vertex_id_re"):
        dest_vertex_id_re = frozen_edge_group.dest_vertex_id_re
      if frozen_edge_group.HasField("dest_vertex_in_index"):
        assert frozen_edge_group.dest_vertex_in_index >= 0
        dest_vertex_in_index = frozen_edge_group.dest_vertex_in_index

      self._frozen_edge_groups.append((
          re.compile(src_vertex_id_re),
          re.compile(dest_vertex_id_re),
          dest_vertex_in_index,
      ))

  def mutate(self, parent: graph_lib.Graph, **kwargs) -> graph_lib.Graph:
    """See base class."""
    child = parent.clone()

    # IDs of edges that do not have a src-vertex.
    dangling_edge_ids: Set[str] = set()

    # Disconnect a random subset of the edges without replacement.
    if self._num_edges_to_replace is None:
      raise ValueError(
          "The number of edges to replace has not been set. This could have "
          "been set through the num_vertices_to_replace in the spec or by "
          "setting the impact (such as by wrapping this mutator in a "
          "multi_impact_mutator)."
      )

    mutable_edge_ids = []
    frozen_edge_ids = set([])
    for edge_id, edge in child.edges.items():
      if self._edge_not_frozen(edge):
        mutable_edge_ids.append(edge_id)
      else:
        frozen_edge_ids.add(edge_id)

    if not mutable_edge_ids:
      return parent.clone()
    replace_edge_ids = self._rng.choice(
        sorted(mutable_edge_ids), size=self._num_edges_to_replace, replace=False
    )
    for edge_id in replace_edge_ids:
      dangling_edge_ids.add(edge_id)
      child.disconnect_src_vertex(edge_id=edge_id)

    # Reconnect the dangling edges.
    shuffled_dangling_edge_ids = sorted(list(dangling_edge_ids))
    self._rng.shuffle(shuffled_dangling_edge_ids)
    topology.reconnect_edges_recursive(
        edge_ids=shuffled_dangling_edge_ids,
        current_edge_index=0,
        graph=child,
        rng=self._rng,
    )

    # Reconnect out-missing vertices
    if self._spec.reconnect_out_missing_vertices:
      topology.try_reconnect_all_out_missing_vertices(
          graph=child, rng=self._rng, frozen_edge_ids=frozen_edge_ids
      )

    if self._spec.connected_only:
      child.prune()

    return child

  def _edge_not_frozen(self, edge: graph_lib.Edge) -> bool:
    return not any(
        self._frozen_group_matches_edge(
            edge, src_re, dst_re, dest_vertex_in_index
        )
        for src_re, dst_re, dest_vertex_in_index in self._frozen_edge_groups
    )

  def _frozen_group_matches_edge(
      self, edge: graph_lib.Edge, src_re, dst_re, dest_vertex_in_index: int
  ) -> bool:
    return (
        src_re.fullmatch(edge.src_vertex_id)
        and dst_re.fullmatch(edge.dest_vertex_id)
        and (
            dest_vertex_in_index == -1
            or edge.dest_vertex_in_index == dest_vertex_in_index
        )
    )

  def set_impact(self, impact_amount: float):
    if self._spec.HasField("num_edges_to_replace"):
      raise ValueError(
          "If setting impact, don't set num_edges_to_replace in the config."
      )
    self._num_edges_to_replace = round(impact_amount)
    if self._num_edges_to_replace <= 0:
      raise ValueError("Impact must round to a positive integer.")


class RetryUntilFunctionalChangeMutator(Mutator):
  """A meta-mutator that repeats a mutator until there is functional change.

  See the documentation in the RetryUntilFunctionalChangeMutatorSpec proto
  for more details.
  """

  def __init__(
      self,
      spec: mutators_spec_pb2.RetryUntilFunctionalChangeMutatorSpec,
      rng: np.random.RandomState,
      hasher: hasher_interface.Hasher,
      build_mutator_fn,
      **kwargs,
  ):
    """Constructs the instance.

    Args:
      spec: the specification for the instance.
      rng: a random number generator.
      hasher: a functional hasher. Only needed if `ensure_functional_difference`
        is enabled.
      build_mutator_fn: a function to allow recursive building of mutators based
        on their spec. See `build_standard_mutator`.
      **kwargs: discarded or passed to downstream mutators (if any).
    """
    self._spec = spec
    self._rng = rng
    assert self._spec.mutator
    self._dependent_mutator = build_mutator_fn(
        spec=self._spec.mutator, rng=rng, hasher=hasher, **kwargs
    )
    self._hasher = hasher
    if self._hasher is None:
      raise ValueError("A hasher is required.")
    if self._spec.max_retries <= 0:
      raise ValueError(
          "The maximum number of retries must be greater than zero."
      )

  def mutate(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      parent: graph_lib.Graph,
      stats: Optional[search_algorithm_stats_pb2.SearchAlgorithmStats],
      **kwargs,
  ) -> graph_lib.Graph:
    """See base class."""
    start_time = time.time()
    algorithm_was_dumped = False
    with timed_measurement.measure("hash_time"):
      parent_hash = self._hasher.hash(parent)
    child = parent
    for _ in range(self._spec.max_retries):
      if (
          self._spec.dump_algorithm_secs > 0.0
          and time.time() - start_time > self._spec.dump_algorithm_secs
          and not algorithm_was_dumped
      ):
        printing.print_now(
            "Took to long to mutate pastable algorithm %s"
            % pastable.BinaryToPastable(parent.serialize())
        )
        algorithm_was_dumped = True
      if self._spec.accumulate_mutations:
        child_or_parent = child
      else:
        child_or_parent = parent
      child = self._dependent_mutator.mutate(
          child_or_parent, stats=stats, **kwargs
      )
      with timed_measurement.measure("hash_time"):
        child_hash = self._hasher.hash(child)
      if child_hash != parent_hash:
        return child
    raise RuntimeError(
        "Max number of mutator retries reached. "
        "No functional changes found.\n"
        f"Parent: {parent.debug_string()}\n"
        "Pastable graph: %s"
        % pastable.BinaryToPastable(parent.serialize())
    )


class FailMutator(Mutator):
  """A mutator that always crashes."""

  def __init__(self, spec: mutators_spec_pb2.FailMutatorSpec, **kwargs):
    """Constructs the instance.

    Args:
      spec: the specification for the instance.
      **kwargs: discarded or passed to downstream mutators (if any).
    """
    del spec
    del kwargs

  def mutate(self, parent: graph_lib.Graph, **kwargs) -> graph_lib.Graph:
    """See base class."""

    raise RuntimeError("FailMutator.mutate was called.")
