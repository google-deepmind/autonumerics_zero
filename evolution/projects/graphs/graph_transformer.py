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

"""Objects that transform one graph into another, given some preset rules."""

from typing import Any

from evolution.lib.python import pastable
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import graph_manipulation
from evolution.projects.graphs import graph_transformer_spec_pb2


def build_graph_transformer(
    spec: graph_transformer_spec_pb2.GraphTransformerSpec, op_init_params: Any
) -> "GraphTransformer":
  """Builds a graph transformer based on the spec."""
  if spec.HasField("followed"):
    return FollowedGraphTransformer(spec.followed, op_init_params)
  elif spec.HasField("continued"):
    return ContinuedGraphTransformer(spec.continued, op_init_params)
  elif spec.HasField("sequence"):
    return SequenceGraphTransformer(spec.sequence, op_init_params)
  else:
    raise ValueError("Unknown graph transformer.")


class GraphTransformer(object):
  """Base class for graph transformers."""

  def transform(self, graph: graph_lib.Graph) -> graph_lib.Graph:
    raise NotImplementedError("Must be implemented by subclass.")


class FollowedGraphTransformer(GraphTransformer):
  """A transformer that follows/composes the graph with a fixed graph."""

  def __init__(
      self,
      spec: graph_transformer_spec_pb2.FollowedGraphTransformerSpec,
      op_init_params: Any,
  ):
    self._following_graph = graph_lib.Graph()
    self._following_graph.deserialize(
        pastable.PastableToBinary(spec.following_graph),
        op_init_params=op_init_params,
    )

  def transform(self, graph: graph_lib.Graph) -> graph_lib.Graph:
    graph = graph.clone()
    graph.prune()
    graph_manipulation.dedup_vertex_ids(
        in_graph=graph, wrt_graph=self._following_graph
    )
    graph_manipulation.compose_graph(
        followed_graph=graph, following_graph=self._following_graph
    )
    return graph


class ContinuedGraphTransformer(GraphTransformer):
  """A composer that turns a graph into a continued graph."""

  def __init__(
      self,
      spec: graph_transformer_spec_pb2.ContinuedGraphTransformerSpec,
      op_init_params: Any,
  ):
    self._spec = spec
    if self._spec.max_num_vertices < 1:
      raise ValueError("Must specify a positive number of vertices.")
    if not self._spec.root_graph:
      raise ValueError("Must specify a root graph.")
    self._root_graph = graph_lib.Graph()
    self._root_graph.deserialize(
        pastable.PastableToBinary(self._spec.root_graph),
        op_init_params=op_init_params,
    )

  def transform(self, graph: graph_lib.Graph) -> graph_lib.Graph:
    graph = graph.clone()
    if self._spec.prune:
      graph.prune()
    num_root_optional_vertices = (
        len(self._root_graph.vertices)
        - len(self._root_graph.required_inputs)
        - len(self._root_graph.required_outputs)
    )
    num_required_vertices = (
        len(graph.required_inputs) - 1 + len(graph.required_outputs)
    )
    repeat_num_vertices = (
        len(graph.vertices)
        - len(graph.required_inputs)
        - len(graph.required_outputs)
    )
    available_num_vertices = (
        self._spec.max_num_vertices
        - num_root_optional_vertices
        - num_required_vertices
    )
    if repeat_num_vertices > 0:
      depth = int(available_num_vertices / repeat_num_vertices)
    else:
      depth = 1
    return graph_manipulation.continue_graph(
        graph=graph, depth=depth, root_graph=self._root_graph
    )


class SequenceGraphTransformer(GraphTransformer):
  """A transformer that applies a sequence of transformers."""

  def __init__(
      self,
      spec: graph_transformer_spec_pb2.SequenceGraphTransformerSpec,
      op_init_params: Any,
  ):
    if not spec.graph_transformers:
      raise ValueError("Must have at least one graph transformer.")
    self._transformers = []
    for transformer_spec in spec.graph_transformers:
      self._transformers.append(
          build_graph_transformer(transformer_spec, op_init_params)
      )

  def transform(self, graph: graph_lib.Graph) -> graph_lib.Graph:
    for transformer in self._transformers:
      graph = transformer.transform(graph)
    return graph
