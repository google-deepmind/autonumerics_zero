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

"""Tools to manipulate graphs."""

from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import op as op_lib
from evolution.projects.graphs.joy import constant_vertex_collapser_spec_pb2
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import interpretation


ConstantVertexCollapserSpec = (
    constant_vertex_collapser_spec_pb2.ConstantVertexCollapserSpec
)
LearnableParams = graph_lib.LearnableParams
FinalizedFn = interpretation.FinalizedFn
JnpPreciseFloat = data_lib.JnpPreciseFloat


class ConstantVertexCollapser:
  """See ConstantVertexCollapserSpec."""

  def __init__(self, spec: ConstantVertexCollapserSpec):
    self._spec = spec

  def collapse(self, graph: graph_lib.Graph) -> graph_lib.Graph:
    """Collapses constant vertices in a graph.

    This must be used before vertices have any values associated with them.
    Thus, the should not have been initialized or trained.

    Evolvable params in vertices are ignored. This can cause issues if there
    are evolvable parameters that affect the constants.

    Args:
      graph: the graph with vertices to collapse.

    Returns:
      The graph with collapsed vertices.
    """
    graph = graph.clone()

    graph.prune()
    for vertex_id in graph.topological_sort():
      vertex = graph.vertices[vertex_id]
      if vertex.op.has_evolvable_params:
        raise ValueError("Evolvable params not supported.")
      if self._is_collapsable(vertex, graph):
        self._collapse_vertex(vertex=vertex, graph=graph)

    # Remove disconnected inputs.
    graph.prune()

    return graph

  def _is_collapsable(
      self, vertex: graph_lib.Vertex, graph: graph_lib.Graph
  ) -> bool:
    """Whether all its source vertices are non-required inputs."""
    if not vertex.in_edges:
      # Nothing to collapse.
      return False
    for edge in vertex.in_edges:
      assert edge is not None
      src_vertex = graph.vertices[edge.src_vertex_id]
      if src_vertex.op.in_types:
        # This source vertex is not an input.
        return False
      elif src_vertex.op.op_id == "ProduceXOp":
        # This source vertex is the required input.
        return False
    if vertex.vertex_id in graph.required_outputs:
      # This vertex is an output.
      return False
    return True

  def _collapse_vertex(self, vertex: graph_lib.Vertex, graph: graph_lib.Graph):
    """Modifies the graph by collapsing the given vertex."""
    # Gather features to construct collapsed vertex.
    collapsed_vertex_id = vertex.vertex_id
    out_edge_ids = sorted(list(vertex.out_edge_ids))

    # Construct collapsed vertex op.
    assert len(vertex.in_edges) == 2
    assert vertex.in_edges[0] is not None
    assert vertex.in_edges[1] is not None
    assert (
        graph.vertices[vertex.in_edges[0].src_vertex_id].op.op_id
        == "RandomInitVariableOp"
    )
    assert (
        graph.vertices[vertex.in_edges[1].src_vertex_id].op.op_id
        == "RandomInitVariableOp"
    )
    collapsed_op = op_lib.Op.build_op(
        op_id="RandomInitVariableOp", op_init_params=None
    )

    # Remove collapsing vertex.
    graph.remove_vertex(vertex_id=vertex.vertex_id, remove_out_edges=False)
    del vertex

    # Construct collapsed vertex.
    graph.insert_vertex(
        vertex_id=collapsed_vertex_id,
        op=collapsed_op,
        evolvable_params=None,
        mark_required_input=False,
        mark_required_output=False,
    )

    # Join collapsed vertex to graph.
    for edge_id in out_edge_ids:
      graph.connect_src_vertex(
          src_vertex_id=collapsed_vertex_id, edge_id=edge_id
      )
