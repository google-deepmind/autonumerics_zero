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

"""Utilities to manipulate graphs.

Provides the utilities:
-optional_vertex_ids
-disjoint_optional_vertex_ids
-dedup_graph_ids
-combine_outputs
-insert_subgraph
-compose_graph
-continue_graph
"""

import copy
from typing import Any, List, Set

from evolution.projects.graphs import graph as graph_lib


def optional_vertex_ids(graph: graph_lib.Graph) -> Set[str]:
  """Returns a set of the optional vertex IDs in the given graph."""
  ids = set(graph.vertices)
  ids.difference_update(graph.required_inputs)
  ids.difference_update(graph.required_outputs)
  return ids


def disjoint_optional_vertex_ids(
    graph1: graph_lib.Graph, graph2: graph_lib.Graph
) -> bool:
  """Returns True if the graphs have no optional vertex IDs in common."""
  ids1 = optional_vertex_ids(graph1)
  ids2 = optional_vertex_ids(graph2)
  return ids1.isdisjoint(ids2)


def dedup_vertex_ids(in_graph: graph_lib.Graph, wrt_graph: graph_lib.Graph):
  """Renames vertex IDs so that graphs don't have any in common.

  Required inputs and required outputs vertices are not deduped.

  Args:
    in_graph: one of the graphs to dedup. This graph's IDs may be modified.
    wrt_graph: the other graph to dedup. This graph will be kept constant.
  """

  def new_vertex_id():
    while True:
      vertex_id = in_graph.new_vertex_id()
      if vertex_id not in wrt_graph.vertices:
        return vertex_id

  vertex_ids_to_rename: List[str] = []
  for vertex_id in in_graph.vertices:
    if (
        (vertex_id not in in_graph.required_inputs)
        and (vertex_id not in in_graph.required_outputs)
        and vertex_id in wrt_graph.vertices
    ):
      vertex_ids_to_rename.append(vertex_id)
  for vertex_id in vertex_ids_to_rename:
    in_graph.rename_vertex(vertex_id, new_vertex_id())


def combine_outputs(
    into_graph: graph_lib.Graph,
    from_graph: graph_lib.Graph,
    combine_op_ids: List[str],
    combine_evolvable_params: List[Any],
    op_init_params: Any,
):
  """Merges a graph into another by combining their outputs.

  The resulting graph will contain graphs A (`into_graph`) and B (`from_graph`).
  Graph A will be modified in place and graph B will remain unchanged. The
  inputs will be shared by A and B. The outputs will be the result of applying
  the given ops to the outputs of A and B. For example, to produce the sum of
  two single-output graphs:
    combine_outputs(graph_a, graph_b, ["SumOp"], [None], op_init_params)

  A and B must have the same required inputs, as inputs are kept unaltered.
  They must also have the same outputs (outputs are combined in corresponding
  pairs).

  There must be no common vertex IDs between A and B, except for required inputs
  and required outputs.

  Args:
    into_graph: one of the two graphs to combine. Also the resulting graph, as
      it will be modified in place. Vertex IDs are preserved in resulting graph
      (and also the IDs of edges that do not need modification).
    from_graph: the other graph to combine. This one will not be modified.
      Vertex IDs are preserved in the resulting graph (but not edge IDs).
    combine_op_ids: a list of IDs of ops to use for combination. One combination
      vertex will be built with each. The elements of the list must correspond
      to the number of required outputs. The i-th combination vertex (with the
      i-th op in this list) will be used to combine the i-th output.
    combine_evolvable_params: a list of evolvable params to use for the
      combination vertices. The i-th element of this list will be used for the
      i-th combination vertex.
    op_init_params: the op init params.
  """
  if from_graph.required_inputs != into_graph.required_inputs:
    raise ValueError(
        "Cannot parallel-combine graphs with different required inputs."
    )
  if from_graph.required_outputs != into_graph.required_outputs:
    raise ValueError(
        "Cannot parallel-combine graphs with different required outputs."
    )
  if not disjoint_optional_vertex_ids(into_graph, from_graph):
    raise ValueError(
        "into_graph and from_graph must not have any vertex IDs in common, "
        "except for the required inputs and the required outputs. Consider "
        "calling `dedup_vertex_ids` first."
    )
  _handle_required_input_vertices_for_combine_outputs(
      into_graph=into_graph, from_graph=from_graph
  )
  _handle_optional_vertices_for_combine_outputs(
      into_graph=into_graph, from_graph=from_graph
  )
  _handle_required_output_vertices_for_combine_outputs(
      into_graph=into_graph,
      from_graph=from_graph,
      combine_op_ids=combine_op_ids,
      combine_evolvable_params=combine_evolvable_params,
      op_init_params=op_init_params,
  )
  into_graph.validate()


def _handle_required_input_vertices_for_combine_outputs(
    into_graph: graph_lib.Graph, from_graph: graph_lib.Graph
):
  assert from_graph.required_inputs == into_graph.required_inputs


def _handle_optional_vertices_for_combine_outputs(
    into_graph: graph_lib.Graph, from_graph: graph_lib.Graph
):
  """Combines the optional vertices and their input edges.

  Requires that the two graphs not share any vertex ID.

  Args:
    into_graph: see `combine_outputs`.
    from_graph: see `combine_outputs`.
  """
  sorted_vertex_ids_from = from_graph.topological_sort()
  for vertex_id in sorted_vertex_ids_from:
    if (
        vertex_id not in from_graph.required_inputs
        and vertex_id not in from_graph.required_outputs
    ):
      # The vertex in question in the from_graph.
      vertex_from = from_graph.vertices[vertex_id]

      # Copy the vertex.
      into_graph.insert_vertex(
          vertex_id=vertex_id,
          op=vertex_from.op,
          evolvable_params=copy.deepcopy(vertex_from.evolvable_params),
          mark_required_input=False,
          mark_required_output=False,
      )

      # Copy the vertex's input edges.
      for edge_id_from in vertex_from.in_edge_ids:
        edge_from = from_graph.edges[edge_id_from]
        # Use the fact that vertex IDs in both graphs match.
        into_graph.insert_edge(
            src_vertex_id=edge_from.src_vertex_id,
            dest_vertex_id=edge_from.dest_vertex_id,
            dest_vertex_in_index=edge_from.dest_vertex_in_index,
        )


def _handle_required_output_vertices_for_combine_outputs(
    into_graph: graph_lib.Graph,
    from_graph: graph_lib.Graph,
    combine_op_ids: List[str],
    combine_evolvable_params: List[Any],
    op_init_params: Any,
):
  """Combines the required outputs.

  Requires that the two graphs not share any vertex ID.

  Args:
    into_graph: see `combine_outputs`.
    from_graph: see `combine_outputs`.
    combine_op_ids: see `combine_outputs`.
    combine_evolvable_params: see `combine_outputs`.
    op_init_params: see `combine_outputs`.
  """
  assert from_graph.required_outputs == into_graph.required_outputs
  assert len(combine_op_ids) == len(into_graph.required_outputs)
  assert len(combine_evolvable_params) == len(into_graph.required_outputs)

  for output_vertex_id, combine_op_id, combine_params in zip(
      into_graph.required_outputs, combine_op_ids, combine_evolvable_params
  ):
    # We now generate the combination vertex for this output vertex.

    # First identify the src vertices of the combination vertex. These are the
    # edges that connect to the corresponding to the output in question in the
    # original graphs. Note that output vertex IDs in the original graphs
    # match, but the IDs of their src vertexes may not.
    output_vertex_into = (
        # The output vertex in the into_graph.
        into_graph.vertices[output_vertex_id]
    )
    output_vertex_from = (
        # The output vertex in the from_graph.
        from_graph.vertices[output_vertex_id]
    )
    assert len(output_vertex_into.in_edges) == 1  # Since it's an output vertex.
    assert len(output_vertex_from.in_edges) == 1  # Since it's an output vertex.
    assert output_vertex_into.in_edges[0] is not None
    assert output_vertex_from.in_edges[0] is not None
    src_vertex_id_into = output_vertex_into.in_edges[0].src_vertex_id
    src_vertex_id_from = output_vertex_from.in_edges[0].src_vertex_id

    # Disconnect the output vertex in the to_graph.
    into_graph.remove_edge(edge_id=output_vertex_into.in_edge_ids[0])

    # Create the combining vertex.
    combine_vertex_id = into_graph.new_vertex_id()
    combine_op = graph_lib.Op.build_op(
        combine_op_id, op_init_params=op_init_params
    )
    if len(combine_op.in_types) != 2:
      raise ValueError("The combine op must be binary (combining 2 graphs).")
    into_graph.insert_vertex(
        vertex_id=combine_vertex_id,
        op=combine_op,
        evolvable_params=combine_params,
        mark_required_input=False,
        mark_required_output=False,
    )

    # Connect the combining vertex to its 2 src vertices (the vertices that used
    # to be connected to the output in the original graphs).
    into_graph.insert_edge(
        src_vertex_id=src_vertex_id_into,
        dest_vertex_id=combine_vertex_id,
        dest_vertex_in_index=0,
    )
    into_graph.insert_edge(
        src_vertex_id=src_vertex_id_from,
        dest_vertex_id=combine_vertex_id,
        dest_vertex_in_index=1,
    )

    # Connect the combining vertex to the output.
    into_graph.insert_edge(
        src_vertex_id=combine_vertex_id,
        dest_vertex_id=output_vertex_id,
        dest_vertex_in_index=0,
    )


def insert_subgraph(
    in_graph: graph_lib.Graph,
    subgraph: graph_lib.Graph,
    glue_input_ids: List[str],
    glue_output_ids: List[str],
    glue_output_in_indexes: List[int],
):
  """Inserts a graph into another.

  There must be no common vertex IDs between A and B (ignoring the required
  inputs and required outputs of `subgraph`, which are discarded).

  Args:
    in_graph: the graph where to insert. This graph will be modified in place.
      Vertex IDs are preserved in resulting graph (and also the IDs of edges
      that do not need modification). No restrictions in the number of required
      inputs or outputs.
    subgraph: the graph to insert. This graph will not be modified. Vertex IDs
      are preserved in the resulting graph (but not edge IDs).
    glue_input_ids: a list of vertex IDs specifying how to connect the
      subgraph's required inputs. There must be one ID for each subgraph
      required input; the subgraph's input will be connected to the vertex with
      that ID in the `in_graph`. All IDs must exist in the `in_graph`.
    glue_output_ids: a list of vertex IDs specifying how to connect the
      subgraph's required outputs. There must be one ID for each subgraph
      required output; the subgraph's output will be connected to the vertex
      with that ID in the `in_graph`. All IDs must exist in the `in_graph`.
    glue_output_in_indexes: a list of in-indexes specifying how to connect the
      subgraph's required outputs. There must be one in-index value I for each
      subgraph required output; the subgraph's output will be connected to I-th
      input of a vertex in the `in_graph` (which vertex that is is determined by
      `glue_output_ids`).
  """
  if len(glue_input_ids) != len(subgraph.required_inputs):
    raise ValueError("The source IDs must correspond with the required inputs.")
  for src_id in glue_input_ids:
    if src_id not in in_graph.vertices:
      raise ValueError("Unknown source ID.")
  if len(glue_output_ids) != len(subgraph.required_outputs):
    raise ValueError(
        "The destination IDs must correspond with the required inputs."
    )
  for dest_id in glue_output_ids:
    if dest_id not in in_graph.vertices:
      raise ValueError("Unknown destination ID.")
  if not disjoint_optional_vertex_ids(subgraph, in_graph):
    raise ValueError(
        "into_graph and from_graph must not have any vertex IDs in common, "
        "except for the required inputs and the required outputs. Consider "
        "calling `dedup_vertex_ids` first."
    )

  # Iterate over the vertices in the subgraph in topological order.
  original_sorted_vertex_ids = subgraph.topological_sort()
  for original_vertex_id in original_sorted_vertex_ids:
    original_vertex = subgraph.vertices[original_vertex_id]

    if (
        original_vertex_id not in subgraph.required_inputs
        and original_vertex_id not in subgraph.required_outputs
    ):
      # This is an optional vertex, so it must be copied.
      target_vertex_id = original_vertex_id
      in_graph.insert_vertex(
          vertex_id=target_vertex_id,
          op=original_vertex.op,
          evolvable_params=copy.deepcopy(original_vertex.evolvable_params),
          mark_required_input=False,
          mark_required_output=False,
      )

    # Iterate over this subgraph vertex's inputs.
    for original_in_edge in original_vertex.in_edges:
      # Determine desired connectivity.
      assert original_in_edge
      original_src_vertex_id = original_in_edge.src_vertex_id
      target_src_vertex_id = _target_src_vertex_id_for_insert_subgraph(
          original_src_vertex_id, subgraph, glue_input_ids
      )
      original_dest_vertex_id = original_vertex_id
      target_dest_vertex_id = _target_dest_vertex_id_for_insert_subgraph(
          original_dest_vertex_id, subgraph, glue_output_ids
      )
      original_dest_vertex_in_index = original_in_edge.dest_vertex_in_index
      target_dest_vertex_in_index = (
          _target_dest_vertex_in_index_for_insert_subgraph(
              original_dest_vertex_id=original_dest_vertex_id,
              original_dest_vertex_in_index=original_dest_vertex_in_index,
              subgraph=subgraph,
              glue_output_in_indexes=glue_output_in_indexes,
          )
      )

      # Remove old edge in target graph.
      target_dest_vertex = in_graph.vertices[target_dest_vertex_id]
      old_edge_id = target_dest_vertex.in_edge_ids[target_dest_vertex_in_index]
      if old_edge_id is not None:
        # The `if` above is because if this vertex was just inserted, it won't
        # have in-edges. Also, the original graph may have missing input
        # connections that are fixed by the subgraph insertion being performed.
        in_graph.remove_edge(old_edge_id)

      # Replace it with the new edge in target graph.
      in_graph.insert_edge(
          src_vertex_id=target_src_vertex_id,
          dest_vertex_id=target_dest_vertex_id,
          dest_vertex_in_index=target_dest_vertex_in_index,
      )

  in_graph.validate()


def _target_src_vertex_id_for_insert_subgraph(
    original_src_vertex_id: str,
    subgraph: graph_lib.Graph,
    glue_input_ids: List[str],
) -> str:
  """Determines the vertex ID in the target graph.

  Args:
    original_src_vertex_id: the source vertex ID in the subgraph.
    subgraph: see `insert_subgraph`.
    glue_input_ids: see `insert_subgraph`.

  Returns:
    The corresponding vertex ID in the target graph (i.e. the `in_graph`).
  """
  if original_src_vertex_id in subgraph.required_inputs:
    index = subgraph.required_inputs.index(original_src_vertex_id)
    return glue_input_ids[index]
  else:
    return original_src_vertex_id


def _target_dest_vertex_id_for_insert_subgraph(
    original_dest_vertex_id: str,
    subgraph: graph_lib.Graph,
    glue_output_ids: List[str],
) -> str:
  """Determines the vertex ID in the target graph.

  Args:
    original_dest_vertex_id: the destination vertex ID in the subgraph.
    subgraph: see `insert_subgraph`.
    glue_output_ids: see `insert_subgraph`.

  Returns:
    The corresponding vertex ID in the target graph (i.e. the `in_graph`).
  """
  if original_dest_vertex_id in subgraph.required_outputs:
    index = subgraph.required_outputs.index(original_dest_vertex_id)
    return glue_output_ids[index]
  else:
    return original_dest_vertex_id


def _target_dest_vertex_in_index_for_insert_subgraph(
    original_dest_vertex_id: str,
    original_dest_vertex_in_index: int,
    subgraph: graph_lib.Graph,
    glue_output_in_indexes: List[int],
) -> int:
  """Determines the in-index in the target graph.

  Args:
    original_dest_vertex_id: the destination vertex ID in the subgraph.
    original_dest_vertex_in_index: the in-index of the destination vertex in the
      subgraph.
    subgraph: see `insert_subgraph`.
    glue_output_in_indexes: see `insert_subgraph`.

  Returns:
    The corresponding in-index of the vertex in the target graph (i.e. the
    `in_graph`).
  """
  if original_dest_vertex_id in subgraph.required_outputs:
    index = subgraph.required_outputs.index(original_dest_vertex_id)
    return glue_output_in_indexes[index]
  else:
    return original_dest_vertex_in_index


def compose_graph(
    followed_graph: graph_lib.Graph, following_graph: graph_lib.Graph
):
  """Composes the given graphs.

  Given graphs G(x) and H(x, y), constructs the graph H(x, G(x)). Thus,
  G must have two required inputs and 1 required output, and H must have 1
  required input and 1 required output. The first input of both graphs
  represents the same quantity and therefore can have the same ID, but this
  is not enforced. There must be no common ID among the optional vertices of G
  and H.

  Args:
    followed_graph: the graph G in the description above. Will be modified in
      place to produce the resulting graph. Vertex IDs are preserved in
      resulting graph (and also the IDs of edges that do not need modification).
    following_graph: the graph H in the description above. Will be left
      unchanged. Vertex IDs are preserved in the resulting graph (but not edge
      IDs).
  """
  if len(followed_graph.required_inputs) != 1:
    raise ValueError("The followed graph must have 1 input.")
  if len(following_graph.required_inputs) != 2:
    raise ValueError("The following graph must have 2 inputs.")

  if len(followed_graph.required_outputs) != 1:
    raise ValueError("The followed graph must have 1 output.")
  if len(following_graph.required_outputs) != 1:
    raise ValueError("The following graph must have 1 output.")

  if not disjoint_optional_vertex_ids(followed_graph, following_graph):
    raise ValueError(
        "followed_graph and following_graph must not have any vertex IDs in "
        "common, except for possibly the required inputs and the required "
        "outputs. Consider calling `dedup_vertex_ids` first."
    )

  input_id = followed_graph.required_inputs[0]
  output_id = followed_graph.required_outputs[0]
  output_in_edge = followed_graph.vertices[output_id].in_edges[0]
  assert output_in_edge is not None  # Required for PyType.
  output_src_id = output_in_edge.src_vertex_id
  insert_subgraph(
      in_graph=followed_graph,
      subgraph=following_graph,
      glue_input_ids=[input_id, output_src_id],
      glue_output_ids=[output_id],
      glue_output_in_indexes=[0],
  )


def continue_graph(
    graph: graph_lib.Graph, depth: int, root_graph: graph_lib.Graph
) -> graph_lib.Graph:
  """Creates a continued graph based on the given graph.

  Given G(x,y), produces the continued graph G^D(x) defined by:
    G^D(x) = G(x, G(x, G(x, ... G(x, R)))),
  where G appears D times on the RHS. The graph G must have two required inputs
  and one required output. The graph R must have zero required inputs and one
  required output.

  Args:
    graph: the graph G in the description above. It will be left unchanged.
    depth: the depth D in the description above.
    root_graph: the graph R in the description above.

  Returns:
    The graph G^N in the description above.
  """
  # Avoid modifying `graph` in place, as we'll need to dedup vertex IDs.
  graph = graph.clone()

  continued_graph = graph_lib.Graph()

  # Create the graph G^0.
  input_vertex_id = graph.required_inputs[0]
  continued_graph.insert_vertex(
      vertex_id=input_vertex_id,
      op=graph.vertices[input_vertex_id].op,
      evolvable_params=copy.deepcopy(
          graph.vertices[input_vertex_id].evolvable_params
      ),
      mark_required_input=True,
      mark_required_output=False,
  )
  output_vertex_id = graph.required_outputs[0]
  continued_graph.insert_vertex(
      vertex_id=output_vertex_id,
      op=graph.vertices[output_vertex_id].op,
      evolvable_params=copy.deepcopy(
          graph.vertices[output_vertex_id].evolvable_params
      ),
      mark_required_input=False,
      mark_required_output=True,
  )
  insert_subgraph(
      in_graph=continued_graph,
      subgraph=root_graph,
      glue_input_ids=[],
      glue_output_ids=[output_vertex_id],
      glue_output_in_indexes=[0],
  )

  # Continue the graph by iteratively inserting G, starting with G^0.
  for _ in range(depth):
    # Replace G^k with G^(k+1) by inserting G once.
    dedup_vertex_ids(in_graph=graph, wrt_graph=continued_graph)
    compose_graph(followed_graph=continued_graph, following_graph=graph)

  return continued_graph
