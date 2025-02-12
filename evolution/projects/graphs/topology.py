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

r"""Common utilities for graphs that need to consider its topology.

Basic modifications, like inserting an edge between fixed vertices should go in
the Graph class (in graph.py). Here we put more complex ops, like graph
algorithms or stochastic graph edits.
"""

import collections
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import op as op_lib


def insert_vertex_recursive(
    allowed_ops: List[graph_lib.Op],
    vertex_ids: List[str],
    graph: graph_lib.Graph,
    rng: np.random.RandomState,
    op_init_params: Any,
    current_vertex_index: int = 0,
    op_ids_need_different_inputs: Optional[Set[str]] = None,
) -> bool:
  """Recursive helper to randomly insert vertices into the graph.

  Will insert vertices one at a time. Any existing vertices can be the source
  of new edges. Each inserted vertex is connected to a random subset of
  available existing vertices, matching types. Once connected, the new vertex
  provides a new potential source of edges to connect to for subsequently
  inserted vertices. The function returns True once the graph reaches the
  desired number of vertices. If it is not possible to insert vertices in a
  type-consistent manner, this function returns false.

  This method will not introduce inconsistencies to the graph, but the output
  of any inserted vertex may not be used (in fact, the output of the last
  inserted vertex is not used). Typically, after this method is called more
  steps would take place to possibly add such connections.

  Args:
    allowed_ops: a list of allowed ops for the new vertices.
    vertex_ids: the IDs of the vertices to insert.
    graph: the graph where to insert.
    rng: the random number generator to use
    op_init_params: parameters to pass to the op initialization. Can be any
      container with all the relevant parameters (typically a proto or a dict).
      It is passed to the __init__ method of the ops as-is and it is never read
      elsewhere.
    current_vertex_index: the index in the list of vertex_ids to insert. Used in
      the recursion to keep track of call stack depth. Leave as 0 when calling
      this function from outside this function.
    op_ids_need_different_inputs: a set of op_ids that need different inputs.

  Returns:
    Whether all the vertices have been inserted. I.e. whether the maximum
    recursion depth has been reached.
  """
  if current_vertex_index == len(vertex_ids):
    return True
  if not allowed_ops:
    raise ValueError("Need allowed ops.")
  op_ids_need_different_inputs = op_ids_need_different_inputs or set()
  candidate_ops = allowed_ops[:]
  rng.shuffle(candidate_ops)
  for candidate_op in candidate_ops:
    vertex_id = vertex_ids[current_vertex_index]
    if try_insert_new_connected(
        candidate_op=candidate_op,
        vertex_id=vertex_id,
        mark_required_input=False,
        mark_required_output=False,
        graph=graph,
        rng=rng,
        op_init_params=op_init_params,
        need_different_inputs=(
            candidate_op.id() in op_ids_need_different_inputs
        ),
    ):
      # The candidate was inserted because type-matching connections were found,
      # so we move on with the recursion to insert other vertices.
      if insert_vertex_recursive(
          allowed_ops=allowed_ops,
          vertex_ids=vertex_ids,
          graph=graph,
          rng=rng,
          op_init_params=op_init_params,
          current_vertex_index=current_vertex_index + 1,
          op_ids_need_different_inputs=op_ids_need_different_inputs,
      ):
        # The recursion reached the maximum depth (all vertices were added) and
        # we are on our way out.
        return True
      else:
        # The candidate inserted with try_insert_new_connected proved to be a
        # bad idea and we are backtracking. Remove the vertex before the
        # for-loop moves on to the next candidate.
        graph.remove_vertex(vertex_id)
    else:
      # The candidate was not inserted because types did not match. Move on
      # to the next candidate.
      pass
  # We went through all the candidates and none of them worked out. We return
  # false as a signal to backtrack.
  return False


def try_insert_new_connected(
    candidate_op: graph_lib.Op,
    vertex_id: str,
    mark_required_input: bool,
    mark_required_output: bool,
    graph: graph_lib.Graph,
    rng: np.random.RandomState,
    op_init_params: Any,
    source_vertices: Optional[Sequence[graph_lib.Vertex]] = (),
    need_different_inputs: bool = False,
) -> bool:
  """Attempts to randomly insert a new vertex.

  If successful, all inputs of the vertex will be connected. If not successful,
  will do nothing.

  Args:
    candidate_op: the op that the new vertex should have. The op will determine
      the in-types that have to be satisfied by this method.
    vertex_id: the ID to give to the new vertex.
    mark_required_input: whether to mark this vertex as a required input. Can
      also be done manually separately.
    mark_required_output: whether to mark this vertex as a required output. Can
      also be done manually separately.
    graph: the graph in which to insert the vertex.
    rng: the random number generator to use.
      op_init_params: parameters to pass to the op initialization. Can be any
        container with all the relevant parameters (typically a proto or a
        dict). It is passed to the __init__ method of the ops as-is and it is
        never read elsewhere.
    source_vertices: If provided, we connect the new vertex to these source
      vertices (The underlying library checks type matching and may raise
      exceptions).
    need_different_inputs: If True, the op is required to be connected to
      different vertices.

  Returns:
    Whether successful.

  Raises:
    ValueError: If source_vertices is provided and its length mismatches the
      length of candidate_op's inputs.
  """
  if not source_vertices:
    src_vertices = []
    src_vertex_id_set = set()
    for in_type in candidate_op.in_types:
      candidate_src_vertices = []
      for v in graph.vertices.values():
        if v.op.out_type != in_type:
          continue
        if need_different_inputs and v.vertex_id in src_vertex_id_set:
          continue
        candidate_src_vertices.append(v)
      if not candidate_src_vertices:
        return False
      candidate_src_vertices.sort(key=lambda v: v.vertex_id)
      chosen_src_vertex = rng.choice(candidate_src_vertices)
      src_vertices.append(chosen_src_vertex)
      src_vertex_id_set.add(chosen_src_vertex.vertex_id)
  else:
    if len(source_vertices) != len(candidate_op.in_types):
      raise ValueError("Length mismatch between source vertices and op inputs.")
    src_vertices = source_vertices
  op = op_lib.Op.build_op(
      op_id=candidate_op.op_id, op_init_params=op_init_params
  )
  evolvable_params = None
  if op.has_evolvable_params:
    evolvable_params = op.create_evolvable_params(rng)
  graph.insert_vertex(
      vertex_id=vertex_id,
      op=op,
      evolvable_params=evolvable_params,
      mark_required_input=mark_required_input,
      mark_required_output=mark_required_output,
  )
  for in_index, src_vertex in enumerate(src_vertices):
    graph.insert_edge(
        src_vertex_id=src_vertex.vertex_id,
        dest_vertex_id=vertex_id,
        dest_vertex_in_index=in_index,
    )
  return True


def try_connect_dangling_inputs(
    vertex_id: str, graph: graph_lib.Graph, rng: np.random.RandomState
):
  """Attempts to randomly reconnect the inputs of the given vertex.

  Args:
    vertex_id: the ID of the vertex to reconnect. It is expected that this
      vertex has one or more "dangling" inputs (an in-edge that is not connected
      to a src-vertex).
    graph: the graph in which to insert the vertex.
    rng: the random number generator to use.

  Returns:
    Whether successful.

  Raises:
    KeyError: when a key is invalid.
  """
  if vertex_id not in graph.vertices:
    raise KeyError("Vertex ID not found in graph.")
  vertex = graph.vertices[vertex_id]

  # Src-vertex candidates are those vertices which would not form a loop.
  src_vertex_id_candidates = set(graph.vertices)
  src_vertex_id_candidates -= set(
      [v.vertex_id for v in forward_vertices(start_vertex=vertex, graph=graph)]
  )
  src_vertex_id_candidates -= set([vertex_id])
  src_vertex_id_candidates = sorted(list(src_vertex_id_candidates))

  # Here we'll temporarily store the inputs that matched each in-edge's type.
  src_vertex_ids = [None] * len(vertex.in_edge_ids)

  for in_index, in_edge in enumerate(vertex.in_edges):
    if in_edge is None:
      raise ValueError("in_edge is None")
    if in_edge.src_vertex is None:
      # This is a dangling edge.
      desired_type = vertex.op.in_types[in_index]
      rng.shuffle(src_vertex_id_candidates)
      success = False
      for src_candidate_id in src_vertex_id_candidates:  # Try all candidates.
        src_candidate = graph.vertices[src_candidate_id]
        if src_candidate.op.out_type == desired_type:
          # Found matching type.
          src_vertex_ids[in_index] = src_candidate_id
          success = True
      if not success:
        # Failed to find a src-vertex for this in-edge, so the whole process
        # must fail.
        return False
    else:
      # This is not a dangling edge. Nothing to do.
      # Note that this leaves a `None` entry in src_vertex_ids.
      pass

  # Found src-vertices for all in-edges. Connect them.
  for in_edge_id, src_vertex_id in zip(vertex.in_edge_ids, src_vertex_ids):
    if src_vertex_id is not None:  # If this was a dangling edge.
      graph.connect_src_vertex(src_vertex_id=src_vertex_id, edge_id=in_edge_id)
  return True


def try_reconnect_out_missing_vertex(
    vertex_id: str,
    graph: graph_lib.Graph,
    rng: np.random.RandomState,
    frozen_edge_ids: Optional[Set[str]] = None,
) -> bool:
  """Attempts to randomly reconnect vertex with missing out edge.

  Args:
    vertex_id: The vertex that potentially needs to be reconnected.
    graph: The graph that contains `vertex_id`.
    rng: The random number generator to use.
    frozen_edge_ids: A set of edges that do not mutate.

  Returns:
    True if the `vertex_id` is an OutputOp or already has out edge(s) or
    we successfully reconnect an edge for its output.

  Raises:
    KeyError: If the `vertex_id` is not found in `graph`.
  """

  if vertex_id not in graph.vertices:
    raise KeyError("Vertex ID not found in graph.")
  vertex = graph.vertices[vertex_id]

  if vertex.op.out_type is None or vertex.out_edge_ids:
    # The vertex is an output op (no out edge) or the vertex already has
    # out edges, we are done.
    return True

  edge_id_to_reconnect = None
  all_edge_ids = sorted(graph.edges.keys())
  rng.shuffle(all_edge_ids)
  for edge_id in all_edge_ids:
    if frozen_edge_ids is not None and edge_id in frozen_edge_ids:
      continue

    edge = graph.edges[edge_id]

    # Make sure the hero vertex's out_type can be connected to the edge's
    # dest_vertex.
    if edge.dest_type is None or edge.dest_type != vertex.op.out_type:
      continue

    # Make sure the edge's src_vertex has more than 1 out edges, so after
    # removing the edge, we will not creating new vertex with free out edges.
    if edge.src_vertex is not None and len(edge.src_vertex.out_edge_ids) <= 1:
      continue

    # Make sure if we connect hero vertex to the edge's dest_vertex, we do not
    # create a loop.
    if edge.dest_vertex is None:
      raise ValueError("edge.dest_vertex is None")
    if vertex_id == edge.dest_vertex.vertex_id:
      continue
    has_loop = False
    for v in forward_vertices(start_vertex=edge.dest_vertex, graph=graph):
      if vertex_id == v.vertex_id:
        has_loop = True
        break
    if has_loop:
      continue

    # Found an edge to reconnect
    edge_id_to_reconnect = edge_id
    break

  # Cannot find an edge to reconnect
  if edge_id_to_reconnect is None:
    return False

  graph.disconnect_src_vertex(edge_id_to_reconnect)
  graph.connect_src_vertex(
      src_vertex_id=vertex_id, edge_id=edge_id_to_reconnect
  )
  return True


def try_reconnect_all_out_missing_vertices(
    graph: graph_lib.Graph,
    rng: np.random.RandomState,
    frozen_edge_ids: Optional[Set[str]] = None,
) -> bool:
  """Attempts to reconnect all vertices with missing out edge.

  Args:
    graph: The graph.
    rng: The random number generator to use.
    frozen_edge_ids: A set of edges that do not mutate.

  Returns:
    True if successfully reconnect ALL vertices with missing out edge.
  """
  all_vertex_ids = sorted(graph.vertices.keys())
  rng.shuffle(all_vertex_ids)

  success = True
  for vertex_id in all_vertex_ids:
    if not try_reconnect_out_missing_vertex(
        vertex_id=vertex_id,
        graph=graph,
        rng=rng,
        frozen_edge_ids=frozen_edge_ids,
    ):
      success = False
  return success


def forward_vertices_ids(
    start_vertex: str,
    graph: graph_lib.Graph,
    cache: Optional[Dict[str, Set[str]]] = None,
) -> List[str]:
  """See `forward_vertices`, but uses str IDs instead of `Vertex` objects."""
  start_vertex_object = graph.vertices[start_vertex]
  if cache is None:
    cache_objects = None
  else:
    cache_objects = {
        k: {graph.vertices[vid] for vid in s} for k, s in cache.items()
    }
  vertex_objects = forward_vertices(
      start_vertex=start_vertex_object, graph=graph, cache=cache_objects
  )
  vertex_ids = [v.vertex_id for v in vertex_objects]
  if cache is not None:
    cache.clear()
    for k, s in cache_objects.items():
      cache[k] = {v.vertex_id for v in s}
  return vertex_ids


def forward_vertices(
    start_vertex: graph_lib.Vertex,
    graph: graph_lib.Graph,
    cache: Optional[Dict[str, Set[graph_lib.Vertex]]] = None,
) -> List[graph_lib.Vertex]:
  """Returns all the vertices that are downstream of the given vertex.

  Assumes the graph is a DAG. Does not include the start vertex.

  Args:
    start_vertex: the forward vertices from this vertex will be returned. This
      vertex is excluded.
    graph: the graph in which the start_vertex is.
    cache: a map from vertex ID to its forward vertices to be used as a cache to
      avoid recomputing partial forward vertex sets. Each value in the map is
      assumed not to include the vertex whose forward set represents. The cache
      may be full, partially full, or empty at the start; this function may add
      to it. If `None`, caching is not used.

  Returns:
    The list of forward vertices.
  """
  # Collect vertices through a DFS.
  vertices_found = forward_vertices_recursive(
      start_vertex=start_vertex, graph=graph, cache=cache
  )
  return list(vertices_found)


def forward_vertices_recursive(
    start_vertex: graph_lib.Vertex,
    graph: graph_lib.Graph,
    cache: Optional[Dict[str, Set[graph_lib.Vertex]]],
):
  """DFS helper to forward_vertices.

  Args:
    start_vertex: the current vertex in the DFS.
    graph: the graph in which we are searching.
    cache: see `forward_vertices` args.

  Returns:
    The forward set of `start_vertex`.

  Raises:
    RuntimeError: if a cycle is found.
  """
  vertices_found = set()
  for out_edge in start_vertex.out_edges:
    assert out_edge is not None
    if out_edge.dest_vertex is not None:
      dest_vertex = out_edge.dest_vertex
      vertices_found.add(dest_vertex)
      if cache is not None and dest_vertex.vertex_id in cache:
        more_vertices_found = cache[dest_vertex.vertex_id]
      else:
        more_vertices_found = forward_vertices_recursive(
            start_vertex=dest_vertex, graph=graph, cache=cache
        )
        if cache is not None:
          cache[dest_vertex.vertex_id] = more_vertices_found
      vertices_found = vertices_found.union(more_vertices_found)
  return vertices_found


def is_connected(
    graph: graph_lib.Graph, from_vertex_id: str, to_vertex_id: str
) -> bool:
  for vertex in forward_vertices(
      start_vertex=graph.vertices[from_vertex_id], graph=graph
  ):
    if vertex.vertex_id == to_vertex_id:
      return True
  return False


def get_forward_reachable_vertices(
    graph: graph_lib.Graph, start_vertices: Set[str]
) -> Set[str]:
  """Returns all the vertices that are downstream of the given set of vertices.

  Assumes the graph is a DAG.

  Args:
    graph: the graph in which the start_vertices are in.
    start_vertices: the forward vertices from these vertices will be returned.
      These vertices will be included.

  Returns:
    The list of forward vertices.
  """
  reachable_nodes: Set[str] = set()
  buffer = list(start_vertices)
  while buffer:
    current_vertex_id = buffer.pop(0)
    current_vertex = graph.vertices[current_vertex_id]
    for edge_id in current_vertex.out_edge_ids:
      if edge_id is None:
        continue
      edge = graph.edges[edge_id]
      dest_vertex_id = edge.dest_vertex_id
      if dest_vertex_id is None:
        raise ValueError("dest_vertex is None")
      if dest_vertex_id not in reachable_nodes:
        reachable_nodes.add(dest_vertex_id)
        buffer.append(dest_vertex_id)
  return reachable_nodes


def find_random_connected_subgraph(
    graph: graph_lib.Graph, max_size: int, rng: np.random.RandomState
) -> List[str]:
  """Finds a random connected subgraph.

  Finds a random connected subgraph that execludes input and output
  vertices.

  Args:
    graph: the graph in which we are searching.
    max_size: the maximum size of the generated subset.
    rng: the random number generator to use.

  Returns:
    A connected subset of vertices.
  """
  if max_size == 0:
    return []

  internal_vertex_ids = set(graph.vertices)
  internal_vertex_ids -= set(graph.required_inputs)
  internal_vertex_ids -= set(graph.required_outputs)
  current_vertex_id = rng.choice(list(internal_vertex_ids))
  current_vertex = graph.vertices[current_vertex_id]
  result = [current_vertex_id]
  # stores the vertices with in-edges coming from the returned component.
  out_vertices: Set[str] = set()
  # stores the vertices with out-edges going into the returned component.
  in_vertices: Set[str] = set()
  candidates = []
  max_size -= 1
  while max_size > 0:
    for edge_id in current_vertex.out_edge_ids:
      if edge_id is None:
        continue
      dest_vertex_id = graph.edges[edge_id].dest_vertex_id
      if dest_vertex_id is None:
        raise ValueError("dest_vertex_id is None")
      if dest_vertex_id not in result:
        out_vertices.add(dest_vertex_id)
      if not (
          dest_vertex_id in result
          or dest_vertex_id in candidates
          or dest_vertex_id in graph.required_outputs
      ):
        candidates.append(dest_vertex_id)
    for edge_id in current_vertex.in_edge_ids:
      if edge_id is None:
        continue
      src_vertex_id = graph.edges[edge_id].src_vertex_id
      if src_vertex_id is None:
        raise ValueError("src_vertex_id is None")
      if src_vertex_id not in result:
        in_vertices.add(src_vertex_id)
      if not (
          src_vertex_id in result
          or src_vertex_id in candidates
          or src_vertex_id in graph.required_inputs
      ):
        candidates.append(src_vertex_id)
    # Add a new vertex to result.
    if not candidates:
      # It is fine to return a subset of size < max_size.
      break
    while candidates:
      current_vertex_id = rng.choice(candidates)
      current_vertex = graph.vertices[current_vertex_id]
      candidates.remove(current_vertex_id)
      # Check if current_vertex is a valid choice.
      if _can_add_vertex_to_connected_subset(
          graph, current_vertex, result, out_vertices, in_vertices
      ):
        break
      current_vertex = None
    if not current_vertex:
      # It is fine to return a subset of size < max_size.
      break
    result.append(current_vertex_id)
    if current_vertex_id in out_vertices:
      out_vertices.remove(current_vertex_id)
    if current_vertex_id in in_vertices:
      in_vertices.remove(current_vertex_id)
    max_size -= 1

  return result


def _can_add_vertex_to_connected_subset(
    graph: graph_lib.Graph,
    vertex: graph_lib.Vertex,
    connected_subset: List[str],
    out_vertices: Set[str],
    in_vertices: Set[str],
) -> bool:
  """Helper method for find_random_connected_subgraph.

  Returns True if: the set of vertices with in-edges coming from the newly
  connected subset does not have any forward path to the set of set of vertices
  with out-edges going into the newly connected subset which does not pass
  through the new connected_subset (i.e. connected_subset union vertex).

  Args:
    graph: the graph in which we are searching.
    vertex: vertex to check.
    connected_subset: current connected subset.
    out_vertices: the vertices with in-edges coming from the returned component.
    in_vertices: the vertices with out-edges going into the connected component.

  Returns:
    True if: the set of vertices with in-edges coming from the newly connected
    subset does not have any forward path to the set of set of vertices with
    out-edges going into the newly connected subset which does not pass through
    the new connected_subset (i.e. connected_subset union vertex). Otherwsie,
    returns False.
  """
  # Work with clones as we will modify these sets.
  out_vertices_clone = out_vertices.copy()
  in_vertices_clone = in_vertices.copy()
  if vertex.vertex_id in out_vertices_clone:
    out_vertices_clone.remove(vertex.vertex_id)
  if vertex.vertex_id in in_vertices_clone:
    in_vertices_clone.remove(vertex.vertex_id)
  for edge_id in vertex.out_edge_ids:
    if edge_id is None:
      continue
    dest_vertex_id = graph.edges[edge_id].dest_vertex_id
    if dest_vertex_id in in_vertices:
      return False
    elif dest_vertex_id not in connected_subset:
      out_vertices_clone.add(dest_vertex_id)
  for edge_id in vertex.in_edge_ids:
    if edge_id is None:
      continue
    src_vertex_id = graph.edges[edge_id].src_vertex_id
    if src_vertex_id in out_vertices:
      return False
    elif src_vertex_id not in connected_subset:
      in_vertices_clone.add(src_vertex_id)

  # Make sure that every forward going path from out_vertices_clone to
  # in_vertices_clone passes through the connected_subset union vertex.
  graph_clone = graph.clone()
  graph_clone.remove_vertex(vertex.vertex_id)
  for vertex_id in connected_subset:
    graph_clone.remove_vertex(vertex_id)
  forward_reachable_vertices = get_forward_reachable_vertices(
      graph_clone, out_vertices_clone
  )
  for vertex_id in forward_reachable_vertices:
    if vertex_id in in_vertices_clone:
      return False
  return True


def reconnect_edges_recursive(
    edge_ids: List[str],
    graph: graph_lib.Graph,
    rng: np.random.RandomState,
    current_edge_index: int,
):
  """Recursive helper to randomly reconnect dangling edges.

  Will reconnect edges one at a time. For each reconnection, will match types
  and avoid loops. The function returns once all edges are reconnected.

  Args:
    edge_ids: the IDs of the edges to reconnect.
    graph: the graph where to insert.
    rng: the random number generator to use
    current_edge_index: the index in the list of edge_ids to reconnect. Used in
      the recursion to keep track of call stack depth. Set to 0 when calling
      this function from outside this function.

  Returns:
    Whether all the edges have been reconnected. I.e. whether the maximum
    recursion depth has been reached.
  """
  if current_edge_index == len(edge_ids):
    return True

  edge_id = edge_ids[current_edge_index]
  src_vertex_id_candidates = _get_src_vertex_id_candidates(edge_id, graph)
  assert src_vertex_id_candidates
  rng.shuffle(src_vertex_id_candidates)
  for src_vertex_id_candidate in src_vertex_id_candidates:
    graph.connect_src_vertex(
        src_vertex_id=src_vertex_id_candidate, edge_id=edge_id
    )
    if reconnect_edges_recursive(
        edge_ids=edge_ids,
        graph=graph,
        rng=rng,
        current_edge_index=current_edge_index + 1,
    ):
      # The recursion reached the maximum depth (all vertices were added) and
      # we are on our way out.
      return True
    else:
      # The candidate proved to be a bad one and we are backtracking. Disconnect
      # the candidate before the for-loop moves on to the next candidate.
      graph.disconnect_src_vertex(edge_id=edge_id)

  # None of the candidates worked out. Tell the recursive caller to backtrack.
  return False


def _get_src_vertex_id_candidates(
    edge_id: str, graph: graph_lib.Graph
) -> List[str]:
  """Finds possible source vertices for a given edge.

  Args:
    edge_id: the ID of the edge to reconnect.
    graph: the graph to which the edge belongs.

  Returns:
    A list of vertex IDs.

  Raises:
    KeyError: when a key is invalid.
  """
  if edge_id not in graph.edges:
    raise KeyError("Edge ID not found in graph.")
  edge = graph.edges[edge_id]

  # Src-vertex candidates are those vertices which would not form a loop.
  assert edge.dest_vertex is not None
  no_loop_candidate_ids = set(graph.vertices)
  dest_vertex_forward_set = forward_vertices(
      start_vertex=edge.dest_vertex, graph=graph
  )
  no_loop_candidate_ids -= set([v.vertex_id for v in dest_vertex_forward_set])
  no_loop_candidate_ids -= set([edge.dest_vertex_id])

  # Src-vertex candidates must match type.
  desired_type = edge.dest_vertex.op.in_types[edge.dest_vertex_in_index]
  match_type_candidates = []
  for src_candidate_id in no_loop_candidate_ids:
    src_candidate = graph.vertices[src_candidate_id]
    if src_candidate.op.out_type == desired_type:
      match_type_candidates.append(src_candidate_id)
  match_type_candidates.sort()

  return match_type_candidates


def contributing_subgraph_vertex_ids(
    graph: graph_lib.Graph,
    contributing_to_vertex_ids: List[str],
    ignore_edge_ids: List[str],
    required_inputs_behavior: str,
) -> Set[str]:
  """Computes a contributing subgraph.

  Args:
    graph: the graph in question.
    contributing_to_vertex_ids: the subgraph to be sized is that which
      contributes to these vertex IDs.
    ignore_edge_ids: the contribution through these edges will be ignored.
    required_inputs_behavior: how to consider required input vertices. If set to
      "consider", required inputs receive no special treatment. If set to
      "include", required inputs are counted regardless of whether they
      contribute. If set to "exclude", required inputs are not counted
      regardless of whether they contribute.

  Returns:
    A set of vertex IDs in the contributing subgraph, including the final
    vertex.
  """
  ignore_edge_ids = set(ignore_edge_ids)
  vertex_ids_seen = set()
  front = collections.deque()
  front.extend(contributing_to_vertex_ids)

  while front:
    vertex_id = front.popleft()
    vertex_ids_seen.add(vertex_id)
    vertex = graph.vertices[vertex_id]
    for edge in vertex.in_edges:
      if edge is None:
        raise ValueError("edge is None")
      if edge.edge_id in ignore_edge_ids:
        continue
      front.append(edge.src_vertex_id)

  if required_inputs_behavior == "consider":
    pass
  elif required_inputs_behavior == "include":
    for vertex_id in graph.required_inputs:
      vertex_ids_seen.add(vertex_id)
  elif required_inputs_behavior == "exclude":
    for vertex_id in graph.required_inputs:
      if vertex_id in vertex_ids_seen:
        vertex_ids_seen.remove(vertex_id)
  else:
    raise ValueError("Unknown required_inputs_behavior.")

  return vertex_ids_seen


def contributing_subgraph_size(
    graph: graph_lib.Graph,
    contributing_to_vertex_ids: List[str],
    ignore_edge_ids: List[str],
    required_inputs_behavior: str,
) -> int:
  """Computes a the number of vertices in a contributing subgraph.

  Args:
    graph: the graph in question.
    contributing_to_vertex_ids: the subgraph to be sized is that which
      contributes to these vertex IDs.
    ignore_edge_ids: the contribution through these edges will be ignored.
    required_inputs_behavior: how to consider required input vertices. If set to
      "consider", required inputs receive no special treatment. If set to
      "include", required inputs are counted regardless of whether they
      contribute. If set to "exclude", required inputs are not counted
      regardless of whether they contribute.

  Returns:
    The number of vertices in the contributing subgraph, including the final
    vertex.
  """
  return len(
      contributing_subgraph_vertex_ids(
          graph=graph,
          contributing_to_vertex_ids=contributing_to_vertex_ids,
          ignore_edge_ids=ignore_edge_ids,
          required_inputs_behavior=required_inputs_behavior,
      )
  )


def is_connected_graph(graph: graph_lib.Graph):
  """Whether this graph is connected.

  We define a graph to be connected if every point in the graph is connected to
  an output.

  Args:
    graph: the graph in question.

  Returns:
    True or False.
  """
  all_vertex_ids = {v.vertex_id for v in graph.vertices.values()}
  contributing_vertex_ids = contributing_subgraph_vertex_ids(
      graph=graph,
      contributing_to_vertex_ids=graph.required_outputs,
      ignore_edge_ids=[],
      required_inputs_behavior="include",
  )
  return len(all_vertex_ids) == len(contributing_vertex_ids)
