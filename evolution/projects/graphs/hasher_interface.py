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

r"""Interface for hashers.

All hashers must derive from this base class. Hashers should be in the
appropriate subdirectory (e.g. graphs/auto_rl).
"""

from typing import Any, Callable, Optional, Tuple

from evolution.projects.graphs import graph as graph_lib

UNHASHABLE = b"unhashable"


class Hasher(object):
  """Base class for objects that hash graphs.

  Typically this will be used to generate FEC keys.
  """

  def hash(self, graph: graph_lib.Graph) -> bytes:
    """Performs the hashing.

    Args:
      graph: the graph to hash.

    Returns:
      The hash.
    """
    raise NotImplementedError("Must be implemented by the subclass.")


def hash_subgraph(
    graph: graph_lib.Graph,
    vertex: graph_lib.Vertex,
    distinguish_inputs: bool = True,
    sorting_function: Callable[[Any], Any] = lambda x: x,
    hashing_function: Optional[Callable[[Tuple[Any, ...]], Any]] = None,
    include_evolvable_parameters_in_hash: bool = False,
) -> Any:
  """Hash a subgraph of a DAG upstream from a vertex.

  Hashing will be carried out by recursively nesting tuples of the hash of the
  ops of a vertex and its children vertices' hashes. If a hashing_function is
  provided, it will be applied at every step of the recursion.
  A sorting function is required for sorting the hashes of the children,
  in order for the ordering to be unique.

  Args:
    graph: graph object, assumed to represent a DAG.
    vertex: a vertex belonging to the graph.
    distinguish_inputs: whether to distinguish each input node according to its
      position in graph.required_inputs.
    sorting_function: a sorting function to be applied to the children of a
      vertex.
    hashing_function: an optional function that hashes a nested tuple.
    include_evolvable_parameters_in_hash: Include the evolvable parameter
      information of the nodes in the hash.

  Returns:
    Either a nested tuple or the hashing_function applied to the nested tuple.
  """
  if include_evolvable_parameters_in_hash and vertex.op.has_evolvable_params:
    vertex_hash = (vertex.op.op_id, vertex.evolvable_params)
  else:
    vertex_hash = vertex.op.op_id
  if vertex.vertex_id in graph.required_inputs:
    if distinguish_inputs:
      hash_tuple = (vertex_hash, graph.required_inputs.index(vertex.vertex_id))
    else:
      hash_tuple = (vertex_hash, ())
  elif not vertex.in_edge_ids:
    hash_tuple = (vertex_hash, ())
  else:
    sub_hashes = []
    for group in vertex.op.input_index_groups():
      group_hashes = []
      for index in group:
        e_id = vertex.in_edge_ids[index]
        v_id = graph.edges[e_id].src_vertex_id
        v = graph.vertices[v_id]
        group_hashes += [
            hash_subgraph(
                graph,
                v,
                distinguish_inputs,
                sorting_function,
                hashing_function,
                include_evolvable_parameters_in_hash,
            )
        ]
      group_hashes = sorted(group_hashes, key=sorting_function)
      group_hashes = tuple(group_hashes)
      sub_hashes += [group_hashes]
    sub_hashes = tuple(sub_hashes)
    hash_tuple = (vertex_hash, sub_hashes)
  if hashing_function:
    return hashing_function(hash_tuple)
  else:
    return hash_tuple
