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

r"""Graph representation.

General notes:
-if it is sufficient to compare or hash Vertex IDs, do that instead of comparing
 Vertex instances. Same goes for Edges and Types.
"""

import heapq
import itertools
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import jax
import jax.numpy as jnp
import pydot

from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs import graph_spec_pb2
from evolution.projects.graphs import learnable_params as learnable_params_lib
from evolution.projects.graphs import op as op_lib
from evolution.projects.graphs import type as type_lib

T = type_lib.T
OptionalTensorShape = type_lib.OptionalTensorShape
Op = op_lib.Op
LearnableParams = learnable_params_lib.LearnableParams


class Vertex(object):
  """Represents a vertex in a graph.

  Public attributes:
    in_edge_ids: the list of input edges. A fixed size list (matching the op)
      with `None` where an input has not yet been set.
    out_edge_ids: the output edges as a set of edge IDs. This set must be empty
      if the op has no output.
  """

  def __init__(
      self,
      vertex_id: str,
      op: Op,
      evolvable_params: Optional[Any],
      graph: "Graph",
  ):
    """Constructs a vertex.

    Args:
      vertex_id: a unique nonempty string to identify this vertex.
      op: the op represented by this vertex.
      evolvable_params: a evolvable params object. Can be None if the op does
        not support evolvable params.
      graph: the containing Graph object.
    """
    check_id(vertex_id)
    self._vertex_id = vertex_id

    self._op = op
    self.evolvable_params = None
    if self._op.has_evolvable_params:
      self.evolvable_params = evolvable_params

    if not isinstance(graph, Graph):
      raise TypeError("Invalid graph.")
    self._graph = graph

    # All in-edges must have matching types in the op. There must be one in-edge
    # for each type and they must be in the same order. `None` means not set.
    self.in_edge_ids: List[Optional[str]] = [None] * len(self.op.in_types)

    # All out edges must be of the out_type of the op. There can be any number
    # of out edges, each representing a copy of the output of this op.
    self.out_edge_ids: Set[str] = set()

  def __eq__(self, other: "Vertex") -> bool:
    if other is None:
      return False
    if self._vertex_id != other.vertex_id:
      return False
    if self._op.op_id != other.op.op_id:
      return False
    if self.in_edge_ids != other.in_edge_ids:
      return False
    if sorted(self.out_edge_ids) != sorted(other.out_edge_ids):
      return False
    if self._op.has_evolvable_params or other.op.has_evolvable_params:
      assert self._op.has_evolvable_params
      assert other.op.has_evolvable_params
      if self.evolvable_params != other.evolvable_params:
        return False
    return True

  def __ne__(self, other: "Vertex") -> bool:
    return not self == other

  def __hash__(self) -> int:
    return hash((
        self.vertex_id,
        self.op.op_id,
        tuple(self.in_edge_ids),
        tuple(sorted(self.out_edge_ids)),
    ))

  @property
  def vertex_id(self) -> str:
    return self._vertex_id

  @property
  def op(self) -> Op:
    """The op applied by this vertex."""
    return self._op

  @property
  def in_edges(self) -> Tuple[Optional["Edge"], ...]:
    returning = []
    for edge_id in self.in_edge_ids:
      if edge_id is None:
        returning.append(None)
      else:
        returning.append(self._graph.edges[edge_id])
    return tuple(returning)

  @property
  def out_edges(self) -> Tuple["Edge", ...]:
    return tuple([self._graph.edges[eid] for eid in sorted(self.out_edge_ids)])

  def clone(self, to_graph: "Graph") -> "Vertex":
    """Clones this vertex into a new graph."""
    if not isinstance(to_graph, Graph):
      raise TypeError("Not a graph.")
    if to_graph is self._graph:
      raise ValueError("Must clone into a different graph.")
    cloned_evolvable_params = None
    if self._op.has_evolvable_params:
      cloned_evolvable_params = self._op.inherit_evolvable_params(
          self.evolvable_params
      )
    cloned = Vertex(
        vertex_id=self._vertex_id,
        op=self._op,
        evolvable_params=cloned_evolvable_params,
        graph=to_graph,
    )
    cloned.in_edge_ids = self.in_edge_ids[:]
    cloned.out_edge_ids = self.out_edge_ids.copy()  # pytype: disable=annotation-type-mismatch
    return cloned

  def rename(self, new_vertex_id: str):
    check_id(new_vertex_id)
    self._vertex_id = new_vertex_id

  def replace_op(self, op: Op, evolvable_params: Optional[Any]):
    """Replace the op assigned to this vertex.

    Args:
      op: the new op. Must be compatible with the old op in terms of the number
        of inputs, the input types, and the output type.
      evolvable_params: a evolvable params object. Can be None if the op does
        not support evolvable params.
    """
    if len(self._op.in_types) != len(op.in_types):
      raise ValueError("Mismatch in the number of inputs.")
    for old_in_type, new_in_type in zip(self._op.in_types, op.in_types):
      if old_in_type != new_in_type:
        raise ValueError("Mismatch in input type.")
    if self._op.out_type != op.out_type:
      raise ValueError("Mismatch in output type.")

    self._op = op
    self.evolvable_params = None
    if self._op.has_evolvable_params:
      self.evolvable_params = evolvable_params

  def validate(self, graph: "Graph"):
    """Checks whether this vertex is valid.

    In particular:
    -must have an op of type Op and it must be valid.
    -the number of in-edges/out-edges must be consistent with the op.
    -all in-edges must be assigned, be of Edge type, and be within the graph.
    -all out-edges must be assigned, be of Edge type, and be within the graph.

    Note that type-validation between two vertices connected by an edge is
    done in Edge.validate().

    Args:
      graph: the graph this vertex should be in.

    Raises:
      KeyError: if a key is invalid.
      TypeError: if a type is invalid.
      ValueError: if a value is invalid.
      RuntimeError: if something else is invalid.
    """
    if self._graph is not graph:
      raise RuntimeError("Inconsistent graph container.")
    if not isinstance(self._op, Op):
      raise TypeError("Found a non-op.")
    self._op.validate()
    if len(self.in_edge_ids) != len(self._op.in_types):
      raise RuntimeError("Vertex has invalid number of inputs.")
    if self._op.out_type is None and self.out_edge_ids:
      raise RuntimeError("Output vertex has outgoing edge.")
    for edge_id in itertools.chain(self.in_edge_ids, self.out_edge_ids):
      if edge_id is None:
        raise RuntimeError("Incomplete edge.")
      check_id(edge_id)
      if edge_id not in self._graph.edges:
        raise KeyError("Edge ID not in graph.")

  def debug_string(self, brief: bool = False) -> str:
    """Returns a representation of this vertex as readable string.

    Args:
      brief: if a brief string is desired. This is shorter and easier to read,
        but assumes the graph is consistent. If a consistency bug is being
        suspected, use False.
    """
    result = (
        "Vertex " + debug_string(self.vertex_id) + " (" + self.op.op_id + "):\n"
    )
    if not brief:
      result += "  in-types: " + type_ids_debug_string(self.op.in_types) + "\n"
    result += "  in-edges: " + iterable_debug_string(self.in_edge_ids) + "\n"
    if not brief:
      result += "  out-type: " + type_id_debug_string(self.op.out_type) + "\n"
      result += (
          "  out-edges: " + iterable_debug_string(self.out_edge_ids) + "\n"
      )
    return result


class Edge(object):
  """Represents an edge in a graph.

  Public attributes:
    src_vertex_id: the ID of the vertex this edge connects from.
    dest_vertex_id: the ID of the vertex this edge connects to.
    dest_vertex_in_index: what input number this is from the point of
      view of the dest_vertex. For example, if this is 0, this edge
      connects to the first input of the dest_vertex.
  """

  def __init__(self, edge_id: str, graph: "Graph"):
    """Constructs an edge.

    Args:
      edge_id: a unique nonempty string to identify this edge.
      graph: the containing Graph object.
    """
    check_id(edge_id)
    self._edge_id = edge_id
    if not isinstance(graph, Graph):
      raise TypeError("Invalid graph.")
    self._graph = graph
    self.src_vertex_id: Optional[str] = None  # None means "not set".
    self.dest_vertex_id: Optional[str] = None  # None means "not set".
    self.dest_vertex_in_index: Optional[int] = None  # None means "not set".

  def __eq__(self, other: "Edge") -> bool:
    if other is None:
      return False
    if self._edge_id != other.edge_id:
      return False
    if self.src_vertex_id != other.src_vertex_id:
      return False
    if self.dest_vertex_id != other.dest_vertex_id:
      return False
    if self.dest_vertex_in_index != other.dest_vertex_in_index:
      return False
    return True

  def __ne__(self, other: "Edge") -> bool:
    return not self == other

  def __hash__(self) -> int:
    return hash((
        self.edge_id,
        self.src_vertex_id,
        self.dest_vertex_id,
        self.dest_vertex_in_index,
    ))

  @property
  def edge_id(self) -> str:
    return self._edge_id

  @property
  def src_type(self) -> Optional[T]:
    """The type of the source or None if disconnected from source."""
    local_src_vertex = self.src_vertex
    if local_src_vertex is None:
      return None
    return local_src_vertex.op.out_type

  @property
  def dest_type(self) -> Optional[T]:
    """The type of the destination or None if disconnected from destination."""
    local_dest_vertex = self.dest_vertex
    if local_dest_vertex is None:
      return None
    return local_dest_vertex.op.in_types[self.dest_vertex_in_index]

  @property
  def type(self) -> T:
    """The type that flows through this edge."""
    local_src_type = self.src_type
    local_dest_type = self.dest_type
    if (
        local_src_type is not None
        and local_dest_type is not None
        and local_src_type.type_id != local_dest_type.type_id
    ):
      raise RuntimeError(
          "Inconsistent edge type: src type=%s, dest type=%s"
          % (local_src_type.type_id, local_dest_type.type_id)
      )
    if local_src_type is not None:
      return local_src_type
    elif local_dest_type is not None:
      return local_dest_type
    else:
      raise RuntimeError(
          "Edge type not known because edge is fully disconnected."
      )

  @property
  def src_vertex(self) -> Optional[Vertex]:
    if self.src_vertex_id is None:
      return None
    else:
      return self._graph.vertices[self.src_vertex_id]

  @property
  def dest_vertex(self) -> Optional[Vertex]:
    if self.dest_vertex_id is None:
      return None
    else:
      return self._graph.vertices[self.dest_vertex_id]

  def clone(self, to_graph: "Graph") -> "Edge":
    if not isinstance(to_graph, Graph):
      raise TypeError("Not a graph.")
    if to_graph is self._graph:
      raise ValueError("Must clone into a different graph.")
    cloned = Edge(self._edge_id, to_graph)
    cloned.src_vertex_id = self.src_vertex_id
    cloned.dest_vertex_id = self.dest_vertex_id
    cloned.dest_vertex_in_index = self.dest_vertex_in_index
    return cloned

  def validate(self, graph: "Graph"):
    """Checks whether this edge is valid.

    In particular:
    -the src-vertex must be assigned, of Vertex type, and be within the graph.
    -the dest-vertex must be assigned, of Vertex type, and be within the graph.
    -the edge must connect in a type-consistent manner.

    Args:
      graph: the graph this edge should be in.

    Raises:
      KeyError: if a key is invalid.
      TypeError: if a type is invalid.
      ValueError: if a value is invalid.
      RuntimeError: if something else is invalid.
    """
    if self._graph is not graph:
      raise RuntimeError("Inconsistent graph container.")
    for vertex_id in [self.src_vertex_id, self.dest_vertex_id]:
      if vertex_id is None:
        raise RuntimeError("Incomplete edge.")
      check_id(vertex_id)
      if vertex_id not in self._graph.vertices:
        raise KeyError("Vertex ID not in graph.")
    if self.dest_vertex_in_index is None:
      raise RuntimeError("Incomplete edge.")
    if not isinstance(self.dest_vertex_in_index, int):
      raise TypeError("Found a non-int index.")
    if self.src_vertex is None:
      raise RuntimeError("Incomplete graph.")
    else:
      edge_in_type = self.src_vertex.op.out_type  # pytype: disable=attribute-error  # always-use-property-annotation
    if self.dest_vertex is None:
      raise RuntimeError("Incomplete graph.")
    else:
      edge_out_type = self.dest_vertex.op.in_types[self.dest_vertex_in_index]  # pytype: disable=attribute-error  # always-use-property-annotation
    if edge_in_type != edge_out_type:
      raise RuntimeError("Inconsistent-type connection.")

  def debug_string(self) -> str:
    """Returns a representation of this edge as readable string."""
    result = "Edge: " + debug_string(self.edge_id) + ":" + "\n"
    result += "  src-vertex: " + debug_string(self.src_vertex_id) + "\n"
    result += "  dest-vertex: " + debug_string(self.dest_vertex_id) + "\n"
    result += (
        "  dest-vertex in-index: "
        + debug_string(self.dest_vertex_in_index)
        + "\n"
    )
    return result


class GraphMetadata:
  """Graph metadata base class.

  Stores graph / global level information.
  """

  def clone(self) -> "GraphMetadata":
    raise NotImplementedError

  def serialize(self) -> bytes:
    raise NotImplementedError

  def deserialize(self, serialized: bytes):
    raise NotImplementedError

  def __getitem__(self, key: str) -> Any:
    raise NotImplementedError

  def __setitem__(self, key: str, value: Any):
    raise NotImplementedError


class Graph(object):
  """Represents a graph.

  While internal attributes (e.g. `vertices`, `edges`) are accessible, it is
  recommended to manipulate a graph through the convenience methods (e.g.
  `insert_vertex`, `remove_edge`) because those will maintain internal
  consistency (if edge x is annotated as being an out-edge of vertex y, then
  vertex y will be annotated as having out-edge x too). Also, *when possible*,
  the convenience methods will avoid leaving dangling vertices or edges.

  Public attributes:
    vertices: a dictionary from vertex ID to Vertex.
    edges: a dictionary from edge ID to Edge. Note that by convention, the edge
      IDs follow a specific convention (see `get_complete_edge_id`). Without
      this convention a graph may not remain equal after it is serialized and
      deserialized. This is handled automatically if edges are created with
      `insert_edge`.
    required_inputs: a list with the required input vertex IDs, in a canonical
      order. These are vertices that are needed for evaluation and their
      types or order cannot change. Typically added by a Generator and left
      unchanged by Mutators and Recombinators.
    required_outputs: a list with the required output Vertex IDs, in a canonical
      order. These are vertices that are needed for evaluation and their types
      or order cannot change. Typically added by a Generator and left unchanged
      by Mutators and Recombinators.
  """

  def __init__(self):
    """Constructs the Graph instance."""
    self.vertices: Dict[str, Vertex] = {}  # Keyed by vertex ID.
    self.edges: Dict[str, Edge] = {}  # Keyed by edge ID.
    self.required_inputs: List[str] = []
    self.required_outputs: List[str] = []
    self.metadata: Optional[GraphMetadata] = None
    self._latest_vertex_id_suffix = None
    self._latest_edge_id_suffix = None

  def __eq__(self, other) -> bool:
    if other is None:
      return False
    if sorted(self.vertices) != sorted(other.vertices):
      return False
    for key in self.vertices:
      if self.vertices[key] != other.vertices[key]:
        return False
    if sorted(self.edges) != sorted(other.edges):
      return False
    for key in self.edges:
      if self.edges[key] != other.edges[key]:
        return False
    if self.required_inputs != other.required_inputs:
      return False
    if self.required_outputs != other.required_outputs:
      return False
    if (
        (self.metadata is None and other.metadata is not None)
        or (self.metadata is not None and other.metadata is None)
        or self.metadata != other.metadata
    ):
      return False
    return True

  def __hash__(self) -> int:
    vertices_signature = [v for _, v in sorted(self.vertices.items())]
    edges_signature = [e for _, e in sorted(self.edges.items())]
    return hash((
        tuple(vertices_signature),
        tuple(edges_signature),
        tuple(self.required_inputs),
        tuple(self.required_outputs),
    ))

  def clone(self, preserve_ids: bool = True) -> "Graph":
    """Creates a clone of this graph."""
    cloned = Graph()
    for vertex_id, vertex in self.vertices.items():
      cloned.vertices[vertex_id] = vertex.clone(to_graph=cloned)
    for edge_id, edge in self.edges.items():
      cloned.edges[edge_id] = edge.clone(to_graph=cloned)
    cloned.required_inputs = self.required_inputs[:]
    cloned.required_outputs = self.required_outputs[:]
    if not preserve_ids:
      self._simplify_edge_ids()
    # Access in next line is within class.
    cloned._latest_vertex_id_suffix = (  # pylint: disable=protected-access
        self._latest_vertex_id_suffix
    )
    cloned._latest_edge_id_suffix = (  # pylint: disable=protected-access
        self._latest_edge_id_suffix
    )

    if self.metadata is not None:
      cloned.metadata = self.metadata.clone()

    return cloned

  def as_proto(self) -> graph_spec_pb2.GraphProto:
    """Converts this graph to a serializable GraphProto."""
    graph_spec = graph_spec_pb2.GraphProto()
    for vertex in self.vertices.values():
      vertex_spec = graph_spec.vertices.add()
      vertex_spec.vertex_id = vertex.vertex_id
      vertex_spec.op_id = vertex.op.op_id
      if vertex.op.has_evolvable_params:
        vertex_spec.evolvable_params = vertex.op.serialize_evolvable_params(
            vertex.evolvable_params
        )
        assert vertex_spec.evolvable_params
      for in_edge in vertex.in_edges:
        if in_edge is None:
          raise RuntimeError("Incomplete vertex.")
        src_vertex = in_edge.src_vertex
        if src_vertex is None:
          raise RuntimeError("Incomplete edge.")
        vertex_spec.connections.add(
            in_edge_id=in_edge.edge_id, src_vertex_id=src_vertex.vertex_id
        )

    graph_spec.required_input_vertex_ids.extend(self.required_inputs)
    graph_spec.required_output_vertex_ids.extend(self.required_outputs)

    if self.metadata is not None:
      graph_spec.metadata = self.metadata.serialize()

    return graph_spec

  def from_proto(
      self, graph_spec: graph_spec_pb2.GraphProto, op_init_params: Any
  ):
    """Parses from proto."""
    self.vertices.clear()
    self.edges.clear()
    self.required_inputs.clear()
    self.required_outputs.clear()
    for vertex_spec in graph_spec.vertices:
      op = Op.build_op(op_id=vertex_spec.op_id, op_init_params=op_init_params)
      evolvable_params = None
      if op.has_evolvable_params:
        assert vertex_spec.HasField("evolvable_params")
        assert vertex_spec.evolvable_params
        evolvable_params = op.deserialize_evolvable_params(
            vertex_spec.evolvable_params
        )
      self.insert_vertex(
          vertex_spec.vertex_id,
          op=op,
          evolvable_params=evolvable_params,
          mark_required_input=False,
          mark_required_output=False,
      )
    for vertex_spec in graph_spec.vertices:
      for in_index, connection in enumerate(vertex_spec.connections):
        self.insert_edge(
            src_vertex_id=connection.src_vertex_id,
            dest_vertex_id=vertex_spec.vertex_id,
            dest_vertex_in_index=in_index,
            edge_id=connection.in_edge_id,
        )
    self.required_inputs.extend(graph_spec.required_input_vertex_ids)
    self.required_outputs.extend(graph_spec.required_output_vertex_ids)

    if graph_spec.HasField("metadata") and self.metadata:
      self.metadata.deserialize(graph_spec.metadata)
    else:
      self.metadata = None

  def serialize(self) -> bytes:
    """Serializes to a string."""
    # We don't allow serializing invalid graphs because assuming validity
    # allows a simpler serialized representation (see GraphSpec proto).
    self.validate()
    graph_spec = self.as_proto()
    return graph_spec.SerializeToString()

  def deserialize(self, serialized: bytes, op_init_params: Any):
    """Resets this graph with the result of deserializing the given proto."""
    graph_spec = graph_spec_pb2.GraphProto()
    graph_spec.ParseFromString(serialized)
    self.from_proto(graph_spec, op_init_params)
    self.validate()

  def parse(
      self,
      desired_vertices: List[Tuple[str, str]],
      desired_edges: List[Tuple[List[str], str]],
      required_input_vertex_ids: List[str],
      required_output_vertex_ids: List[str],
      op_init_params: Any,
      evolvable_params: Optional[Dict[str, Any]] = None,
      metadata: Optional[GraphMetadata] = None,
  ):
    """Resets this graph with the desired structure specified in the args.

    Args:
      desired_vertices: a list of (vertex id, op id) pairs.
      desired_edges: a list of ([in-vertex ids...], out vertex id) pairs, where
        [in_vertex_ids...] represents all the vertices that connect to the
        corresponding out-vertex ID. These in-vertex IDs must match the number
        of inputs and types of the op of the out-vertex ID. The edge IDs are
        automatically generated from the pair of vertex IDs.
      required_input_vertex_ids: a list of the IDs of the required input
        vertices.
      required_output_vertex_ids: a list of the IDs of the required output
        vertices.
      op_init_params: parameters to pass to the op initialization. Can be any
        container with all the relevant parameters (typically a proto or a
        dict). It is passed to the __init__ method of the ops as-is and it is
        never read elsewhere. Can be None if none of the ops needs params.
      evolvable_params: an optional list of key-value pairs where the key is the
        vertex ID and the value is the setting for that vertex's evolvable
        params.
      metadata: an optional metadata object attached to the graph.

    Raises:
      KeyError: if a key is missing.
    """
    self.vertices.clear()
    self.edges.clear()
    self.required_inputs.clear()
    self.required_outputs.clear()
    for vertex_id, op_id in desired_vertices:
      op = Op.build_op(op_id=op_id, op_init_params=op_init_params)
      vertex_evolvable_params = None
      if op.has_evolvable_params:
        if evolvable_params is None:
          raise ValueError(
              "Graph uses evolvable params but evolvable params not provided."
          )
        if vertex_id not in evolvable_params:
          raise KeyError(
              "Missing evolvable params for vertex ID %s" % vertex_id
          )
        vertex_evolvable_params = evolvable_params[vertex_id]
      self.insert_vertex(
          vertex_id=vertex_id,
          op=op,
          evolvable_params=vertex_evolvable_params,
          mark_required_input=False,
          mark_required_output=False,
      )
    for src_vertex_ids, dest_vertex_id in desired_edges:
      for in_index, src_vertex_id in enumerate(src_vertex_ids):
        self.insert_edge(
            src_vertex_id=src_vertex_id,
            dest_vertex_id=dest_vertex_id,
            dest_vertex_in_index=in_index,
        )
    self.required_inputs = required_input_vertex_ids[:]
    self.required_outputs = required_output_vertex_ids[:]
    self.metadata = metadata
    self.validate()

  def insert_vertex(
      self,
      vertex_id: str,
      op: op_lib.Op,
      evolvable_params: Optional[Any],
      mark_required_input: bool,
      mark_required_output: bool,
  ):
    """Inserts a new vertex into the graph.

    The inserted vertex will be totally disconnected. Therefore, this operation
    introduces new inconsistencies in the graph. To resolve the inconsistencies,
    all the inputs and/or outputs of the vertex must be connected as
    indicated by the op.

    NOTE: for unit tests, prefer building graphs using `parse` instead of
    `insert_vertex` and `insert_edge`.

    Args:
      vertex_id: a unique nonempty identifier string for the vertex.
      op: the op that this vertex represents.
      evolvable_params: the evolvable params object. Use `None` for ops that
        don't have evolvable params.
      mark_required_input: mark this vertex as a required input vertex. If so,
        will be added to the list of required inputs in the order it was
        inserted. It is therefore important to insert in order. Alternatively,
        can leave as False, and set the required_inputs manually.
      mark_required_output: mark this vertex as a required output vertex. If so,
        will be added to the list of required outputs in the order it was
        inserted. It is therefore important to insert in order. Alternatively,
        can leave as False, and set the required_outputs manually.

    Raises:
      KeyError: if the ID is invalid.
      ValueError: if a value is invalid.
    """
    if vertex_id in self.vertices:
      raise KeyError("Vertex ID already exists.")
    self.vertices[vertex_id] = Vertex(
        vertex_id=vertex_id,
        op=op,
        evolvable_params=evolvable_params,
        graph=self,
    )
    if mark_required_input and mark_required_output:
      raise ValueError("A vertex cannot be both an input and an output.")
    if mark_required_input:
      self.required_inputs.append(vertex_id)
    if mark_required_output:
      self.required_outputs.append(vertex_id)

  def insert_edge(
      self,
      src_vertex_id: str,
      dest_vertex_id: str,
      dest_vertex_in_index: int,
      edge_id: Optional[str] = None,
      override_type_check: bool = False,
  ):
    """Inserts a new edge into the graph.

    The edge is properly connected to the indicated src and dest vertices,
    including setting the connections on the vertices themselves. This
    operation does not introduce any new inconsistencies.

    An edge connecting the two vertices must not already exist.

    NOTE: for unit tests, prefer building graphs using `parse` instead of
    `insert_vertex` and `insert_edge`.

    Args:
      src_vertex_id: the ID of the in-vertex, which must already exist.
      dest_vertex_id: the ID of the out-vertex, which must already exist.
      dest_vertex_in_index: an index indicating to which input of the
        dest-vertex this edge should connect to. For example, if this is 0, the
        edge should connect to the zeroth input (recall the number of inputs is
        fixed by the op).
      edge_id: if not None, the edge ID to give to this edge. By default (None),
        it is automatically generated and guaranteed to be unique. If provided
        by the user, then the user must guarantee uniqueness, not just for this
        edge but for all the edges in the graph. Use with caution. Mainly
        intended for deserializing from a proto, where uniqueness was guaranteed
        by the serialization process.
      override_type_check: if True, it will be permitted to insert an edge that
        causes a type mismatch. Only set to True during unit tests.

    Raises:
      KeyError: if a key is invalid.
      IndexError: if an index is invalid.
      TypeError: if a type mismatch occurred.
    """
    if src_vertex_id not in self.vertices:
      raise KeyError("Unknown src-vertex ID: %s" % str(src_vertex_id))
    src_vertex = self.vertices[src_vertex_id]
    if dest_vertex_id not in self.vertices:
      raise KeyError("Unknown dest-vertex ID: %s" % str(dest_vertex_id))
    dest_vertex = self.vertices[dest_vertex_id]
    if dest_vertex.in_edge_ids[dest_vertex_in_index] is not None:
      raise ValueError("Dest-vertex input at given index is already set.")
    if not override_type_check:
      src_type = src_vertex.op.out_type
      assert src_type is not None
      dest_type = dest_vertex.op.in_types[dest_vertex_in_index]
      assert dest_type is not None
      if src_type != dest_type:
        raise TypeError("Edge insertion would result in a type mismatch.")
    if edge_id is None:
      edge_id = self.new_edge_id()
    assert edge_id not in self.edges
    edge = Edge(edge_id, self)
    edge.src_vertex_id = src_vertex_id
    edge.dest_vertex_id = dest_vertex_id
    edge.dest_vertex_in_index = dest_vertex_in_index
    src_vertex.out_edge_ids.add(edge_id)
    if dest_vertex_in_index >= len(dest_vertex.in_edge_ids):
      raise IndexError("Invalid vertex input index.")
    dest_vertex.in_edge_ids[dest_vertex_in_index] = edge_id
    self.edges[edge_id] = edge

  def remove_vertex(
      self,
      vertex_id: str,
      remove_in_edges: bool = True,
      remove_out_edges: bool = True,
      remove_input: bool = False,
  ):
    """Removes a vertex and its connections.

    By default, the vertex's in-edges and out-edges are removed as well and
    they are disconnected properly from the other edges. This op, however, does
    introduce an inconsistency since the dest-vertices of the removed out-edges
    will require reconnecting. Optionally, also the src-vertices of the in-edges
    can be reconnected. Note that it is also possible to disable edge removal
    through the args.

    Args:
      vertex_id: the ID of the vertex to remove. Must exist in the graph.
      remove_in_edges: whether to also remove the vertex's in-edges.
      remove_out_edges: whether to also remove the vertex's out-edges.
      remove_input: whether input vertex can also be removed.

    Raises:
      KeyError: if an ID is invalid.
      RuntimeError: if something else is invalid.
    """
    if vertex_id not in self.vertices:
      raise KeyError("Unknown vertex.")
    if not remove_input and vertex_id in self.required_inputs:
      raise RuntimeError("Cannot remove an input vertex.")
    if vertex_id in self.required_outputs:
      raise RuntimeError("Cannot remove an output vertex.")
    vertex = self.vertices[vertex_id]

    # Handle in-edges.
    for in_edge in vertex.in_edges:
      if in_edge is not None:
        if remove_in_edges:
          # Disconnect from src-vertex and delete edge.
          src_vertex = in_edge.src_vertex
          if src_vertex is not None:
            src_vertex.out_edge_ids.remove(in_edge.edge_id)
          del self.edges[in_edge.edge_id]
        else:
          # Disconnect from dest-vertex (the vertex being removed).
          in_edge.dest_vertex_id = None

    # Handle out-edges.
    for out_edge in vertex.out_edges:
      if out_edge is None:
        raise RuntimeError("Invalid out-edge.")
      if remove_out_edges:
        # Disconnect from dest-vertex and delete edge.
        dest_vertex = out_edge.dest_vertex
        if dest_vertex is not None:
          dest_vertex.in_edge_ids[out_edge.dest_vertex_in_index] = None
        del self.edges[out_edge.edge_id]
      else:
        # Disconnect from src-vertex (the vertex being removed).
        out_edge.src_vertex_id = None

    del self.vertices[vertex_id]

  def remove_edge(self, edge_id: str):
    """Removes an edge and disconnects from the src and dest vertices.

    Does not leave a dangling edge, but leaves the input to the dest vertex
    disconnected. Therefore, it leaves the graph in an invalid state until it
    is reconnected.

    Args:
      edge_id: the ID of the edge to remove. Must exist in the graph.

    Raises:
      KeyError: if an ID is invalid.
      RuntimeError: if something else is invalid.
    """
    if edge_id not in self.edges:
      raise KeyError("Unknown edge.")
    edge = self.edges[edge_id]
    if edge.src_vertex is not None:
      edge.src_vertex.out_edge_ids.remove(edge_id)  # pytype: disable=attribute-error  # always-use-property-annotation
    if edge.dest_vertex is not None:
      edge.dest_vertex.in_edge_ids[edge.dest_vertex_in_index] = None  # pytype: disable=attribute-error  # always-use-property-annotation
    del self.edges[edge_id]

  def connect_src_vertex(self, src_vertex_id: str, edge_id: str):
    """Connects the given vertex to edge as the src-vertex.

    The vertex and edge are assumed to be in the graph. The edge is assumed to
    be dangling (i.e. not have a src-vertex).

    Args:
      src_vertex_id: the ID of the source vertex.
      edge_id: the ID of the edge without a source vertex.

    Raises:
      KeyError: if a key is invalid.
    """
    if src_vertex_id not in self.vertices:
      raise KeyError("Invalid vertex ID.")
    src_vertex = self.vertices[src_vertex_id]
    if edge_id not in self.edges:
      raise KeyError("Invalid edge ID.")
    edge = self.edges[edge_id]
    if edge.src_vertex is not None:
      raise ValueError("Edge is not dangling.")
    assert edge_id not in src_vertex.out_edge_ids
    src_vertex.out_edge_ids.add(edge_id)
    edge.src_vertex_id = src_vertex_id

  def disconnect_src_vertex(self, edge_id: str):
    """Disconnects the src-vertex from the given edge.

    The result is that given edge will be left dangling (i.e. not have a
    src-vertex).

    Args:
      edge_id: the ID of the edge to disconnect.

    Raises:
      KeyError: if a key is invalid.
    """
    if edge_id not in self.edges:
      raise KeyError("Invalid edge ID.")
    edge = self.edges[edge_id]
    src_vertex = self.vertices[edge.src_vertex_id]
    src_vertex.out_edge_ids.remove(edge_id)
    edge.src_vertex_id = None

  def rename_vertex(self, old_vertex_id: str, new_vertex_id: str):
    """Renames a vertex in this graph."""
    if old_vertex_id not in self.vertices:
      raise ValueError("Old vertex ID does not exist.")
    if new_vertex_id in self.vertices:
      raise ValueError("New vertex ID already exists.")
    if new_vertex_id == old_vertex_id:
      return

    # Rename the vertex ID and vertices map key.
    vertex = self.vertices[old_vertex_id]
    del self.vertices[old_vertex_id]
    vertex.rename(new_vertex_id)
    self.vertices[new_vertex_id] = vertex

    # Rename in edges.
    for edge in self.edges.values():
      if edge.src_vertex_id == old_vertex_id:
        edge.src_vertex_id = new_vertex_id
      if edge.dest_vertex_id == old_vertex_id:
        edge.dest_vertex_id = new_vertex_id

    # Rename in inputs and outputs.
    for required_input_index in range(len(self.required_inputs)):
      if self.required_inputs[required_input_index] == old_vertex_id:
        self.required_inputs[required_input_index] = new_vertex_id
    for required_output_index in range(len(self.required_outputs)):
      if self.required_outputs[required_output_index] == old_vertex_id:
        self.required_outputs[required_output_index] = new_vertex_id

  def new_vertex_id(self) -> str:
    vertex_id = None
    while vertex_id is None or vertex_id in self.vertices:
      vertex_id = self._new_vertex_id_candidate()
    return vertex_id

  def _new_vertex_id_candidate(self) -> str:
    if self._latest_vertex_id_suffix is None:
      self._latest_vertex_id_suffix = 0
    else:
      self._latest_vertex_id_suffix += 1
    return "v%d" % self._latest_vertex_id_suffix

  def new_edge_id(self) -> str:
    edge_id = None
    while edge_id is None or edge_id in self.edges:
      edge_id = self._new_edge_id_candidate()
    return edge_id

  def _new_edge_id_candidate(self) -> str:
    if self._latest_edge_id_suffix is None:
      self._latest_edge_id_suffix = 0
    else:
      self._latest_edge_id_suffix += 1
    return "e%d" % self._latest_edge_id_suffix

  def validate(self):
    """Checks whether this graph is valid.

    In particular:
    -vertices/edges must be of the Vertex/Edge type.
    -the IDs used to key a vertex/edge must match the vertex/edge's ID.
    -IDs must not be empty.
    -all vertex/edges must pass their own validate() method.

    Raises:
      KeyError: if a key is invalid.
      TypeError: if a type is invalid.
      RuntimeError: if something else is invalid.
    """
    for vertex_id, vertex in self.vertices.items():
      check_id(vertex_id)
      if not isinstance(vertex, Vertex):
        raise TypeError("Found a non-vertex.")
      if vertex_id != vertex.vertex_id:
        raise RuntimeError("Inconsistent vertex ID.")
      vertex.validate(self)
    for edge_id, edge in self.edges.items():
      check_id(edge_id)
      if not isinstance(edge, Edge):
        raise TypeError("Found a non-edge.")
      if edge_id != edge.edge_id:
        raise RuntimeError("Inconsistent edge ID.")
      edge.validate(self)
    for vertex_id in itertools.chain(
        self.required_inputs, self.required_outputs
    ):
      if vertex_id not in self.vertices:
        raise KeyError("Vertex ID not found.")
    if not is_dag(self):
      raise RuntimeError("Not a DAG.")

  def validate_shapes(
      self, input_shape_dict: Dict[str, OptionalTensorShape]
  ) -> Tuple[Dict[str, OptionalTensorShape], bool]:
    """Validate if the given input shapes can populate correctly.

    Args:
      input_shape_dict: An example of expected input shapes for the graph.

    Returns:
      A tuple. The second entry represents if the input shapes can populate
      correctly (in order to produce all required outputs). The first entry
      represents all successfully populated shapes. (This dictionary will be
      partial if invalid shapes are detected during shape propagation.)
    """
    order = self.topological_sort()
    values = input_shape_dict.copy()

    for vertex_id in order:
      v = self.vertices[vertex_id]

      inputs = []
      if self._is_input_vertex(v):
        if vertex_id in values:
          inputs = [
              values[vertex_id],
          ]
        else:
          inputs = []

      for in_edge in v.in_edges:
        assert in_edge
        src_vertex = in_edge.src_vertex
        assert src_vertex
        assert src_vertex.vertex_id
        src_id = src_vertex.vertex_id
        inputs.append(values[src_id])

      evolvable_params = None
      if v.op.has_evolvable_params:
        evolvable_params = v.evolvable_params
      result, valid = v.op.output_shape(
          inputs, evolvable_params=evolvable_params
      )
      if valid:
        values.update({vertex_id: result})
      else:
        return (values, False)
    return (values, True)

  def debug_string(self, brief: bool = False) -> str:
    """Returns a representation of this graph as readable string.

    Args:
      brief: if a brief string is desired. This is shorter and easier to read,
        but assumes the graph is consistent. If a consistency bug is being
        suspected, use False.
    """
    result = ""
    result += (
        "Required input vertices: "
        + iterable_debug_string(self.required_inputs)
        + "\n"
    )
    for _, vertex in sorted(self.vertices.items()):
      result += vertex.debug_string(brief)
    result += (
        "Required output vertices: "
        + iterable_debug_string(self.required_outputs)
        + "\n"
    )
    if not brief:
      for _, edge in sorted(self.edges.items()):
        result += edge.debug_string()
    return result

  def get_contributing_vertices(self) -> Set[str]:
    """Produce set of vertices that contribute to the output.

    Returns:
      contributing_vertices: Set of vertices that are reachable from the
        output vertices in a reverse traversal.
    """
    contributing_vertices = set(self.required_outputs)
    buffer = []
    for vertex_id in self.required_outputs:
      buffer.append(self.vertices[vertex_id])
    while buffer:
      vertex = buffer.pop(0)
      for e_id in vertex.in_edge_ids:
        src_vertex_id = self.edges[e_id].src_vertex_id
        assert src_vertex_id
        src_vertex = self.vertices[src_vertex_id]
        if src_vertex_id not in contributing_vertices:
          contributing_vertices.add(src_vertex_id)
          buffer.append(src_vertex)
    return contributing_vertices

  def topological_sort(self, contributing_only: bool = False) -> List[str]:
    """See `topological_vertex_id_sort`."""
    return self.topological_vertex_id_sort(contributing_only=contributing_only)

  def topological_vertex_id_sort(
      self, contributing_only: bool = False
  ) -> List[str]:
    """Sorts verteces by topological order and by IDs.

    The order is primarily topological and secondarily alphabetical (on the
    vertex IDs). Only contributing vertices are included.

    Args:
      contributing_only: Specifies if non-contributing (junk) vertices should be
        filtered out.

    Returns:
      A list of vertex IDs.

    Raises:
      RuntimeError: ...
    """
    # Holds the IDs of vertices that cannot be computed because they are missing
    # inputs.
    uncomputable_vertex_ids = set()

    # Holds IDs of edges that have been computed.
    computed_edge_ids = set()

    # Holds the IDs of vertices that are computable (i.e. all its in-edges have
    # been computed). The PriorityQueue produces IDs alphabetically so they
    # can be computed in that order.
    computable_vertex_ids_heap = []

    # Holds the computed vertices sorted in the desired order.
    # The algorithm will progress by moving the vertices from
    # `uncomputable_vertex_ids` into `computable_vertex_ids` as they are ready
    # to compute, and then out of `computable_vertex_ids` into
    # `computed_vertex_ids`.
    computed_vertex_ids = []

    # Initially, input vertices are computable.
    for vertex in self.vertices.values():
      if vertex.in_edges:
        uncomputable_vertex_ids.add(vertex.vertex_id)
      else:
        heapq.heappush(computable_vertex_ids_heap, vertex.vertex_id)

    while computable_vertex_ids_heap:
      # Mark this vertex as computed.
      vertex = self.vertices[heapq.heappop(computable_vertex_ids_heap)]
      computed_vertex_ids.append(vertex.vertex_id)

      for out_edge in vertex.out_edges:
        # Mark out_edge as computed.
        assert out_edge.edge_id not in computed_edge_ids
        computed_edge_ids.add(out_edge.edge_id)

        # Check if dest_vertex is computable.
        dest_vertex = self.vertices[out_edge.dest_vertex_id]
        assert dest_vertex.vertex_id in uncomputable_vertex_ids
        if _topological_vertex_id_sort_is_computable_helper(
            dest_vertex, computed_edge_ids
        ):
          uncomputable_vertex_ids.remove(dest_vertex.vertex_id)
          heapq.heappush(computable_vertex_ids_heap, dest_vertex.vertex_id)

    # Check if we are done.
    if uncomputable_vertex_ids:
      raise RuntimeError("Not a DAG.")

    # Only return the contributing vertices.
    if contributing_only:
      contributing_vertex_ids = self.get_contributing_vertices()
      computed_vertex_ids = [
          v for v in computed_vertex_ids if v in contributing_vertex_ids
      ]
    return computed_vertex_ids

  def execute(
      self,
      input_dict: Dict[str, Any],
      learnable_params: Optional[LearnableParams] = None,
      jax_rng_key: Optional[jnp.ndarray] = None,
      contributing_only: bool = True,
      **op_exec_kwargs,
  ) -> Dict[str, Any]:
    """Executes the graph.

    This is done by calling vertices' ops in topological order.

    Args:
      input_dict: Dictionary keyed to input vertices' ids, with data matching
        their vertices' Ops output types.
      learnable_params: a dictionary (keyed by vertex ID) containing the
        learnable params for each vertex. Can be `None` if none of the ops has
        learnable params.
      jax_rng_key: Jax random key for doing random operations such as sampling.
      contributing_only: Boolean specifying whether or not to filter out
        vertices that do not contribute to the output.
      **op_exec_kwargs: Keyword arguments passed to the op.

    Returns:
      output_dict: a dict from vertex ID to the output of that vertex.

    Raises:
      ValueError: if invalid value.
      RuntimeError: if something else.
    """
    if not self._validate_input_dict(input_dict):
      raise ValueError(
          f"Input dict is incompatible with root vertices! {input_dict}"
      )
    # Feed input dict values into root (Produce) vertices.
    values = input_dict.copy()

    # Get topological order on which to execute Op vertices.
    order = self.topological_sort(contributing_only=contributing_only)

    # Traverse graph according to topological order.
    for vertex_id in order:
      v = self.vertices[vertex_id]

      inputs = []
      if self._is_input_vertex(v):
        if vertex_id in values:
          inputs = [
              values[vertex_id],
          ]
        else:
          inputs = []

      for in_edge in v.in_edges:
        assert in_edge
        assert in_edge.src_vertex
        src_id = in_edge.src_vertex.vertex_id
        inputs.append(values[src_id])

      evolvable_params = None
      if v.op.has_evolvable_params:
        evolvable_params = v.evolvable_params
      vertex_learnable_params = None
      if v.op.has_learnable_params:
        assert isinstance(learnable_params, dict)
        vertex_learnable_params = learnable_params[vertex_id]
      jax_rng_subkey = None
      if v.op.needs_jax_rng_key:
        assert isinstance(jax_rng_key, jnp.ndarray)
        jax_rng_key, jax_rng_subkey = jax.random.split(jax_rng_key)
      result = v.op.execute(
          inputs=inputs,
          vertex_id=vertex_id,
          evolvable_params=evolvable_params,
          learnable_params=vertex_learnable_params,
          jax_rng_key=jax_rng_subkey,
          **op_exec_kwargs,
      )
      if result is None:
        raise RuntimeError("Op execution must return a result.")
      values.update({vertex_id: result})

    if not self._validate_output_dict(values):
      raise ValueError(
          "Output dictionary did not match expectations set by"
          f"leaf vertices! {values}"
      )
    return values

  def check_path_between_vertices(
      self, src_vertex_id: str, dest_vertex_id: str
  ) -> bool:
    """Check if there exists a path from `src_vertex_id` to `dest_vertex_id`.

    Args:
      src_vertex_id: Source vertex id.
      dest_vertex_id: Destination vertex id.

    Returns:
      If path exists, returns True; otherwise returns False.

    Raises:
      KeyError: if any of the vertex ids is not in the graph.
    """
    if src_vertex_id not in self.vertices:
      raise KeyError("src_vertex_id is not in the graph.")
    if dest_vertex_id not in self.vertices:
      raise KeyError("dest_vertex_id is not in the graph.")
    if src_vertex_id == dest_vertex_id:
      return True

    src_vertex = self.vertices[src_vertex_id]
    for edge_id in src_vertex.out_edge_ids:
      edge = self.edges[edge_id]
      if self.check_path_between_vertices(edge.dest_vertex_id, dest_vertex_id):
        return True
    return False

  def prune(self, learnable_params: Optional[LearnableParams] = None):
    """Simplifies the current graph by removing non-contributing vertices.

    Non-contributing (or "junk") vertices are defined as those that do not
    translate into function; i.e. those that do not affect the output.

    Args:
      learnable_params: the learnable params for the graph. The keys in this
        dict will be pruned together with the vertices
    """
    contributing_vertices = self.get_contributing_vertices()
    vertex_ids = list(self.vertices.keys())
    for vertex_id in vertex_ids:
      if (
          vertex_id not in contributing_vertices
          and vertex_id not in self.required_inputs
      ):
        assert vertex_id not in self.required_outputs
        self.remove_vertex(
            vertex_id=vertex_id, remove_in_edges=True, remove_out_edges=True
        )
        if learnable_params is not None:
          del learnable_params[vertex_id]

  def _is_input_vertex(self, vertex: Vertex):
    return not vertex.op.in_types

  def _validate_input_dict(self, input_dict: Dict[str, Any]) -> bool:
    if any([v not in input_dict.keys() for v in self.required_inputs]):
      raise ValueError(
          f"Root vertices {self.required_inputs} not all present in "
          f"input dict {input_dict.keys()}"
      )
    return True

  def _validate_output_dict(self, output_dict: Dict[str, Any]) -> bool:
    """Ensures required output keys are present in `output_dict`."""
    if any([v not in output_dict.keys() for v in self.required_outputs]):
      raise ValueError(
          f"Leaf vertices {self.required_outputs} not all present in "
          f"output dict {output_dict.keys()}"
      )
    return True

  def _simplify_edge_ids(self):
    """Gives new, simpler IDs to all the edges.

    Graph equality is determined by using IDs, so be careful if doing
    comparisons with other graphs. Two graphs that are equal may not remain
    equal if one has its IDs simplified. Requires that the graph be valid.
    """
    self.validate()
    self._latest_edge_id_suffix = None

    old_to_new_ids = {}
    for old_edge_id in sorted(self.edges):
      new_edge_id = self.new_edge_id()
      old_to_new_ids[old_edge_id] = self.new_edge_id()

    edges = list(self.edges.values())
    self.edges.clear()
    for edge in edges:
      old_edge_id = edge.edge_id
      new_edge_id = old_to_new_ids[old_edge_id]
      edge._edge_id = new_edge_id  # pylint: disable=protected-access
      self.edges[new_edge_id] = edge

    for vertex in self.vertices.values():
      for in_index in range(len(vertex.in_edge_ids)):
        in_edge_id = vertex.in_edge_ids[in_index]
        vertex.in_edge_ids[in_index] = old_to_new_ids[in_edge_id]
      out_edge_ids = list(vertex.out_edge_ids)
      vertex.out_edge_ids.clear()
      for out_edge_id in out_edge_ids:
        vertex.out_edge_ids.add(old_to_new_ids[out_edge_id])

    self.validate()

  def plot(
      self,
      contributing_only: bool = False,
      no_constants: bool = False,
      subgraphs_ids: Optional[List[List[str]]] = None,
  ) -> pydot.Dot:
    """Visualize DAG with pydot.

    Args:
      contributing_only: see `Graph.execute`.
      no_constants: removes constant vertices (i.e. yellow vertices).
      subgraphs_ids: list of vertices lists in a subgraph to highlight.

    Returns:
      Dot object of DAG.
    """
    pydot_graph = pydot.Dot(graph_type="digraph", rankdir="TB")
    graph_op_dict = {}
    order = self.topological_vertex_id_sort(contributing_only=contributing_only)
    interacting_vertex_ids_set = self.interacting_vertex_ids()

    def get_fill_color(vertex: Vertex):
      color = "white"
      if subgraphs_ids:
        colors = ["orange", "red", "blue", "pink", "purple"]
        for idx, subgraph_ids in enumerate(subgraphs_ids):
          if vertex.vertex_id in subgraph_ids and idx < len(colors):
            color = colors[idx]
            break
      if vertex.vertex_id in self.required_inputs:
        color = "red"
      elif vertex.vertex_id in self.required_outputs:
        color = "green"
      elif vertex.vertex_id not in interacting_vertex_ids_set:
        color = "yellow"
      return color

    def get_vertex_name(vertex: Vertex):
      name = "%s: %s" % (vertex.vertex_id, vertex.op.id())
      evolvable_params = None
      if v.op.has_evolvable_params:
        evolvable_params = v.evolvable_params
      digest_str = vertex.op.digest_str(evolvable_params, vertex=vertex)
      if digest_str:
        name += "_" + digest_str
      return name

    # Traverse graph according to topological order.
    for vertex_id in order:
      v = self.vertices[vertex_id]
      is_constant_vertex = vertex_id not in interacting_vertex_ids_set

      vertex_out_edges = [self.edges[edge_id] for edge_id in v.out_edge_ids]
      # All out-edges point to constant vertices.
      all_out_edges_constant = all(
          edge.dest_vertex_id not in interacting_vertex_ids_set
          for edge in vertex_out_edges
      )
      if no_constants and is_constant_vertex and all_out_edges_constant:
        continue

      if no_constants and is_constant_vertex:
        label = "..."
        style = ""
        shape = "none"
      else:
        label = get_vertex_name(v)
        style = "filled"
        shape = ""

      node = pydot.Node(
          vertex_id,
          label=label,
          style=style,
          shape=shape,
          fillcolor=get_fill_color(v),
      )

      pydot_graph.add_node(node)
      graph_op_dict[vertex_id] = node

      if no_constants and is_constant_vertex:
        continue

      vertex_in_edges = [self.edges[edge_id] for edge_id in v.in_edge_ids]
      for edge in vertex_in_edges:
        color = "black"
        label = edge.dest_vertex_in_index
        pydot_graph.add_edge(
            pydot.Edge(
                graph_op_dict[edge.src_vertex_id],
                node,
                color=color,
                label=label,
            )
        )

    return pydot_graph

  def visualize(
      self,
      contributing_only: bool = False,
      no_constants: bool = False,
      subgraph_ids: Optional[List[str]] = None,
  ):
    """Visualize DAG with pydot.

    Args:
      contributing_only: Boolean specifying whether or not to filter out
        vertices that do not contribute to the output.
      no_constants: removes constant vertices (i.e. yellow vertices).
      subgraph_ids: Optional list of vertices in a subgraph to highlight.

    Returns:
      Dot object of DAG.
    """
    if not subgraph_ids:
      return self.plot(contributing_only, no_constants)
    return self.plot(contributing_only, no_constants, [subgraph_ids])

  def to_math_expression(
      self,
      op_map: Dict[str, str],
      output_vertex_id: str,
      learnable_params: Optional[LearnableParams] = None,
  ):
    """Converts graph to math expression that can be copy-pasted to mathematica.

    WARNING: The string can be very huge because all recursive references are
      fully expanded out.

    Args:
      op_map: Mapping from vertex operation ID to the math expression. E.g.
        "ProduceMinusOneOp" : "-1" Special symbols: 1. Arguments. For operations
        that accept arguments, you can use {a0}, {a1}, ... to refer to the
        arguments. E.g. If we have. "MultOp": "({a0})*({a1})", and a0's math
        expression is 1 and a1's math expression is 2, then we will get "(1 *
        2)". 2. Vertex id. E.g. "ProduceIntegerOp" : "{vertex_id}" means that
        different vertices that have the same ProduceIntegerOp will be mapped to
        the vertex IDs. 3. Evolvable params. E.g. "ConstantOp" : "
        {evolvable_params:.8f} means that vertices with the ConstantOp are
        assumed to have a single float as evolvable_params and will have the
        math expression obtained by printing it to 8 decimal places. 4.
        Learnable params. E.g. "ZeroInitVariableOp" : " {learnable_params:.8f}
        means that vertices with the ZeroInitVariableOp are assumed to have a
        single float as learnable_params and will have the math expression
        obtained by printing it to 8 decimal places.
      output_vertex_id: The vertex to get the fully expanded expression of.
      learnable_params: maps vertex ID to learned value for ops with learnable
        params. Note that these are not specified in the graph because they are
        obtained at evaluation time. Must include all vertices with ops that use
        the "{learnable_param}" symbol in the op_map arg. If there are no such
        vertices or no such ops, this can be left as `None`.

    Returns:
      the math expression of `output_vertex_id`

    Raises:
      ValueError: if an argument is invalid.
    """
    if output_vertex_id not in self.vertices:
      raise ValueError(f"vertex_id {output_vertex_id} is not in the graph.")
    order = self.topological_vertex_id_sort(contributing_only=True)

    # Vertex id to the fully expanded expression (string)
    v_to_exp = {}
    # Traverse graph according to topological order.
    for vertex_id in order:
      v = self.vertices[vertex_id]
      op_name = v.op.id()

      if op_name not in op_map:
        raise ValueError(f"op {op_name} is not specified in the op_map.")

      # Prepare substitution map
      args_dict = {}
      for i in range(len(v.in_edge_ids)):
        arg_v = self.edges[v.in_edge_ids[i]].src_vertex_id
        assert arg_v in v_to_exp
        args_dict[f"a{i}"] = v_to_exp[arg_v]
      args_dict["vertex_id"] = vertex_id
      if v.op.has_evolvable_params:
        args_dict["evolvable_params"] = v.evolvable_params
      if v.op.has_learnable_params:
        args_dict["learnable_params"] = learnable_params[vertex_id]

      # Compute the fully expanded expression
      v_to_exp[vertex_id] = op_map[op_name].format(**args_dict)

      if vertex_id == output_vertex_id:
        return v_to_exp[vertex_id]
    raise ValueError(
        f"output_vertex_id {output_vertex_id} not encountered in "
        "topological order. Is it a junk node?"
    )

  def interacting_vertex_ids(self) -> Set[str]:
    """Classify vertices as interacting or non-interacting.

    A vertex is interacting if it directly or indirectly operates on the input.

    Returns:
      The set of IDs of interacting vertices.
    """
    vertex_ids = self.topological_vertex_id_sort(contributing_only=True)
    interacting: Dict[str, bool] = {}  # Vertex ID to whether it's interacting.
    for vertex_id in vertex_ids:
      vertex = self.vertices[vertex_id]
      if _is_input_vertex(vertex):
        interacting[vertex_id] = _is_input_vertex_interacting(vertex)
      else:
        interacting[vertex_id] = _is_noninput_vertex_interacting(
            vertex=vertex, interacting=interacting
        )
    return {v for v in vertex_ids if interacting[v]}


def get_k_largest_isomorphic_subgraphs(
    graph_a: Graph, graph_b: Graph, k: int = 1
) -> List[Dict[str, str]]:
  """Brute-force retrieval of matching subgraphs between two graphs.

  Starts with a list of all possible vertex matches (based on the vertex's
  Op ids).

  For each candidate subgraph:
    1) Iterate over every unexplored vertex A from graph a, and corresponding
    unexplored vertex A' from graph B:
    2) Expand the edges of vertex A, and check each destination vertex for
    a corresponding destination vertex in the list of outgoing edges from A'.
    3) If a correspondence exists, add this new <vertex: corresponding vertex>
    pair to the unexplored vertices of this potential subgraph, as well as the
    correspondences of this subgraph.
    4) If no further unexplored vertices exist for this potential subgraph,
    add it to the list of completed correspondences.
  When there are no more unexplored / unevaluated correspondences, sort the
  list of completed correspondences by size and return the top k elements.


  Args:
     graph_a: First graph object for comparison.
     graph_b: Second graph object for comparison.
     k: Maximum number of subgraphs to return.

  Returns:
    correspondences: List of dictionaries representing the vertex ids of
      vertices found in the largest subgraph.
  """

  class _Candidate:
    """Convenient class for encapsulating candidate correspondence data."""

    def __init__(self, root: Vertex, root_correspondence: Vertex):
      self.root = root
      self.num_vertices = 1
      self.correspondences: Dict[str, str] = {
          root.vertex_id: root_correspondence.vertex_id
      }
      self.frontier: List[str] = [
          root.vertex_id,
      ]

  candidates: List[_Candidate] = []
  completed_candidates: List[_Candidate] = []

  sorted_vertices_a = [
      graph_a.vertices[v]
      for v in graph_a.topological_vertex_id_sort(contributing_only=True)
  ]
  sorted_vertices_b = [
      graph_b.vertices[v]
      for v in graph_b.topological_vertex_id_sort(contributing_only=True)
  ]

  # Initialize correspondences between graph A and graph B.
  for v in sorted_vertices_a:
    for w in sorted_vertices_b:
      # TODO(aarontp): Should we add IO / additional checks here for candidates?
      if v.op.id() == w.op.id():
        candidates.append(_Candidate(v, w))

  # Iterate on a set of candidate subgraphs, maintaining and growing a set of
  # frontier vertices and evaluating edges from the frontier vertices for
  # correspondences.
  while candidates:
    candidate = candidates.pop(0)

    # Look for correspondences in the vertices connected to the frontier
    # vertices, if possible. If the frontier is empty, add the candidate to
    # the 'completed_candidates' list.
    frontier = candidate.frontier
    new_frontier = []
    for vertex_id in frontier:
      vertex = graph_a.vertices[vertex_id]
      corresponding_vertex = graph_b.vertices[
          candidate.correspondences[vertex_id]
      ]

      for e in vertex.out_edges:
        # TODO(aarontp): Are there edge cases where we can accidentally match
        # the wrong vertex to two "identical looking" vertices through this
        # method?

        # TODO(aarontp): How do we handle cases where two vertices point to the
        # same node and it corresponds? Or cases where one vertex' edge
        # corresponds to the right Op but another node pointing to the same
        # destination vertex doesn't correspond?
        dest_vertex = e.dest_vertex
        assert dest_vertex
        if dest_vertex not in sorted_vertices_a:
          continue
        for corresponding_edge in corresponding_vertex.out_edges:
          corresponding_dest_vertex = corresponding_edge.dest_vertex
          if corresponding_dest_vertex not in sorted_vertices_b:
            continue
          assert corresponding_dest_vertex
          if dest_vertex.op.id() == corresponding_dest_vertex.op.id():
            candidate.correspondences[dest_vertex.vertex_id] = (
                corresponding_dest_vertex.vertex_id
            )
            new_frontier.append(dest_vertex.vertex_id)

    if not new_frontier:
      completed_candidates.append(candidate)
    else:
      candidate.frontier = new_frontier[:]
      candidates.append(candidate)

  # Sort candidate subgraphs by size.
  completed_candidates.sort(key=lambda c: -len(c.correspondences))
  return [
      c.correspondences
      for c in completed_candidates[: min(len(completed_candidates), k)]
  ]


def check_id(element_id):
  if not element_id:
    raise ValueError("Invalid ID.")


def debug_string(element: Optional[Any]) -> str:
  return "None" if element is None else str(element)


def type_id_debug_string(t: Optional[T]) -> str:
  return "None" if t is None else t.type_id


def iterable_debug_string(elements: Iterable[Optional[Any]]) -> str:
  element_strings = [debug_string(e) for e in elements]
  return "[" + ", ".join(element_strings) + "]"


def type_ids_debug_string(ts: Iterable[Optional[T]]) -> str:
  type_ids = [type_id_debug_string(t) for t in ts]
  return "[" + ", ".join(type_ids) + "]"


def is_dag(graph: Graph):
  """Returns whether the graph is a DAG."""
  temp_graph = graph.clone()
  start_vertices: List[Vertex] = [
      v for v in temp_graph.vertices.values() if not v.in_edge_ids
  ]
  while start_vertices:
    vertex = start_vertices.pop()
    for out_edge in list(vertex.out_edges):
      assert out_edge is not None
      dest_vertex = out_edge.dest_vertex
      assert dest_vertex is not None
      temp_graph.remove_edge(out_edge.edge_id)
      if not has_connected_in_edges(dest_vertex):
        start_vertices.append(dest_vertex)
  return not temp_graph.edges


def has_connected_in_edges(vertex: Vertex) -> bool:
  """Returns whether the vertex has any connected in-edges."""
  for edge in vertex.in_edges:
    if edge is not None:
      return True
  return False


def _is_input_vertex(vertex: Vertex) -> bool:
  """Whether this vertex is an input vertex."""
  return not vertex.op.in_types


def _is_input_vertex_interacting(vertex: Vertex) -> bool:
  """Whether an input vertex interacts with x."""
  # The only way this can happen is if the input vertex is x itself.
  return vertex.op.op_id == "ProduceXOp"


def _is_noninput_vertex_interacting(
    vertex: Vertex, interacting: Dict[str, bool]
) -> bool:
  """Whether a non-input vertex interacts with x."""
  # This happens if any of its src vertices are interacting.
  assert vertex.in_edges
  for edge in vertex.in_edges:
    assert edge
    if interacting[edge.src_vertex_id]:
      return True
  return False


def _topological_vertex_id_sort_is_computable_helper(
    vertex: Vertex, computed_edge_ids: Set[str]
) -> bool:
  """Helper function. Returns whether vertex is computable."""
  for in_edge_id in vertex.in_edge_ids:
    if in_edge_id not in computed_edge_ids:
      return False
  return True
