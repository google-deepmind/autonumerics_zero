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

"""Evaluation of graphs in the toy search space."""

from typing import Any, List, Optional, NamedTuple

from jax import numpy as jnp

from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import eval_metadata_pb2
from evolution.projects.graphs.joy import interpretation


ExecutionOutputs = interpretation.ExecutionOutputs
FingerprintProto = eval_metadata_pb2.FingerprintProto
JnpKey = data_lib.JnpKey
LearnableParams = graph_lib.LearnableParams


class SerializedEvalMetadata(NamedTuple):
  """Serialized version of the metadata proto."""

  permanent: bytes
  transient: bytes


class EvalMetadata(object):
  """The evaluation metadata.

  Attributes:
    init_jnp_key: the random key used to initialize the learnable params.
    train_jnp_key: the random key used to train the learnable params.
    valid_jnp_key: the random key used to validate the learnable params.
    finetune_jnp_key: the random key used to finetune the learnable params.
    learnable_params: the learnable params to include in the metadata.
    fingerprints: the map from vertex ID to jnp array fingerprint.
    parents_info: info about the parents of this individual.
    is_seed: whether this individual is a seed individual.
    is_clone: whether this individual is a clone of its parent. In this case,
      there should be exactly one parent.
    precision_fitness: the precision fitness or `None` if not measured.
      Typically, will have been measured if this individual is a clone.
    speed_fitness: the speed fitness or `None` if not measured. Typically, will
      have been measured if this individual is a clone.
    mutation_id: the id of the mutator applied to the parent of this individual.
      `None` if unknown or if a seed individual.
  """

  def __init__(self):
    """Deserializes the metadata."""
    self.init_jnp_key: Optional[JnpKey] = None
    self.train_jnp_key: Optional[JnpKey] = None
    self.valid_jnp_key: Optional[JnpKey] = None
    self.finetune_jnp_key: Optional[JnpKey] = None
    self.learnable_params: Optional[LearnableParams] = None
    self.fingerprints: ExecutionOutputs = {}
    self.parents_info: List[eval_metadata_pb2.ParentInfo] = []
    self.is_seed: Optional[bool] = None
    self.is_clone: Optional[bool] = None
    self.precision_fitness: Optional[float] = None
    self.speed_fitness: Optional[float] = None
    self.mutation_id: Optional[str] = None

  def as_proto(
      self, graph: Optional[graph_lib.Graph]
  ) -> eval_metadata_pb2.EvalMetadataProto:
    eval_metadata_proto = eval_metadata_pb2.EvalMetadataProto()
    self._serialize_learnable_params(eval_metadata_proto, graph)
    self._serialize_jnp_keys(eval_metadata_proto)
    self._serialize_fingerprints_without_fingerprints(eval_metadata_proto)
    self._serialize_parents_info(eval_metadata_proto)
    self._serialize_fitnesses(eval_metadata_proto)
    self._serialize_mutation_id(eval_metadata_proto)
    return eval_metadata_proto

  def from_proto(
      self,
      proto: eval_metadata_pb2.EvalMetadataProto,
      graph: Optional[graph_lib.Graph],
  ):
    self._deserialize_jnp_keys(proto)
    self._deserialize_learnable_params(proto, graph)
    self._deserialize_fingerprints(proto)
    self._deserialize_parents_info(proto)
    self._deserialize_fitnesses(proto)
    self._deserialize_mutation_id(proto)

  def serialize(
      self, graph: Optional[graph_lib.Graph]
  ) -> SerializedEvalMetadata:
    """Serializes this object into a bytestring.

    Args:
      graph: the graph this metadata belongs to.

    Returns:
      The serialized metadata.
    """
    eval_metadata_proto = self.as_proto(graph)

    eval_transient_metadata_proto = eval_metadata_pb2.EvalMetadataProto()
    self._serialize_learnable_params(eval_transient_metadata_proto, graph)
    self._serialize_jnp_keys(eval_transient_metadata_proto)
    self._serialize_fingerprints(eval_transient_metadata_proto)

    return SerializedEvalMetadata(
        eval_metadata_proto.SerializeToString(),
        eval_transient_metadata_proto.SerializeToString(),
    )

  def deserialize(
      self,
      serialized: Optional[bytes],
      serialized_transient: Optional[bytes],
      graph: Optional[graph_lib.Graph],
  ) -> None:
    """Deserializes the bytestring into this object.

    Args:
      serialized: the serialized metadata (used if transient metadata is none)
      serialized_transient: the serialized transient metadata (default choice,
        if available).
      graph: the graph this metadata belongs to.

    Returns:
      The EvalMetadataProto object
    """

    if serialized_transient is not None:
      return self.deserialize(serialized_transient, None, graph)

    if serialized is None:
      return

    proto = eval_metadata_pb2.EvalMetadataProto()
    proto.ParseFromString(serialized)
    self.from_proto(proto, graph)

  def _serialize_jnp_keys(self, proto: eval_metadata_pb2.EvalMetadataProto):
    """Helper to serialize. Adds this object's jnp keys to the proto."""
    if self.init_jnp_key is not None:
      assert self.init_jnp_key.shape == (2,)
      proto.init_jnp_key.extend(self.init_jnp_key.tolist())
    if self.train_jnp_key is not None:
      assert self.train_jnp_key.shape == (2,)
      proto.train_jnp_key.extend(self.train_jnp_key.tolist())
    if self.valid_jnp_key is not None:
      assert self.valid_jnp_key.shape == (2,)
      proto.valid_jnp_key.extend(self.valid_jnp_key.tolist())
    if self.finetune_jnp_key is not None:
      assert self.finetune_jnp_key.shape == (2,)
      proto.finetune_jnp_key.extend(self.finetune_jnp_key.tolist())

  def _serialize_learnable_params(
      self,
      proto: eval_metadata_pb2.EvalMetadataProto,
      graph: Optional[graph_lib.Graph],
  ):
    """Helper to serialize. Adds this object's params to the proto."""
    # Must distinguish `None` from `{}`.
    if self.learnable_params is None:
      proto.initialized_learnable_params = False
      return  # Nothing to serialize.
    else:
      assert graph is not None
      proto.initialized_learnable_params = True
      for vertex_id, vertex_learnable_params in self.learnable_params.items():
        op = graph.vertices[vertex_id].op
        assert op.has_learnable_params
        serialized_vertex_learnable_params = op.serialize_learnable_params(
            vertex_learnable_params
        )
        proto.learnable_params.append(
            eval_metadata_pb2.LearnableParamsProto(
                vertex_id=vertex_id,
                serialized=serialized_vertex_learnable_params,
            )
        )

  def _serialize_fingerprints_without_fingerprints(
      self, proto: eval_metadata_pb2.EvalMetadataProto
  ):
    """Helper to serialize. Adds this object's fingerprints to the proto."""
    if self.fingerprints:
      # We don't want the fingerprint list in Spanner - so clearing it.
      for vertex_id, fingerprint in self.fingerprints.items():
        dtype_spec = data_lib.get_dtype_spec(fingerprint.dtype)
        proto.fingerprints.append(
            FingerprintProto(
                vertex_id=vertex_id, dtype=dtype_spec, fingerprint=[]
            )
        )

  def _serialize_fingerprints(self, proto: eval_metadata_pb2.EvalMetadataProto):
    """Helper to serialize. Adds this object's fingerprints to the proto."""
    if self.fingerprints:
      for vertex_id, fingerprint in self.fingerprints.items():
        dtype_spec = data_lib.get_dtype_spec(fingerprint.dtype)
        proto.fingerprints.append(
            FingerprintProto(
                vertex_id=vertex_id,
                dtype=dtype_spec,
                fingerprint=fingerprint.tolist(),
            )
        )

  def _serialize_parents_info(self, proto: eval_metadata_pb2.EvalMetadataProto):
    """Helper to serialize. Adds this object's parents info to the proto."""
    for parent_info in self.parents_info:
      proto.parents_info.append(parent_info)
    value_or_none_to_proto(self.is_seed, proto=proto, field_name="is_seed")
    value_or_none_to_proto(self.is_clone, proto=proto, field_name="is_clone")

  def _serialize_fitnesses(self, proto: eval_metadata_pb2.EvalMetadataProto):
    """Helper to serialize. Adds this object's fitnesses to the proto."""
    if self.precision_fitness is not None:
      proto.precision_fitness = self.precision_fitness
    if self.speed_fitness is not None:
      proto.speed_fitness = self.speed_fitness

  def _serialize_mutation_id(self, proto: eval_metadata_pb2.EvalMetadataProto):
    """Helper to serialize. Adds this object's mutation ID to the proto."""
    if self.mutation_id is None:
      proto.ClearField("mutation_id")
    else:
      proto.mutation_id = self.mutation_id

  def _deserialize_jnp_keys(self, proto: eval_metadata_pb2.EvalMetadataProto):
    """Helper to deserialize. Extracts the proto's jnp keys into this object."""
    if proto.init_jnp_key:
      self.init_jnp_key = jnp.array(list(proto.init_jnp_key), dtype=jnp.uint32)
    else:
      self.init_jnp_key = None
    if proto.train_jnp_key:
      self.train_jnp_key = jnp.array(
          list(proto.train_jnp_key), dtype=jnp.uint32
      )
    else:
      self.train_jnp_key = None
    if proto.valid_jnp_key:
      self.valid_jnp_key = jnp.array(
          list(proto.valid_jnp_key), dtype=jnp.uint32
      )
    else:
      self.valid_jnp_key = None
    if proto.finetune_jnp_key:
      self.finetune_jnp_key = jnp.array(
          list(proto.finetune_jnp_key), dtype=jnp.uint32
      )
    else:
      self.finetune_jnp_key = None

  def _deserialize_learnable_params(
      self,
      proto: eval_metadata_pb2.EvalMetadataProto,
      graph: Optional[graph_lib.Graph],
  ):
    """Helper to deserialize. Extracts the proto's params into this object."""
    # Must distinguish `None` from `{}`.
    if proto.initialized_learnable_params:
      assert graph is not None
      self.learnable_params = {}
      for learnable_params_proto in proto.learnable_params:
        vertex_id = learnable_params_proto.vertex_id
        serialized_vertex_learnable_params = learnable_params_proto.serialized
        op = graph.vertices[vertex_id].op
        if not op.has_learnable_params:
          raise RuntimeError(
              "Missing learnable_params in op: %s." % op.op_id
              + "This is probably because the eval-metadata corresponds to the "
              "wrong graph, which can happen if you accumulate mutations when "
              "you shouldn't."
          )
        vertex_learnable_params = op.deserialize_learnable_params(
            serialized_vertex_learnable_params
        )
        self.learnable_params[vertex_id] = vertex_learnable_params
    else:
      self.learnable_params = None

  def _deserialize_fingerprints(
      self, proto: eval_metadata_pb2.EvalMetadataProto
  ):
    """Helper to deserialize. Extracts the proto's fingerprints."""
    self.fingerprints.clear()
    for fingerprint_proto in proto.fingerprints:
      if fingerprint_proto.vertex_id in self.fingerprints:
        raise RuntimeError("Duplicated fingerprint for vertex.")
      dtype = data_lib.get_dtype(fingerprint_proto.dtype)
      self.fingerprints[fingerprint_proto.vertex_id] = jnp.array(
          fingerprint_proto.fingerprint, dtype=dtype
      )

  def _deserialize_parents_info(
      self, proto: eval_metadata_pb2.EvalMetadataProto
  ):
    """Helper to deserialize. Extracts proto's parents info into this object."""
    self.parents_info = []
    for parent_info in proto.parents_info:
      self.parents_info.append(parent_info)
    self.is_seed = value_or_none_from_proto(proto, "is_seed")
    self.is_clone = value_or_none_from_proto(proto, "is_clone")

  def _deserialize_fitnesses(self, proto: eval_metadata_pb2.EvalMetadataProto):
    """Helper to deserialize. Extracts proto's fitnesses into this object."""
    if proto.HasField("precision_fitness"):
      self.precision_fitness = proto.precision_fitness
    else:
      self.precision_fitness = None
    if proto.HasField("speed_fitness"):
      self.speed_fitness = proto.speed_fitness
    else:
      self.speed_fitness = None

  def _deserialize_mutation_id(
      self, proto: eval_metadata_pb2.EvalMetadataProto
  ):
    """Helper to deserialize. Extracts proto's mutation ID into this object."""
    if proto.HasField("mutation_id"):
      self.mutation_id = proto.mutation_id
    else:
      self.mutation_id = None


def value_or_none_to_proto(value: Optional[Any], proto: Any, field_name: str):
  if value is None:
    proto.ClearField(field_name)
  else:
    setattr(proto, field_name, value)


def value_or_none_from_proto(proto: Any, field_name: str) -> Optional[Any]:
  if proto.HasField(field_name):
    return getattr(proto, field_name)
  else:
    return None
