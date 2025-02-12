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

from typing import Any, Dict, List, Optional, Tuple, Type
from jax import numpy as jnp
import numpy as np
from evolution.projects.graphs import learnable_params as learnable_params_lib
from evolution.projects.graphs import type as type_lib

T = type_lib.T
OptionalTensorShape = type_lib.OptionalTensorShape
LearnableParams = learnable_params_lib.LearnableParams


class Op(object):
  """Base class for all ops.

  An Op instance is an abstract representation for an operation. An operation
  has specific input and output types. For example, an "add" operation has two
  float inputs and one float output.

  Ops can support evolvable parameters by setting the `cls_has_evolvable_params`
  attribute and implementing the create_evolvable_params,
  mutate_evolvable_params, inherit_evolvable_params, serialize_evolvable_params,
  and deserialize_evolvable_params methods. These methods control how the
  parameters are handled by the search process. The values of the parameters
  themselves are stored in the `params` field of the vertex, so that different
  appearances of the op in the graph can have different values.

  Ops can support learnable parameters by setting the `cls_has_learnable_params`
  attribute and implementing the init_learnable_params,
  serialize_learnable_params, and deserialize_learnable_params method.
  """

  in_types: Tuple[T, ...]
  out_type: Optional[T]

  # The type of the params for this op. All params-related methods will be
  # expected to comply. If `None`, this op does not support/need params.
  cls_has_evolvable_params: bool = False

  # Marks an op that introduces learnable params into the evaluation. These can
  # be initialized from the evolvable params or can be independent.
  cls_has_learnable_params: bool = False

  # Whether or not this op needs a jax random key for executing.
  cls_needs_jax_rng_key: bool = False

  # Only write to this from `__init_subclass__`.
  subclasses: Dict[str, Type["Op"]] = {}

  @classmethod
  def __init_subclass__(cls, final_cls=None):
    """Registers final subclasses.

    Args:
      final_cls: the final subclass to be registered.
    """
    super().__init_subclass__()
    if final_cls is not None:
      # We are defining a final class (i.e. an actual op in the search space, as
      # opposed to an intermediate sublcass like `InputOp`).

      # Register the class.
      if final_cls.__name__ in Op.subclasses:
        raise TypeError(
            "Found two Ops subclasses with the same name. Did you import a "
            "search space from another project?"
        )
      Op.subclasses[final_cls.__name__] = final_cls

  @classmethod
  def build_op(cls, op_id: str, op_init_params: Any) -> "Op":
    """Builds the op.

    Args:
      op_id: the ID of the op to build.
      op_init_params: an object containing information for initializing the ops.
        This object will be passed to *all* ops. Each op chooses what to read.
        There are no constraints on what this object is, but typically it would
        be a proto or a dict.

    Returns:
      The op instance.
    """
    if op_id not in cls.subclasses:
      raise ValueError(
          f"Op ID not registered: {op_id}. Did you import the search space?"
      )
    return cls.subclasses[op_id](op_init_params=op_init_params)

  @classmethod
  def id(cls) -> str:
    """Returns an id for this class.

    Note this returns the short class name without package names.
    """
    return cls.__name__

  def __new__(cls, *args, **kwargs):
    """Prevents from instantiation without subclassing."""
    del args
    del kwargs

    if cls is Op:
      raise TypeError(
          "Only subclasses of Op can be used as types, not Op itself."
      )
    return Op.__new__(cls)

  def __eq__(self, other: "Op"):
    if other is None:
      return False
    if self.op_id != other.op_id:
      return False
    if [t.type_id for t in self.in_types] != [
        t.type_id for t in other.in_types
    ]:
      return False
    if self.out_type != other.out_type:
      return False
    return True

  @property
  def op_id(self) -> str:
    """Returns the op spec enum corresponding to this op, for serialization."""
    return self.__class__.id()

  @property
  def has_evolvable_params(self) -> bool:
    """Returns whether this Op has params."""
    return self.__class__.cls_has_evolvable_params

  @property
  def has_learnable_params(self) -> bool:
    """Returns whether this Op has params."""
    return self.__class__.cls_has_learnable_params

  @property
  def needs_jax_rng_key(self) -> bool:
    """Returns whether this Op needs a jax random key."""
    return self.__class__.cls_needs_jax_rng_key

  def validate(self):
    """Checks whether this op is valid.

    In particular:
    -must have a valid ID.
    -types must be of the T type.
    -must either have inputs or outputs.

    Raises:
      TypeError: if a type is invalid.
      RuntimeError: if anything else is invalid.
    """
    check_id(self.op_id)
    num_connections = 0
    for in_type in self.in_types:
      if not isinstance(in_type, T):
        raise TypeError("Found a non-T in_type.")
      num_connections += 1
    if self.out_type is not None:
      if not isinstance(self.out_type, T):
        raise TypeError("Found a non-T out_type.")
      num_connections += 1
    if num_connections == 0:
      raise RuntimeError("Op has no connections.")

  def create_evolvable_params(self, rng: np.random.RandomState) -> Any:
    """Initializes the evolvable params.

    This method is called when the vertex applying this op is created for the
    first time. For example, this happens at generation time or when the vertex
    in question is inserted into the graph by a mutation. If the graph is being
    cloned and the vertex already exists in the parent, this method is not
    called.

    Args:
      rng: a random number generator.

    Returns:
      The evolvable params object. Each Op class can choose a type for this
      object.
    """
    raise NotImplementedError(
        "Must be implemented by subclass for all subclasses that "
        "support evolvable params."
    )

  def inherit_evolvable_params(self, evolvable_params: Any) -> Any:
    """Clones the evolvable params.

    This method is called when a parent is cloned to produce a child. If the
    mutation does not affect this vertex, then this is the only change the
    evolvable params will experience (often this method will implement the
    identity, so there may be no change at all). However, the mutation may edit
    this vertex by deleting it (so this vertex's evolvable params will
    disappear) or by mutating its evolvable params (so `mutate_evolvable_params`
    will be called after this method).

    Args:
      evolvable_params: the evolvable params object to clone. Each Op class can
        choose a type for this object. Do not modify this object within this
        function. Instead, make a copy and return that.

    Returns:
      The cloned params object.
    """
    raise NotImplementedError(
        "Must be implemented by subclass for all subclasses that "
        "support evolvable params."
    )

  def mutate_evolvable_params(
      self, evolvable_params: Any, impact: float, rng: np.random.RandomState
  ) -> Any:
    """Mutates the evolvable params.

    This method is called when a EvolvableParamsMutator chooses this vertex.
    It is not called if this vertex is inserted by a mutator or if this vertex
    is not modified by the mutation at all.

    Args:
      evolvable_params: the evolvable params object. Each Op class can choose a
        type for this object. Do not modify this object within this function.
        Instead, make a copy and return that.
      impact: a non-negative number indicating the magnitude of the effect of
        this mutation. Different Op classes may interpret this value
        differently.
      rng: a random number generator.

    Returns:
      The mutated evolvable params object. If not identical to the
      `evolvable_params` arg, make a deep copy and edit that.
    """
    raise NotImplementedError(
        "Must be implemented by subclass for all subclasses that "
        "support evolvable params."
    )

  def update_evolvable_params(
      self, evolvable_params: Any, learnable_params: Any
  ) -> Optional[Any]:
    """Chooses between Darwinian or Lamarckian inheritance.

    This method is called after evaluation to update the evolvable params.
    That is, this method receives the evolvable params and returns them, but
    gets a chance to modify them by leaking learnt information from the
    learnable params.

    By default, this method returns the `evolvable_params` unchanged, resulting
    in Darwinian inheritance.

    The other extreme, in which this method is overridden to return the
    `learnable_params` instead, results in Lamarckian inheritance. Of course,
    this should be used only if it makes sense to the op in question. That is,
    the op must have both evolvable and learnable params and they must be of
    the same format and meaning.

    Args:
      evolvable_params: the evolvable params object to update. Each Op class can
        choose a type for this object. It is ok to return the unmodified
        `evolvable_params`. It is also ok to modify the `evolvable_params` in
        place, with the goal of returning them or otherwise.
      learnable_params: the learnable params object to potentially use to inform
        the update of the evolvable params.

    Returns:
      The updated evolvable params object (or an updated copy).
    """
    del learnable_params  # Darwinian evolution.
    return evolvable_params

  def serialize_evolvable_params(self, evolvable_params: Any) -> bytes:
    """Serializes the evolvable params.

    This method must be the inverse of deserialize_evolvable_params.

    Args:
      evolvable_params: the evolvable params object to clone. Each Op class can
        choose a type for this object. Do not modify this object within this
        function.

    Returns:
      The evolvable params serialized as bytes.
    """
    raise NotImplementedError(
        "Must be implemented by subclass for all subclasses that "
        "support evolvable params."
    )

  def deserialize_evolvable_params(self, serialized: bytes) -> Any:
    """Deserializes the evolvable params.

    This method must be the inverse of serialized_evolvable_params.

    Args:
      serialized: the serialized evolvable params.

    Returns:
      The evolvable params object. Each Op class can choose a type for this
      object.
    """
    raise NotImplementedError(
        "Must be implemented by subclass for all subclasses that "
        "support evolvable params."
    )

  def digest_str(self, evolvable_params: Any, **kwargs) -> str:
    """Returns a digest string of the op.

    Args:
      evolvable_params: the evolvable params object.
      **kwargs: other arguments passed from e.g. graph.plot()

    Returns:
      A digest string to summarize params & evolvable params of the op (
      mainly for graph visualization).
    """
    del evolvable_params, kwargs
    return ""

  def init_learnable_params(
      self, evolvable_params: Any, jnp_key: jnp.ndarray, hashing_round: int
  ) -> LearnableParams:
    """Initializes the learnable params.

    Args:
      evolvable_params: the vertex's evolvable params.
      jnp_key: the key to use to generate random numbers.
      hashing_round: must be >= 0 if the initialization is for the purposes of
        hashing and must be set to -1 if it is for the purposes of evaluation.
        This is because some operations (e.g. see `ZeroInitVariableOp`) require
        different inintialization paths to avoid hash collisions. If set to
        True, it is assumed that `jnp_key` will be kept consistent across
        individuals within an experiment. Therefore, the initialization code
        inside this function can use random numbers, even if hashing. In fact,
        it is recommended that initialization for hashing use the jnp_key so
        that hashers can produce more accurate hashes by passing multiple keys.

    Returns:
      The learnable params object.
    """
    raise NotImplementedError(
        "Must be implemented by subclass for all subclasses that "
        "support learnable params."
    )

  def serialize_learnable_params(self, learnable_params: Any) -> bytes:
    """Serializes the evolvable params.

    This method must be the inverse of deserialize_learnable_params.

    Args:
      learnable_params: the learnable params for a given vertex. The type is
        determined by the search space. Do not modify this object within this
        function. Note: this is a single value of the LearnableParams dict.

    Returns:
      The learnable params object serialized as bytes.
    """
    raise NotImplementedError(
        "Must be implemented by subclass for all subclasses that "
        "support learnable params."
    )

  def deserialize_learnable_params(self, serialized: bytes) -> Any:
    """Deserializes the learnable params.

    This method must be the inverse of serialized_learnable_params.

    Args:
      serialized: the serialized learnable params object.

    Returns:
      The learnable params for a given vertex. The type is determined by the
      search space. Note: this is a single value of the LearnableParams dict.
    """
    raise NotImplementedError(
        "Must be implemented by subclass for all subclasses that "
        "support learnable params."
    )

  def execute(self, inputs: List[Any], **exec_kwargs) -> Any:
    """Abstract data processing method. Must conform to Ops type specs."""
    raise NotImplementedError("All Op subclasses must implement call logic!")

  def output_shape(
      self, input_shapes: List[OptionalTensorShape], **kwargs
  ) -> Tuple[OptionalTensorShape, bool]:
    # Alternatively let everything pass through by default.
    # return (None, True)
    # TODO(b/195374797): Raise error only if the output type is considered
    # shaped.
    raise NotImplementedError(
        f"{self}: All Op subclasses must implement output_shape method!"
    )

  def input_index_groups(self) -> List[List[int]]:
    """Returns groups of input indices that are interchangeable.

    For example, for a commutative binary operation with in_types = [T, T],
    there is one input index group, that both indices belong to, i.e., [[0, 1]].
    On the other hand, for a non-commutative binary operation,
    the two indices 0 and 1 should belong to separate groups, i.e., [[0], [1]].
    This method can be called by hashing functions for a graph.
    """
    if not self.in_types:
      return []
    elif len(self.in_types) == 1:
      return [[0]]
    else:
      raise NotImplementedError(
          f"{self}: input_index_groups method has not been implemented "
          "for this operation, which takes more than one input."
      )


class InputOp(Op):
  """Base class for all input ops.

  An `InputOp` requires no input edges; its sole purpose is to produce a value
  consumed by other vertices.

  In execution, the op takes a singleton input that represents the
  value of this op, and the value is passed through as the output.
  """

  in_types: Tuple[T, ...] = ()
  out_type: Optional[T] = None  # Subclass must override this.

  @classmethod
  def __init_subclass__(cls):
    super().__init_subclass__(final_cls=cls)
    if cls.out_type is None:
      raise ValueError(f"Must specify out_type for {cls}")

  def __new__(cls, *args, **kwargs):
    """Prevents from instantiation without subclassing."""
    del args
    del kwargs

    if cls is InputOp:
      raise TypeError(
          "Only subclasses of InputOp can be used as types, not InputOp itself."
      )
    return object.__new__(cls)

  def __init__(self, *args, **kwargs):
    del args, kwargs

  def execute(self, inputs, **exec_kwargs):
    assert len(inputs) == 1
    return inputs[0]

  def output_shape(
      self, input_shapes: List[OptionalTensorShape], **kwargs
  ) -> Tuple[OptionalTensorShape, bool]:
    # By default, just pass on the input shape, which could be None, or defined.
    # Certain inputs, e.g. neural net parameter, may choose to use its output
    # layer's dimension as a shape hint, even though the parameter object itself
    # isn't of TensorT type.
    return (input_shapes[0], True)


class TransformOp(Op):
  """Base class for all transform ops.

  A `TransformOp` is a (pure) function that maps inputs to an output.
  """

  in_types: Tuple[T, ...] = ()
  out_type: Optional[T] = None

  @classmethod
  def __init_subclass__(cls, final_cls=None, is_abstract=False):
    del final_cls
    super().__init_subclass__(final_cls=cls)
    if is_abstract:
      return
    if not cls.in_types:
      raise ValueError(f"Must specify non-empty in_types for ${cls}")
    if cls.out_type is None:
      raise ValueError(f"Must specify out_type for ${cls}")

  def __new__(cls, *args, **kwargs):
    """Prevents from instantiation without subclassing."""
    del args
    del kwargs

    if cls is TransformOp:
      raise TypeError(
          "Only subclasses of TransformOp can be used as types, "
          "not TransformOp itself."
      )
    return object.__new__(cls)

  def __init__(self, *args, **kwargs):
    del args, kwargs

  def execute(self, inputs, **exec_kwargs):
    raise NotImplementedError(
        "This method must be implemented by the subclass."
    )

  def output_shape(
      self, input_shapes: List[OptionalTensorShape], **kwargs
  ) -> Tuple[OptionalTensorShape, bool]:
    raise NotImplementedError(
        f"{self.__class__} must implement output_shape logic."
    )


class BroadcastableTransformOp(TransformOp, is_abstract=True):
  """Abstract class for all binary transforms that may broadcast.

  Assumes (and asserts) exactly two inputs and their shapes are defined.
  """

  def __init__(self, *args, **kwargs):
    super(BroadcastableTransformOp, self).__init__(*args, **kwargs)
    del args, kwargs

  def output_shape(
      self, input_shapes: List[OptionalTensorShape], **kwargs
  ) -> Tuple[OptionalTensorShape, bool]:
    assert input_shapes
    for input_shape in input_shapes:
      assert input_shape is not None
    maybe_shape = type_lib.try_broadcast(input_shapes)
    return (maybe_shape, maybe_shape is not None)


class NullableBroadcastableTransformOp(
    BroadcastableTransformOp, is_abstract=True
):
  """Abstract class for all transforms that may broadcast with special shapes.

  Allows some shapes to be None.
  """

  def __init__(self, *args, **kwargs):
    super(NullableBroadcastableTransformOp, self).__init__(*args, **kwargs)
    del args, kwargs

  def output_shape(
      self, input_shapes: List[OptionalTensorShape], **kwargs
  ) -> Tuple[OptionalTensorShape, bool]:
    input_shapes = [
        input_shape for input_shape in input_shapes if input_shape is not None
    ]
    assert input_shapes
    maybe_shape = type_lib.try_broadcast(input_shapes)
    return (maybe_shape, maybe_shape is not None)


class OutputOp(Op):
  """Base class for all output ops.

  An `OutputOp` is a terminal vertex whose value is not consumed by other
  vertices.
  """

  in_types: Tuple[T, ...] = ()  # Subclass must override this.
  out_type: Optional[T] = None  # Must be None.

  @classmethod
  def __init_subclass__(cls):
    super().__init_subclass__(final_cls=cls)
    if not cls.in_types:
      raise ValueError(f"OutputOp ${cls} must have non-empty input types.")
    if cls.out_type is not None:
      raise ValueError(f"OutputOp ${cls} must use None as out_type.")

  def __new__(cls, *args, **kwargs):
    """Prevents from instantiation without subclassing."""
    del args
    del kwargs

    if cls is OutputOp:
      raise TypeError(
          "Only subclasses of OutputOp can be used as types, "
          "not OutputOp itself."
      )
    return object.__new__(cls)

  def __init__(self, *args, **kwargs):
    del args, kwargs

  def execute(self, inputs, **exec_kwargs):
    assert len(inputs) == 1
    return inputs[0]

  def output_shape(
      self, input_shapes: List[OptionalTensorShape], **kwargs
  ) -> Tuple[OptionalTensorShape, bool]:
    return (input_shapes[0], True)


def check_id(element_id):
  if not element_id:
    raise ValueError("Invalid ID.")
