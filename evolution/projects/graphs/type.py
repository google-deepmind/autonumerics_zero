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

from typing import List, Optional, Tuple
import numpy as np


class T(object):
  """Base class for all types.

  The T instance is an abstract representation for a type. An operation
  has specific input and output types. Types might be, for example a
  "FloatT" to represent floats, a "PopulationT" to represent a population, or
  a "ReplayBufferT" to represent a replay buffer.
  """

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

    if cls is T:
      raise TypeError(
          "Only subclasses of T can be used as types, not T itself."
      )
    return object.__new__(cls)

  def __eq__(self, other) -> bool:
    if other is None:
      return False
    else:
      return self.type_id == other.type_id

  def __ne__(self, other) -> bool:
    return not self == other

  def __hash__(self):
    return hash(self.type_id)

  @property
  def type_id(self) -> str:
    return self.__class__.id()


class TensorT(T):
  """Generic Tensor type.

  Objects of this class can have arbitrary shapes.
  """


# Tensor or tensor-like objects have shapes. Some object, e.g., a Haiku network
# parameters may use its output dimension as shape hint.
TensorShape = Tuple[int, ...]

# Not all objects in the graph have shape defined. Examples are: Haiku network
# definition, tfd distribution object.
OptionalTensorShape = Optional[TensorShape]


def try_broadcast(shape_list: List[TensorShape]) -> Optional[TensorShape]:
  # We could easily implement our own logic (without allocating memories).
  # However, we don't want to maintain excess unit tests for this helper. The
  # performance difference is negligible.
  """Function that verifies if the shapes in shape_list are compatible."""
  a = np.ones(shape_list[0])
  try:
    for shape in shape_list[1:]:
      a = a + np.ones(shape)
    return a.shape
  except ValueError as e:
    if "broadcast" in str(e):
      return None
    else:
      raise e
