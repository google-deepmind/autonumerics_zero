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

"""Basic arithmetic operations in the search space."""

from typing import List

from jax import numpy as jnp

from evolution.projects.graphs import op as op_lib
from evolution.projects.graphs.joy import search_space_base

FloatT = search_space_base.FloatT


class AddOp(op_lib.TransformOp):
  """An op that adds two numbers."""

  in_types = (FloatT(), FloatT())
  out_type = FloatT()

  def execute(self, inputs: List[jnp.ndarray], **exec_kwargs) -> jnp.ndarray:
    assert len(inputs) == 2
    result = jnp.add(inputs[0], inputs[1])
    return result

  def input_index_groups(self):
    return [[0, 1]]  # Commutative.


class SubOp(op_lib.TransformOp):
  """An op that subtracts two numbers."""

  in_types = (FloatT(), FloatT())
  out_type = FloatT()

  def execute(self, inputs: List[jnp.ndarray], **exec_kwargs) -> jnp.ndarray:
    assert len(inputs) == 2
    result = jnp.subtract(inputs[0], inputs[1])
    return result

  def input_index_groups(self):
    return [[0], [1]]  # Not commutative.


class MultOp(op_lib.TransformOp):
  """An op that multiplies two numbers."""

  in_types = (FloatT(), FloatT())
  out_type = FloatT()

  def execute(self, inputs: List[jnp.ndarray], **exec_kwargs) -> jnp.ndarray:
    assert len(inputs) == 2
    result = jnp.multiply(inputs[0], inputs[1])
    return result

  def input_index_groups(self):
    return [[0, 1]]  # Commutative.


class DivOp(op_lib.TransformOp):
  """An op that divides two numbers."""

  in_types = (FloatT(), FloatT())
  out_type = FloatT()

  def execute(self, inputs: List[jnp.ndarray], **exec_kwargs) -> jnp.ndarray:
    assert len(inputs) == 2
    result = jnp.divide(inputs[0], inputs[1])
    return result

  def input_index_groups(self):
    return [[0], [1]]  # Not commutative.


class SqrtOp(op_lib.TransformOp):
  """An op that takes the square root of a number."""

  in_types = (FloatT(),)
  out_type = FloatT()

  def execute(self, inputs: List[jnp.ndarray], **exec_kwargs) -> jnp.ndarray:
    assert len(inputs) == 1
    result = jnp.sqrt(inputs[0])
    return result

  def input_index_groups(self):
    return [[0]]  # Commutativity is irrelevant.


class FmaOp(op_lib.TransformOp):
  """An op that stands for a fused multiply-add.

  The operation will only be guaranteed to be fused when using the autonum
  library.
  """

  in_types = (FloatT(), FloatT(), FloatT())
  out_type = FloatT()

  def execute(self, inputs: List[jnp.ndarray], **exec_kwargs) -> jnp.ndarray:
    assert len(inputs) == 3
    result = jnp.add(jnp.multiply(inputs[0], inputs[1]), inputs[2])
    return result

  def input_index_groups(self) -> List[List[int]]:
    return [[0], [1], [2]]  # Non-commutative.


class IdentityOp(op_lib.TransformOp):
  """An op that returns its input."""

  in_types = (FloatT(),)
  out_type = FloatT()

  def execute(self, inputs: List[jnp.ndarray], **exec_kwargs) -> jnp.ndarray:
    assert len(inputs) == 1
    return inputs[0]  # pytype: disable=attribute-error  # jax-ndarray


class DoubleOp(op_lib.TransformOp):
  """An op that doubles its input."""

  in_types = (FloatT(),)
  out_type = FloatT()

  def execute(self, inputs: List[jnp.ndarray], **exec_kwargs) -> jnp.ndarray:
    assert len(inputs) == 1
    two = jnp.array(2, dtype=inputs[0].dtype)
    result = jnp.multiply(inputs[0], two)  # pytype: disable=attribute-error  # jax-ndarray
    return result


class HalfOp(op_lib.TransformOp):
  """An op that halves its input."""

  in_types = (FloatT(),)
  out_type = FloatT()

  def execute(self, inputs: List[jnp.ndarray], **exec_kwargs) -> jnp.ndarray:
    assert len(inputs) == 1
    two = jnp.array(2, dtype=inputs[0].dtype)
    result = jnp.divide(inputs[0], two)  # pytype: disable=attribute-error  # jax-ndarray
    return result


class QuadrupleOp(op_lib.TransformOp):
  """An op that quadruples its input."""

  in_types = (FloatT(),)
  out_type = FloatT()

  def execute(self, inputs: List[jnp.ndarray], **exec_kwargs) -> jnp.ndarray:
    assert len(inputs) == 1
    four = jnp.array(4, dtype=inputs[0].dtype)
    result = jnp.multiply(inputs[0], four)  # pytype: disable=attribute-error  # jax-ndarray
    return result


class QuarterOp(op_lib.TransformOp):
  """An op that divides its input by 4."""

  in_types = (FloatT(),)
  out_type = FloatT()

  def execute(self, inputs: List[jnp.ndarray], **exec_kwargs) -> jnp.ndarray:
    assert len(inputs) == 1
    four = jnp.array(4, dtype=inputs[0].dtype)
    result = jnp.divide(inputs[0], four)  # pytype: disable=attribute-error  # jax-ndarray
    return result
