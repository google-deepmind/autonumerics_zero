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

"""The basic components of the search space.

These include types, basic base classes, required inputs, and required outputs.
"""

from typing import List, Optional

from jax import numpy as jnp

from evolution.projects.graphs import op as op_lib
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import search_space_base

FloatT = search_space_base.FloatT
JnpFloatDType = data_lib.JnpFloatDType
JnpPreciseFloat = data_lib.JnpPreciseFloat


class _LiteralOpBase(op_lib.InputOp):
  """Non-evolvable float base class.

  Sublclasses must define the `value` attribute.
  """

  out_type = FloatT()

  # The literal value. Required to be defined by subclass.
  value: Optional[str] = None

  def __init__(self, *args, **kwargs):
    super(_LiteralOpBase, self).__init__(*args, **kwargs)
    if self.__class__.value is None:
      raise NotImplementedError("Value must be set by subclass.")
    self._value = jnp.array(self.__class__.value, dtype=JnpPreciseFloat)

  def execute(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      inputs: List[jnp.ndarray],
      evolvable_params: float,
      dtype: JnpFloatDType,
      **kwargs
  ) -> jnp.ndarray:
    assert not inputs
    return jnp.array(self._value, dtype=dtype)


class ZeroOp(_LiteralOpBase):
  """The value 0."""

  value = "0"


class OneOp(_LiteralOpBase):
  """The value 1."""

  value = "1"


class TwoOp(_LiteralOpBase):
  """The value 2."""

  value = "2"
