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

from typing import List

from jax import numpy as jnp

from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import op as op_lib
from evolution.projects.graphs.joy import data as data_lib

JnpFloatDType = data_lib.JnpFloatDType


class FloatT(graph_lib.T):
  """A float type."""


class ProduceOp(op_lib.InputOp):
  """An input to the graph."""

  out_type = FloatT()

  def execute(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self, inputs: List[jnp.ndarray], dtype: JnpFloatDType, **kwargs
  ) -> jnp.ndarray:
    assert isinstance(inputs, list)
    ret = super().execute(inputs, **kwargs)
    assert ret.dtype == dtype
    return ret


class ProduceXOp(ProduceOp):
  """The x-input to the graph."""


class ProduceYOp(ProduceOp):
  """The y-input to the graph."""


class ProduceZOp(ProduceOp):
  """The z-input to the graph."""


class ConsumeOp(op_lib.OutputOp):
  """An op that consumes a FloatT."""

  in_types = (FloatT(),)
  out_type = None

  def execute(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self, inputs: List[jnp.ndarray], dtype: JnpFloatDType, **kwargs
  ) -> jnp.ndarray:
    assert isinstance(inputs, list)
    ret = super().execute(inputs, **kwargs)
    assert isinstance(ret, jnp.ndarray)
    return ret


class ConsumeFOp(ConsumeOp):
  """The first output of the graph."""


class ConsumeGOp(ConsumeOp):
  """The second output of the graph."""
