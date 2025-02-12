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

"""Search space ops specific to evolving erf."""

from typing import List

from jax import numpy as jnp

from evolution.projects.graphs import op as op_lib
from evolution.projects.graphs.joy import search_space_base

FloatT = search_space_base.FloatT


# Should have been ErfcLeadingOrderOp, but cannot change because class names are
# stored in GraphSpec protos.
class ErfLeadingOrderOp(op_lib.TransformOp):
  """An op that encodes the exponential factor of erfc(x)."""

  in_types = (FloatT(),)
  out_type = FloatT()

  def execute(self, inputs: List[FloatT], **exec_kwargs) -> FloatT:
    assert len(inputs) == 1
    result = jnp.exp(-jnp.square(inputs[0]))  # pytype: disable=wrong-arg-types  # jnp-types
    return result  # pytype: disable=bad-return-type  # jax-types


class ErfcAsymptoticBehaviorOp(op_lib.TransformOp):
  """An op that encodes the leading asymptotic behavior of erfc(x)."""

  in_types = (FloatT(),)
  out_type = FloatT()

  def execute(self, inputs: List[FloatT], **exec_kwargs) -> FloatT:
    assert len(inputs) == 1
    result = jnp.exp(-jnp.square(inputs[0])) / inputs[0] / jnp.sqrt(jnp.pi)  # pytype: disable=wrong-arg-types  # jnp-types
    return result  # pytype: disable=bad-return-type  # jax-types
