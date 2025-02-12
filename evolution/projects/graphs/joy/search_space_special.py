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

"""Special functions in the search space."""

from typing import List

from jax import numpy as jnp

from evolution.projects.graphs import op as op_lib
from evolution.projects.graphs.joy import search_space_base


FloatT = search_space_base.FloatT


class ExpOp(op_lib.TransformOp):
  """An op that exponentiates e to a float power."""

  in_types = (FloatT(),)
  out_type = FloatT()

  def execute(self, inputs: List[jnp.ndarray], **exec_kwargs) -> jnp.ndarray:
    assert len(inputs) == 1
    result = jnp.exp(inputs[0].value)  # pytype: disable=attribute-error  # jax-ndarray
    return result
