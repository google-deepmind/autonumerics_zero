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

"""Constants in the search space.

These are ops with no inputs and no learnable parameters.
"""

import pickle
from typing import List

from jax import numpy as jnp
import numpy as np

from evolution.projects.graphs import op as op_lib
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import search_space_base

FloatT = search_space_base.FloatT
JnpFloatDType = data_lib.JnpFloatDType


class ConstantOp(op_lib.InputOp):
  """Evolvable float base class.

  Sublclasses must define the `_generate` and `_mutate` methods and can
  optionally set the `upper` and `lower` class attributes.
  """

  out_type = FloatT()
  cls_has_evolvable_params = True

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    evolvable_params = self._generate(rng)
    return evolvable_params

  def mutate_evolvable_params(
      self, evolvable_params: float, impact: float, rng: np.random.RandomState
  ) -> float:
    evolvable_params = self._modify(evolvable_params, impact, rng)
    return evolvable_params

  def inherit_evolvable_params(self, evolvable_params: float) -> float:
    return evolvable_params

  def serialize_evolvable_params(self, evolvable_params: float) -> bytes:
    return pickle.dumps(evolvable_params)

  def deserialize_evolvable_params(self, serialized: bytes) -> float:
    return pickle.loads(serialized)

  def execute(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      inputs: List[jnp.ndarray],
      evolvable_params: float,
      dtype: JnpFloatDType,
      **kwargs
  ) -> jnp.ndarray:
    assert not inputs
    return dtype(evolvable_params)

  def _generate(self, rng: np.random.RandomState) -> float:
    return float(rng.normal(0.0, 1.0))

  def _modify(
      self, value: float, impact: float, rng: np.random.RandomState
  ) -> float:
    return value + float(rng.normal(0.0, impact))
