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

"""Variables in the search space.

These are ops with no inputs that have a learnable parameter.
"""

import pickle
from typing import Any, List, Optional

from jax import numpy as jnp
from jax import random as jrandom
import numpy as np

from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs import op as op_lib
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import search_space_base

FloatT = search_space_base.FloatT
JnpFloat = data_lib.JnpFloat
JnpFloatDType = data_lib.JnpFloatDType
JnpPreciseFloat = data_lib.JnpPreciseFloat


class _RandomInitVariableOp(op_lib.InputOp):
  """Base class for variables with random initialization.

  Learning takes place at evaluation time.

  Subclasses must define the class attribute `init_scale`.
  """

  out_type = FloatT()
  cls_has_learnable_params = True

  # The initialization size of the variable. Required.
  init_scale: Optional[str] = None

  def __init__(self, *args, **kwargs):
    super(_RandomInitVariableOp, self).__init__(*args, **kwargs)
    if self.__class__.init_scale is None:
      raise NotImplementedError("init_scale must be set by subclass.")
    self._init_scale = jnp.array(
        self.__class__.init_scale, dtype=JnpPreciseFloat
    )

  def init_learnable_params(
      self, evolvable_params: float, jnp_key: jnp.ndarray, hashing_round: int
  ) -> JnpPreciseFloat:
    del hashing_round  # Same behavior for hashing and evaluation.
    return jnp.multiply(
        jrandom.normal(jnp_key, dtype=JnpPreciseFloat), self._init_scale
    )

  def serialize_learnable_params(self, learnable_params: Any) -> bytes:
    return pickle.dumps(learnable_params.tolist())

  def deserialize_learnable_params(self, serialized: bytes) -> Any:
    return jnp.array(pickle.loads(serialized), dtype=JnpPreciseFloat)

  def execute(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      inputs: List[jnp.ndarray],
      learnable_params: JnpFloat,
      dtype: JnpFloatDType,
      **kwargs
  ) -> jnp.ndarray:
    raise NotImplementedError("Must be implemented by subclass.")


class _ScaledVariableOp(_RandomInitVariableOp):
  """A learned scaled variable with random initialization.

  Learning takes place at evaluation time.

  Subclasses must define the class attribute `init_scale` and can define the
  class attribute `execute_scale` (if not defined, 1 is used).
  """

  # The variable will be multiplied by this number at execution time. If `None`,
  # there is no such scaling.
  execute_scale: Optional[str] = None

  def __init__(self, *args, **kwargs):
    super(_ScaledVariableOp, self).__init__(*args, **kwargs)
    if self.__class__.execute_scale is None:
      self._execute_scale = None
    else:
      self._execute_scale = jnp.array(
          self.__class__.execute_scale, dtype=JnpPreciseFloat
      )

  def execute(
      self,
      inputs: List[jnp.ndarray],
      learnable_params: JnpFloat,
      dtype: JnpFloatDType,
      **kwargs
  ) -> jnp.ndarray:
    assert not inputs
    assert not learnable_params.shape
    if learnable_params.dtype != JnpPreciseFloat:
      raise RuntimeError(
          "Unexpected learnable params dtype: %s" % str(learnable_params.dtype)
      )
    return_value = jnp.array(learnable_params, dtype=dtype)
    if self._execute_scale is not None:
      return_value = jnp.multiply(return_value, self._execute_scale)
    return return_value


class ZeroInitVariableOp(_ScaledVariableOp):
  """See base class."""

  init_scale = "0"


class TinyInitVariableOp(_ScaledVariableOp):
  """See base class."""

  init_scale = "0.01"


class RandomInitVariableOp(_ScaledVariableOp):
  """See base class."""

  init_scale = "1"


class _HeritableVariableOp(op_lib.InputOp):
  """Learnable variable with inherited initial condition."""

  out_type = FloatT()

  # The evolvable parameter is the initial condition for the learning of the
  # learnable parameter.
  cls_has_evolvable_params = True
  cls_has_learnable_params = True

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    raise NotImplementedError("Must be implemented by subclass.")

  def mutate_evolvable_params(
      self, evolvable_params: float, impact: float, rng: np.random.RandomState
  ) -> float:
    raise NotImplementedError("Must be implemented by subclass.")

  def inherit_evolvable_params(self, evolvable_params: float) -> float:
    return evolvable_params

  def serialize_evolvable_params(self, evolvable_params: float) -> bytes:
    return pickle.dumps(evolvable_params)

  def deserialize_evolvable_params(self, serialized: bytes) -> float:
    return pickle.loads(serialized)

  def init_learnable_params(
      self, evolvable_params: float, jnp_key: jnp.ndarray, hashing_round: int
  ) -> JnpPreciseFloat:
    if hashing_round == -1 or hashing_round == 0:
      # This is not a hashing round (i.e. it is an evaluation) or this is the
      # first round of hashing.
      return JnpPreciseFloat(evolvable_params)
    elif hashing_round > 0:
      # This is an additional round of hashing.
      return JnpPreciseFloat(evolvable_params) + jrandom.normal(
          jnp_key, dtype=JnpPreciseFloat
      )
    else:
      raise ValueError("Invalid hashing_round.")

  def serialize_learnable_params(self, learnable_params: Any) -> bytes:
    return pickle.dumps(learnable_params.tolist())

  def deserialize_learnable_params(self, serialized: bytes) -> Any:
    return jnp.array(pickle.loads(serialized), dtype=JnpPreciseFloat)

  def execute(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      inputs: List[jnp.ndarray],
      learnable_params: JnpFloat,
      dtype: JnpFloatDType,
      **kwargs
  ) -> jnp.ndarray:
    assert not inputs
    assert not learnable_params.shape
    assert learnable_params.dtype == JnpPreciseFloat
    return jnp.array(learnable_params, dtype=dtype)


class AnchoredVariableOp(_HeritableVariableOp):
  """Learnable variable with inherited initial condition.

  Mutations are additive.
  """

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    evolvable_params = float(rng.normal(0.0, 1.0))
    return evolvable_params

  def mutate_evolvable_params(
      self, evolvable_params: float, impact: float, rng: np.random.RandomState
  ) -> float:
    evolvable_params += float(rng.normal(0.0, impact))
    return evolvable_params


class _MultiplicativeHeritableVariableOp(_HeritableVariableOp):
  """Learned variable with inherited initial condition.

  Mutations are multiplicative.
  """

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    raise NotImplementedError("Must be implemented by subclass.")

  def mutate_evolvable_params(
      self, evolvable_params: float, impact: float, rng: np.random.RandomState
  ) -> float:
    evolvable_params *= float(rng.uniform(1.0 / (1.0 + impact), 1.0 + impact))
    return evolvable_params


class PositiveScale64AnchoredVariableOp(_MultiplicativeHeritableVariableOp):
  """Learnable positive variable with inherited initial condition.

  Details:
  -Initial condition is always positive.
  -Initial condition seeding is on a log scale over 64 orders of magnitude.
  -Initial condition mutations are multiplicative.
  """

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    scale = float(rng.uniform(-64.0, 0.0))
    evolvable_params = 10.0**scale
    return evolvable_params


class NegativeScale64AnchoredVariableOp(_MultiplicativeHeritableVariableOp):
  """Learnable negative variable with inherited initial condition.

  Details:
  -Initial condition is always negative.
  -Initial condition seeding is on a log scale over 64 orders of magnitude.
  -Initial condition mutations are multiplicative.
  """

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    scale = float(rng.uniform(-64.0, 0.0))
    evolvable_params = -(10.0**scale)
    return evolvable_params


class PositiveScale32AnchoredVariableOp(_MultiplicativeHeritableVariableOp):
  """Learnable positive variable with inherited initial condition.

  Details:
  -Initial condition is always positive.
  -Initial condition seeding is on a log scale over 32 orders of magnitude.
  -Initial condition mutations are multiplicative.
  """

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    scale = float(rng.uniform(-32.0, 0.0))
    evolvable_params = 10.0**scale
    return evolvable_params


class NegativeScale32AnchoredVariableOp(_MultiplicativeHeritableVariableOp):
  """Learnable negative variable with inherited initial condition.

  Details:
  -Initial condition is always negative.
  -Initial condition seeding is on a log scale over 32 orders of magnitude.
  -Initial condition mutations are multiplicative.
  """

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    scale = float(rng.uniform(-32.0, 0.0))
    evolvable_params = -(10.0**scale)
    return evolvable_params


class PositiveScale16AnchoredVariableOp(_MultiplicativeHeritableVariableOp):
  """Learnable positive variable with inherited initial condition.

  Details:
  -Initial condition is always positive.
  -Initial condition seeding is on a log scale over 16 orders of magnitude,
   making it suitable for 64-bit experiments.
  -Initial condition mutations are multiplicative.
  """

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    scale = float(rng.uniform(-16.0, 0.0))
    evolvable_params = 10.0**scale
    return evolvable_params


class NegativeScale16AnchoredVariableOp(_MultiplicativeHeritableVariableOp):
  """Learnable negative variable with inherited initial condition.

  Details:
  -Initial condition is always negative.
  -Initial condition seeding is on a log scale over 16 orders of magnitude,
   making it suitable for 64-bit experiments.
  -Initial condition mutations are multiplicative.
  """

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    scale = float(rng.uniform(-16.0, 0.0))
    evolvable_params = -(10.0**scale)
    return evolvable_params


class PositiveScale16NonAnchoredVariableOp(PositiveScale16AnchoredVariableOp):
  """Like PositiveScale16AnchoredVariableOp but ablates the inheritance.

  This variable ignores its evolvable params and reinitializes the learnable
  params to random values at the start of each training.
  """

  def init_learnable_params(
      self, evolvable_params: float, jnp_key: jnp.ndarray, hashing_round: int
  ) -> JnpPreciseFloat:
    del evolvable_params  # Ignore the evolvable params.
    if hashing_round != -1:
      raise ValueError("Hashing not supported.")
    scale = jrandom.uniform(
        jnp_key, shape=(), dtype=JnpPreciseFloat, minval=-16.0, maxval=0.0
    )
    evolvable_params = jnp.power(10.0, scale)
    return evolvable_params


class NegativeScale16NonAnchoredVariableOp(NegativeScale16AnchoredVariableOp):
  """Like NegativeScale16AnchoredVariableOp but ablates the inheritance."""

  def init_learnable_params(
      self, evolvable_params: float, jnp_key: jnp.ndarray, hashing_round: int
  ) -> JnpPreciseFloat:
    del evolvable_params  # Ignore the evolvable params.
    if hashing_round != -1:
      raise ValueError("Hashing not supported.")
    scale = jrandom.uniform(
        jnp_key, shape=(), dtype=JnpPreciseFloat, minval=-16.0, maxval=0.0
    )
    evolvable_params = -jnp.power(10.0, scale)
    return evolvable_params


class PositiveScale8AnchoredVariableOp(_MultiplicativeHeritableVariableOp):
  """Learnable positive variable with inherited initial condition.

  Details:
  -Initial condition is always positive.
  -Initial condition seeding is on a log scale over 16 orders of magnitude,
   making it suitable for 64-bit experiments.
  -Initial condition mutations are multiplicative.
  """

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    scale = float(rng.uniform(-8.0, 0.0))
    evolvable_params = 10.0**scale
    return evolvable_params


class NegativeScale8AnchoredVariableOp(_MultiplicativeHeritableVariableOp):
  """Learnable negative variable with inherited initial condition.

  Details:
  -Initial condition is always negative.
  -Initial condition seeding is on a log scale over 16 orders of magnitude,
   making it suitable for 64-bit experiments.
  -Initial condition mutations are multiplicative.
  """

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    scale = float(rng.uniform(-8.0, 0.0))
    evolvable_params = -(10.0**scale)
    return evolvable_params


class PositiveScale8NonAnchoredVariableOp(PositiveScale8AnchoredVariableOp):
  """Like PositiveScale8AnchoredVariableOp but ablates the inheritance.

  This variable ignores its evolvable params and reinitializes the learnable
  params to random values at the start of each training.
  """

  def init_learnable_params(
      self, evolvable_params: float, jnp_key: jnp.ndarray, hashing_round: int
  ) -> JnpPreciseFloat:
    del evolvable_params  # Ignore the evolvable params.
    if hashing_round != -1:
      raise ValueError("Hashing not supported.")
    scale = jrandom.uniform(
        jnp_key, shape=(), dtype=JnpPreciseFloat, minval=-8.0, maxval=0.0
    )
    evolvable_params = jnp.power(10.0, scale)
    return evolvable_params


class NegativeScale8NonAnchoredVariableOp(NegativeScale8AnchoredVariableOp):
  """Like NegativeScale8AnchoredVariableOp but ablates the inheritance."""

  def init_learnable_params(
      self, evolvable_params: float, jnp_key: jnp.ndarray, hashing_round: int
  ) -> JnpPreciseFloat:
    del evolvable_params  # Ignore the evolvable params.
    if hashing_round != -1:
      raise ValueError("Hashing not supported.")
    scale = jrandom.uniform(
        jnp_key, shape=(), dtype=JnpPreciseFloat, minval=-8.0, maxval=0.0
    )
    evolvable_params = -jnp.power(10.0, scale)
    return evolvable_params


class PositiveScale16LamarckianVariableOp(PositiveScale16AnchoredVariableOp):
  """A lamarckian version of PositiveScale16AnchoredVariableOp.

  For purely Lamarckian behavior, don't mutate the evolvable params. If you
  do, there will be some randomization of initial conditions just before
  training (which may or may not be desirable).
  """

  def update_evolvable_params(
      self, evolvable_params: Any, learnable_params: Any
  ) -> Optional[Any]:
    del evolvable_params
    return learnable_params  # Lamarckian evolution.


class NegativeScale16LamarckianVariableOp(NegativeScale16AnchoredVariableOp):
  """A lamarckian version of NegativeScale16AnchoredVariableOp.

  For purely Lamarckian behavior, don't mutate the evolvable params. If you
  do, there will be some randomization of initial conditions just before
  training (which may or may not be desirable).
  """

  def update_evolvable_params(
      self, evolvable_params: Any, learnable_params: Any
  ) -> Optional[Any]:
    del evolvable_params
    return learnable_params  # Lamarckian evolution.


class PositiveScale8LamarckianVariableOp(PositiveScale8AnchoredVariableOp):
  """A lamarckian version of PositiveScale8AnchoredVariableOp.

  For purely Lamarckian behavior, don't mutate the evolvable params. If you
  do, there will be some randomization of initial conditions just before
  training (which may or may not be desirable).
  """

  def update_evolvable_params(
      self, evolvable_params: Any, learnable_params: Any
  ) -> Optional[Any]:
    del evolvable_params
    return learnable_params  # Lamarckian evolution.


class NegativeScale8LamarckianVariableOp(NegativeScale8AnchoredVariableOp):
  """A lamarckian version of NegativeScale8AnchoredVariableOp.

  For purely Lamarckian behavior, don't mutate the evolvable params. If you
  do, there will be some randomization of initial conditions just before
  training (which may or may not be desirable).
  """

  def update_evolvable_params(
      self, evolvable_params: Any, learnable_params: Any
  ) -> Optional[Any]:
    del evolvable_params
    return learnable_params  # Lamarckian evolution.


class PositiveScale4AnchoredVariableOp(_MultiplicativeHeritableVariableOp):
  """Learnable positive variable with inherited initial condition.

  Details:
  -Initial condition is always positive.
  -Initial condition seeding is on a log scale over 16 orders of magnitude,
   making it suitable for 64-bit experiments.
  -Initial condition mutations are multiplicative.
  """

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    scale = float(rng.uniform(-4.0, 0.0))
    evolvable_params = 10.0**scale
    return evolvable_params


class NegativeScale4AnchoredVariableOp(_MultiplicativeHeritableVariableOp):
  """Learnable negative variable with inherited initial condition.

  Details:
  -Initial condition is always negative.
  -Initial condition seeding is on a log scale over 16 orders of magnitude,
   making it suitable for 64-bit experiments.
  -Initial condition mutations are multiplicative.
  """

  def create_evolvable_params(self, rng: np.random.RandomState) -> float:
    scale = float(rng.uniform(-4.0, 0.0))
    evolvable_params = -(10.0**scale)
    return evolvable_params


class PositiveScale4NonAnchoredVariableOp(PositiveScale4AnchoredVariableOp):
  """Like PositiveScale4AnchoredVariableOp but ablates the inheritance.

  This variable ignores its evolvable params and reinitializes the learnable
  params to random values at the start of each training.
  """

  def init_learnable_params(
      self, evolvable_params: float, jnp_key: jnp.ndarray, hashing_round: int
  ) -> JnpPreciseFloat:
    del evolvable_params  # Ignore the evolvable params.
    if hashing_round != -1:
      raise ValueError("Hashing not supported.")
    scale = jrandom.uniform(
        jnp_key, shape=(), dtype=JnpPreciseFloat, minval=-4.0, maxval=0.0
    )
    evolvable_params = jnp.power(10.0, scale)
    return evolvable_params


class NegativeScale4NonAnchoredVariableOp(NegativeScale4AnchoredVariableOp):
  """Like NegativeScale4AnchoredVariableOp but ablates the inheritance."""

  def init_learnable_params(
      self, evolvable_params: float, jnp_key: jnp.ndarray, hashing_round: int
  ) -> JnpPreciseFloat:
    del evolvable_params  # Ignore the evolvable params.
    if hashing_round != -1:
      raise ValueError("Hashing not supported.")
    scale = jrandom.uniform(
        jnp_key, shape=(), dtype=JnpPreciseFloat, minval=-4.0, maxval=0.0
    )
    evolvable_params = -jnp.power(10.0, scale)
    return evolvable_params


class _LogRandomInitVariableOp(op_lib.InputOp):
  """Base class for learned variables with log-random initialization.

  Learning takes place at evaluation time.

  Subclasses must define the class attribute `init_scale`.
  """

  out_type = FloatT()
  cls_has_learnable_params = True

  # The range of the distribution of the initialization size in log scale.
  # Required.
  log_scale_min: Optional[float] = None
  log_scale_max: Optional[float] = None

  def __init__(self, *args, **kwargs):
    super(_LogRandomInitVariableOp, self).__init__(*args, **kwargs)
    if self.__class__.log_scale_min is None:
      raise NotImplementedError("log_scale_min must be set by subclass.")
    if self.__class__.log_scale_max is None:
      raise NotImplementedError("log_scale_max must be set by subclass.")
    self._log_scale_min = jnp.array(
        self.__class__.log_scale_min, dtype=JnpPreciseFloat
    )
    self._log_scale_max = jnp.array(
        self.__class__.log_scale_max, dtype=JnpPreciseFloat
    )

  def init_learnable_params(
      self, evolvable_params: float, jnp_key: jnp.ndarray, hashing_round: int
  ) -> JnpPreciseFloat:
    if hashing_round != -1:
      raise ValueError("Hashing not supported.")

    base = jnp.array(10.0, dtype=JnpPreciseFloat)
    scale = jrandom.uniform(
        jnp_key,
        shape=(),
        dtype=JnpPreciseFloat,
        minval=self._log_scale_min,
        maxval=self._log_scale_max,
    )
    sign = jrandom.choice(
        key=jnp_key, a=jnp.array([1.0, -1.0], dtype=JnpPreciseFloat), shape=()
    )
    init_value = sign * jnp.power(base, scale)
    return init_value

  def serialize_learnable_params(self, learnable_params: Any) -> bytes:
    return pickle.dumps(learnable_params.tolist())

  def deserialize_learnable_params(self, serialized: bytes) -> Any:
    return jnp.array(pickle.loads(serialized), dtype=JnpPreciseFloat)

  def execute(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      inputs: List[jnp.ndarray],
      learnable_params: JnpFloat,
      dtype: JnpFloatDType,
      **kwargs
  ) -> jnp.ndarray:
    assert not inputs
    assert not learnable_params.shape
    if learnable_params.dtype != JnpPreciseFloat:
      raise RuntimeError(
          "Unexpected learnable params dtype: %s" % str(learnable_params.dtype)
      )
    return_value = jnp.array(learnable_params, dtype=dtype)
    return return_value


class Scale4LogRandomInitVariableOp(_LogRandomInitVariableOp):
  """Learnable variable with log-scale initialization and no heritability.

  Details:
  -Even probability of initializing to positive or negative.
  -Uniform initialization on a log scale over 4 orders of magnitude.
  """

  log_scale_min = -4.0
  log_scale_max = 0.0


class Scale8LogRandomInitVariableOp(_LogRandomInitVariableOp):
  """Learnable variable with log-scale initialization and no heritability.

  Details:
  -Even probability of initializing to positive or negative.
  -Uniform initialization on a log scale over 8 orders of magnitude.
  """

  log_scale_min = -8.0
  log_scale_max = 0.0


class Scale16LogRandomInitVariableOp(_LogRandomInitVariableOp):
  """Learnable variable with log-scale initialization and no heritability.

  Details:
  -Even probability of initializing to positive or negative.
  -Uniform initialization on a log scale over 16 orders of magnitude.
  """

  log_scale_min = -16.0
  log_scale_max = 0.0
