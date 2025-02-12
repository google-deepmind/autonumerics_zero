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

"""Ground truth calculator."""

from typing import Callable, Optional
import jax.numpy as jnp
from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs.joy import conversions
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import error_util_pb2
from evolution.projects.graphs.joy import ground_truth as ground_truth_lib

JnpFloatDType = data_lib.JnpFloatDType
JnpPreciseFloat = data_lib.JnpPreciseFloat


class ErrorAccumulator:
  """Aggregates the maximum relative error."""

  def __init__(
      self,
      ground_truth: ground_truth_lib.GroundTruth,
      epsilon: JnpPreciseFloat,
      ulp_dtype: Optional[JnpFloatDType],
      override_ulp_with_value_at: Optional[float],
  ):
    """Initializes the instance.

    Args:
      ground_truth: the object to use to compute ground truth comparisons.
      epsilon:  non-negative value that will be added to the absolute value of
        the denominator when calculating the relative error. This is to avoid
        division by zero.
      ulp_dtype: the dtype in which to measure error in ULPs, or `None` if not
        measuring ULPs.
      override_ulp_with_value_at: a value x s.t. all ULP values for releative
        error calculation are set to ulp(x). Normal ULP values used if this arg
        is `None`.
    """
    self._ground_truth = ground_truth
    self._epsilon = epsilon
    self._ulp_dtype = ulp_dtype
    self._override_ulp_with_value_at = override_ulp_with_value_at
    self._max_error_value = jnp.array(-jnp.inf, dtype=JnpPreciseFloat)
    self._max_error_input = None

  def reset(self):
    self._max_error_value = jnp.array(-jnp.inf, dtype=JnpPreciseFloat)
    self._max_error_input = None

  def accumulate_error(self, error: JnpPreciseFloat):
    """Accumulates the given error.

    Args:
      error: the error found from examples already computed. This is typically
        obtained from calling the `Error` method in other instantiations of this
        class.
    """
    assert error.dtype == JnpPreciseFloat
    self._max_error_value = jnp.maximum(self._max_error_value, error)

  def accumulate_error_with_input(
      self, error: JnpPreciseFloat, error_input: JnpPreciseFloat
  ):
    """Accumulates the given error.

    Args:
      error: the error found from examples already computed. This is typically
        obtained from calling the `Error` method in other instantiations of this
        class.
      error_input: The input that produced the maximum error, for tracking
        purposes.
    """
    assert error.dtype == JnpPreciseFloat
    if error > self._max_error_value:
      self._max_error_input = error_input
      self._max_error_value = error

  def accumulate_examples(self, inputs: jnp.ndarray, predictions: jnp.ndarray):
    """Accumulates the error incurred by the given examples.

    Args:
      inputs: input values.
      predictions: outputs corresponding to the `inputs` arg.
    """
    signed_errors = self._ground_truth.signed_relative_errors(
        inputs=inputs,
        predictions=predictions,
        epsilon=self._epsilon,
        ulp_dtype=self._ulp_dtype,
        override_ulp_with_value_at=self._override_ulp_with_value_at,
    )
    assert signed_errors.dtype == JnpPreciseFloat
    abs_errors = jnp.abs(signed_errors)
    max_example_error = jnp.max(abs_errors)
    if max_example_error > self._max_error_value:
      self._max_error_value = max_example_error
      max_error_index = jnp.where(abs_errors == max_example_error)[0][0]
      self._max_error_input = inputs[max_error_index]

  def max_relative_error(self) -> JnpPreciseFloat:
    """Returns the accumulated error."""
    return self._max_error_value

  def max_relative_error_input(self) -> JnpPreciseFloat:
    """Returns the input that produced the max relative error."""
    return self._max_error_input


def get_quality_fitness_fn(
    spec: error_util_pb2.QualityFitnessSpec,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Returns a function to convert error to quality fitness."""
  if spec.HasField("log_error"):
    if not spec.log_error.HasField("max"):
      raise ValueError("Missing `max`.")
    quality_fitness_max = JnpPreciseFloat(spec.log_error.max)

    def quality_fitness_fn(error):
      return conversions.semi_squashed_log(error, limit=quality_fitness_max)

    return quality_fitness_fn
  elif spec.HasField("minus_error"):
    min_quality_limit = JnpPreciseFloat(spec.minus_error.min)
    max_quality_limit = (
        JnpPreciseFloat(spec.minus_error.max)
        if spec.minus_error.HasField("max")
        else jnp.inf
    )

    def quality_fitness_fn(error):
      quality_fitness = -error
      quality_fitness = jnp.maximum(quality_fitness, min_quality_limit)
      quality_fitness = jnp.minimum(quality_fitness, max_quality_limit)
      return quality_fitness

    return quality_fitness_fn
  else:
    raise NotImplementedError("Unknown quality_fitness_type.")
