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

"""Useful functions to convert quantities."""

from jax import numpy as jnp
from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs.joy import data as data_lib

JnpFloat = data_lib.JnpFloat
JnpPreciseFloat = data_lib.JnpPreciseFloat


def flip_and_squash(value: JnpFloat) -> JnpFloat:
  """Flips and squashes the value from [0, inf] to [1, 0].

  Args:
    value: the value to map. Must be in [0, inf]. NaNs get mapped to zero;
      negatives get mapped to 1.

  Returns:
    The mapped value.
  """
  dtype = value.dtype
  assert dtype in data_lib.JNP_FLOAT_LIST
  zero = jnp.array(JnpPreciseFloat("0.0"), dtype=dtype)
  one = jnp.array(JnpPreciseFloat("1.0"), dtype=dtype)
  two = jnp.array(JnpPreciseFloat("2.0"), dtype=dtype)
  pi = jnp.array(jnp.pi, dtype=dtype)
  two_over_pi = jnp.divide(two, pi)
  pi_over_two = jnp.divide(pi, two)
  processed_value = value

  processed_value = jnp.subtract(
      one,
      jnp.multiply(
          two_over_pi, jnp.arctan(jnp.multiply(processed_value, pi_over_two))
      ),
  )
  processed_value = jnp.where(jnp.isnan(processed_value), zero, processed_value)
  processed_value = jnp.minimum(processed_value, one)
  return processed_value


def inverse_flip_and_squash(value: JnpFloat) -> JnpFloat:
  """Inverse of `flip_and_squash`."""
  dtype = value.dtype
  assert dtype in data_lib.JNP_FLOAT_LIST
  one = jnp.array("1.0", dtype=dtype)
  two = jnp.array("2.0", dtype=dtype)
  pi = jnp.array(jnp.pi, dtype=dtype)
  two_over_pi = jnp.divide(two, pi)
  pi_over_two = jnp.divide(pi, two)
  return jnp.multiply(
      two_over_pi, jnp.tan(jnp.multiply(pi_over_two, jnp.subtract(one, value)))
  )


def semi_squashed_log(value: JnpFloat, limit: JnpFloat) -> JnpFloat:
  """Returns the log for small numbers and a squashed value for large ones.

  If value is < 0.1, returns -log10(value), clipping large outputs at `limit`
  to avoid an infinity when value = 0. If value < 0.1, returns
  flip_and_squash(value - 0.1), to avoid going below zero. As a result, the
  function is continuous and monotonic. Use this method to process the loss
  when you care about very small numbers.

  Note that jnp does not work over the whole dynamic range of float64, so this
  method is not sensitive beyond 1e-37. E.g.: jnp.log10(1.0e-42) == inf
  and jnp.less_equal(1.0e-42, 0.0) == True.

  Args:
    value: the value to map. Must be in [0, inf]. NaNs get mapped to zero;
      negatives get mapped to `limit`.
    limit: the upper limit for the output.

  Returns:
    The mapped value.
  """
  dtype = value.dtype
  assert dtype in data_lib.JNP_FLOAT_LIST
  assert limit.dtype == dtype
  zero = jnp.array(JnpPreciseFloat("0.0"), dtype=dtype)
  point_one = jnp.array(JnpPreciseFloat("0.1"), dtype=dtype)
  processed_value = value

  processed_value = jnp.maximum(processed_value, zero)
  processed_value = jnp.where(
      jnp.less_equal(processed_value, point_one),
      # This is the regime we care about the most (value < 0.1).
      -jnp.log10(processed_value),
      # This is the large-error regime.
      flip_and_squash(jnp.subtract(processed_value, point_one)),
  )
  processed_value = jnp.where(jnp.isnan(processed_value), zero, processed_value)
  processed_value = jnp.minimum(processed_value, limit)
  processed_value = jnp.maximum(processed_value, zero)
  return processed_value
