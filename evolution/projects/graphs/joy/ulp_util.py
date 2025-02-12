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

"""Utility for measuring the size of an ULP (unit-in-the-last-place)."""

import math
from jax import numpy as jnp
import numpy as np
from evolution.lib.python import deconstructed
from evolution.projects.graphs.joy import data as data_lib

JnpFloatDType = data_lib.JnpFloatDType
JnpPreciseFloat = data_lib.JnpPreciseFloat


def ulp_array(x: jnp.ndarray, dtype: JnpFloatDType) -> jnp.ndarray:
  """Computes 1 ULP for each value.

  Args:
    x: an array of JnpPreciseFloat values.
    dtype: the dtype defining the ULP.

  Returns:
    An array of JnpPreciseFloat values measuring 1 ULP of each of the values
    in `x`.
  """
  assert x.dtype == JnpPreciseFloat
  if dtype == jnp.float64:
    ulp_fn = ulp_float64
  elif dtype == jnp.float32:
    ulp_fn = ulp_float32
  elif dtype == jnp.bfloat16:
    ulp_fn = ulp_bfloat16
  else:
    raise NotImplementedError("Unsupported dtype.")
  ulps = []
  for v in x.tolist():
    ulps.append(ulp_fn(v))
  return jnp.array(ulps, dtype=JnpPreciseFloat)


def ulp_float(x: float, dtype: JnpFloatDType) -> float:
  """Computes 1 ULP for one value.

  Args:
    x: a value.
    dtype: the dtype defining the ULP.

  Returns:
    A float measuring 1 ULP of `x`.
  """
  if dtype == jnp.float64:
    return ulp_float64(x)
  elif dtype == jnp.float32:
    return ulp_float32(x)
  elif dtype == jnp.bfloat16:
    return ulp_bfloat16(x)
  else:
    raise NotImplementedError("Unsupported dtype.")


def ulp_float64(x: float) -> float:
  return abs(math.ulp(x))


def ulp_float32(x: float) -> float:
  """Measures 1 ULP in float32."""
  if np.isnan(x):
    return float("nan")
  elif np.isinf(x):
    return float("inf")
  else:
    x = np.float32(x)
    u = np.abs(np.spacing(x))
    return float(u)


def ulp_bfloat16(x: float) -> float:
  return deconstructed.RoughBfloat16Ulp(x)
