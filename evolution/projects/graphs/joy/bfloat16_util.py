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

"""A dataset computed in Python."""

from typing import Optional
from jax import numpy as jnp
from evolution.lib.python import deconstructed


def all_bfloat16_values(
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    skip_every: Optional[int] = None,
    inputs_min_inclusive: bool = True,
    inputs_max_inclusive: bool = True,
) -> jnp.ndarray:
  """Returns all bfloat16 values.

  This includes all possible values representable as bfloat16. The args allow
  constraining the output to a specified subset.

  Args:
    min_value: if not None, only values greater or equal to this are included.
    max_value: if not None, only values greater or equal to this are included.
    skip_every: it not None, only one point every this many is produced.
    inputs_min_inclusive: if not True, min_value is not included in the final
      list.
    inputs_max_inclusive: if not True, max_value is not included in the final
      list.
  """
  if min_value is not None and min_value < -3.38953e38:
    raise ValueError(
        "Incorrect value. Min value lower than lowest bfloat16 value."
    )
  if max_value is not None and max_value > 3.38953e38:
    raise ValueError(
        "Incorrect value. Max value higher than highest bfloat16 value."
    )
  if skip_every is not None and skip_every <= 0:
    raise ValueError(
        "Incorrect value. skip_every should be a non-zero positive number."
    )

  values_list = []
  for exponent in range(-126, 128):
    for significand in range(0, 128):
      binary_significand = format(significand, "b").zfill(7)
      deconstructed_representation = (
          deconstructed.DeconstructedBFloat16.ConstructFromExactRepr(
              binary_significand, exponent
          )
      )
      values_list.append(deconstructed_representation.ToFloat())
      values_list.append(deconstructed_representation.ToFloat() * -1)
  values_list.append(0)
  values_array = jnp.array(values_list, dtype=jnp.bfloat16)

  # Truncate the inputs to the range if specified.
  if min_value is not None:
    min_value = jnp.bfloat16(min_value)
    if inputs_min_inclusive:
      values_array = values_array[(values_array >= min_value)]
    else:
      values_array = values_array[(values_array > min_value)]
  if max_value is not None:
    max_value = jnp.bfloat16(max_value)
    if inputs_max_inclusive:
      values_array = values_array[(values_array <= max_value)]
    else:
      values_array = values_array[(values_array < max_value)]

  values_array = jnp.sort(values_array)
  if skip_every is not None:
    values_array = values_array[0 : len(values_array) : skip_every]

  return values_array
