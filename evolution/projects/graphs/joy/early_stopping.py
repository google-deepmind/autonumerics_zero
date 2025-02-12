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

"""A CMA-ES-based trainer for coefficients."""

from jax import numpy as jnp
from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import early_stopping_spec_pb2

JnpFloat = data_lib.JnpFloat
JnpPreciseFloat = data_lib.JnpPreciseFloat


class EarlyStopping:
  """A class to handle the early stopping logic."""

  def __init__(self, spec: early_stopping_spec_pb2.EarlyStoppingSpec):
    self._spec = spec
    self._latest_improved_quantity = None
    self._latest_improved_iters = None
    if not self._spec.HasField(
        "required_absolute_improvement"
    ) and not self._spec.HasField("required_fractional_improvement"):
      raise ValueError(
          "Missing required_absolute_improvement or "
          "required_fractional_improvement."
      )
    if not self._spec.HasField("min_iters"):
      raise ValueError("Missing min_iters.")
    if not self._spec.HasField("max_no_improvement_iters_fraction"):
      raise ValueError("Missing max_no_improvement_iters_fraction.")
    if not self._spec.HasField("asymptotic_value"):
      raise ValueError("Missing asymptotic_value.")
    if not self._spec.HasField("from_above"):
      raise ValueError("Missing from_above.")

  def start(self):
    self._latest_improved_quantity = None
    self._latest_improved_iters = None

  def should_stop(self, iters: int, quantity: jnp.float64) -> bool:
    """Returns whether to stop early."""
    quantity = self._standardize_quantity(quantity)
    if quantity == 0.0:
      return True  # Training is complete.
    if self._met_required_improvement(quantity):
      self._latest_improved_iters = iters
      self._latest_improved_quantity = quantity
    if iters < self._spec.min_iters:
      # Never stop very early on in the search.
      return False
    no_improvement_iters = iters - self._latest_improved_iters
    max_no_improvement_iters = (
        iters * self._spec.max_no_improvement_iters_fraction
    )
    max_no_improvement_iters = max(max_no_improvement_iters, 1)
    if no_improvement_iters < round(max_no_improvement_iters):
      return False
    return True

  def _standardize_quantity(self, quantity: jnp.float64) -> jnp.float64:
    """Turns the quantity into an error that aims for 0 from above."""
    assert quantity.dtype == jnp.float64
    quantity = quantity - self._spec.asymptotic_value
    if not self._spec.from_above:
      quantity = -quantity
    if quantity < 0.0:
      raise RuntimeError("Wrong direction.")
    return quantity

  def _met_required_improvement(self, quantity: jnp.float64):
    """Whether enough progress has taken place."""
    if self._latest_improved_quantity is None:
      # No information yet.
      return True
    if (
        self._spec.HasField("required_absolute_improvement")
        and self._latest_improved_quantity - quantity
        <= self._spec.required_absolute_improvement
    ):
      # Did not meet the required (absolute) improvement.
      return False
    if (
        self._spec.HasField("required_fractional_improvement")
        and (self._latest_improved_quantity - quantity)
        / abs(self._latest_improved_quantity)
        <= self._spec.required_fractional_improvement
    ):
      # Did not meet the required (fractional) improvement.
      return False
    # Met all improvement criteria.
    return True
