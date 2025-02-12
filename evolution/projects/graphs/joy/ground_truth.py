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

"""Tools to calculate the maximum relative error to a ground truth."""

import fractions
import math
from typing import List, Optional
import jax.numpy as jnp
from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import ground_truth_spec_pb2
from evolution.projects.graphs.joy import ulp_util
from evolution.projects.graphs.joy.python import float80_ground_truth

Fraction = fractions.Fraction
JnpFloatDType = data_lib.JnpFloatDType
JnpPreciseFloat = data_lib.JnpPreciseFloat


class GroundTruth:
  """Interface for ground truth objects."""

  def labels(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Computes the labels.

    Args:
      inputs: the inputs.

    Returns:
      The labels.
    """
    raise NotImplementedError("Must be implemented by subclass.")

  def signed_relative_errors(
      self,
      inputs: jnp.ndarray,
      predictions: jnp.ndarray,
      epsilon: JnpPreciseFloat,
      ulp_dtype: Optional[JnpFloatDType],
      override_ulp_with_value_at: Optional[float],
  ) -> jnp.ndarray:
    """Computes the relative errors element-wise.

    Intended to provide a more precise implementation. Internal calculations
    may be done in higher precision.

    Args:
      inputs: the inputs.
      predictions: the predicted outputs.
      epsilon: non-negative value that will be added to the absolute value of
        the denominator when calculating the relative errors. This is to
        optionally avoid division by zero.
      ulp_dtype: the dtype to use for calculating ULPs. If `None`, calculates
        relative error w.r.t. to the full value of the label. If not `None`,
        calculates relative error w.r.t. to 1 ULP of the label.
      override_ulp_with_value_at: a value x s.t. ulp(x) overrides all the ulp
        values for relative error calculation. If `None`, ulp(label) is used.

    Returns:
      The relative errors in float64 dtype.
    """
    raise NotImplementedError("Must be implemented by subclass.")


def build(spec: ground_truth_spec_pb2.GroundTruthSpec):
  """Builds the ground truth."""
  if spec.HasField("exp2_float80"):
    return WrappedExp2Float80GroundTruth(spec.exp2_float80)
  elif spec.HasField("exp2_vetted"):
    return WrappedExp2VettedGroundTruth(spec.exp2_vetted)
  elif spec.HasField("expe_float80"):
    return WrappedExpeFloat80GroundTruth(spec.expe_float80)
  elif spec.HasField("log2_float80"):
    return WrappedLog2Float80GroundTruth(spec.log2_float80)
  elif spec.HasField("loge_float80"):
    return WrappedLogeFloat80GroundTruth(spec.loge_float80)
  elif spec.HasField("erf_float80"):
    return WrappedErfFloat80GroundTruth(spec.erf_float80)
  elif spec.HasField("erfc_float80"):
    return WrappedErfcFloat80GroundTruth(spec.erfc_float80)
  elif spec.HasField("wavy_float80"):
    return WrappedWavyFloat80GroundTruth(spec.wavy_float80)
  elif spec.HasField("airy_float80"):
    return WrappedAiryFloat80GroundTruth(spec.airy_float80)
  elif spec.HasField("bessel_float80"):
    return WrappedBesselFloat80GroundTruth(spec.bessel_float80)
  elif spec.HasField("exp2_convergent"):
    return Exp2ConvergentGroundTruth(spec.exp2_convergent)
  elif spec.HasField("rough_two_x"):
    return RoughTwoXGroundTruth(spec.rough_two_x)
  else:
    raise NotImplementedError("Ground truth type not implemented.")


class WrappedFloat80GroundTruth(GroundTruth):
  """Base class for XXWrappedFloat80GroundTruth classes below."""

  def __init__(self):
    self._wrapped_ground_truth = None

  def labels(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Computes labels in float80, the casts them to float64.

    May double-round.

    Args:
      inputs: see base class.

    Returns:
      The labels.
    """
    if self._wrapped_ground_truth is None:
      raise NotImplementedError(
          "Must be implemented by subclass's constructor."
      )
    if len(inputs.shape) != 1:
      raise ValueError("Invalid array shape.")
    if inputs.shape[0] == 0:
      raise ValueError("Empty array.")
    labels: List[float] = []
    for x in inputs:
      labels.append(self._wrapped_ground_truth.label(float(x)))
    return jnp.array(labels, dtype=JnpPreciseFloat)

  def signed_relative_errors(
      self,
      inputs: jnp.ndarray,
      predictions: jnp.ndarray,
      epsilon: JnpPreciseFloat,
      ulp_dtype: Optional[JnpFloatDType],
      override_ulp_with_value_at: Optional[float],
  ) -> jnp.ndarray:
    """Computes the element-wise errors in float80, casts to float64 at the end.

    While it double rounds float80->float64, it compares predictions and labels
    at float80 precision.

    Args:
      inputs: see base class.
      predictions: see base class.
      epsilon: see base class.
      ulp_dtype: see base class.
      override_ulp_with_value_at: see base class.

    Returns:
      The errors. The return dtype is float64.
    """
    if self._wrapped_ground_truth is None:
      raise NotImplementedError(
          "Must be implemented by subclass's constructor."
      )
    _check_inputs_and_predictions(inputs, predictions)
    ulp_dtype_spec = data_lib.get_dtype_spec(ulp_dtype)
    errors: List[float] = []
    for x, y in zip(inputs, predictions):
      if override_ulp_with_value_at is None:
        errors.append(
            self._wrapped_ground_truth.signed_relative_error(
                float(x), float(y), float(epsilon), ulp_dtype_spec
            )
        )
      else:
        errors.append(
            self._wrapped_ground_truth.signed_relative_error_with_ulp_at(
                float(x),
                float(y),
                float(epsilon),
                ulp_dtype_spec,
                override_ulp_with_value_at,
            )
        )
    return jnp.array(errors, dtype=JnpPreciseFloat)


class WrappedExp2Float80GroundTruth(WrappedFloat80GroundTruth):
  """See the `Exp2Float80GroundTruthSpec` proto."""

  def __init__(self, spec: ground_truth_spec_pb2.Exp2Float80GroundTruthSpec):
    super().__init__()
    del spec
    self._wrapped_ground_truth = float80_ground_truth.Exp2Float80GroundTruth()


class WrappedExp2VettedGroundTruth(WrappedFloat80GroundTruth):
  """See the `Exp2VettedGroundTruthSpec` proto."""

  def __init__(self, spec: ground_truth_spec_pb2.Exp2VettedGroundTruthSpec):
    super().__init__()
    del spec
    self._wrapped_ground_truth = float80_ground_truth.Exp2VettedGroundTruth()


class WrappedExpeFloat80GroundTruth(WrappedFloat80GroundTruth):
  """See the `ExpeFloat80GroundTruthSpec` proto."""

  def __init__(self, spec: ground_truth_spec_pb2.ExpeFloat80GroundTruthSpec):
    super().__init__()
    del spec
    self._wrapped_ground_truth = float80_ground_truth.ExpeFloat80GroundTruth()


class WrappedLog2Float80GroundTruth(WrappedFloat80GroundTruth):
  """See the `Log2Float80GroundTruthSpec` proto."""

  def __init__(self, spec: ground_truth_spec_pb2.Log2Float80GroundTruthSpec):
    super().__init__()
    del spec
    self._wrapped_ground_truth = float80_ground_truth.Log2Float80GroundTruth()


class WrappedLogeFloat80GroundTruth(WrappedFloat80GroundTruth):
  """See the `LogeFloat80GroundTruthSpec` proto."""

  def __init__(self, spec: ground_truth_spec_pb2.LogeFloat80GroundTruthSpec):
    super().__init__()
    del spec
    self._wrapped_ground_truth = float80_ground_truth.LogeFloat80GroundTruth()


class WrappedErfFloat80GroundTruth(WrappedFloat80GroundTruth):
  """See the `ErfFloat80GroundTruthSpec` proto."""

  def __init__(self, spec: ground_truth_spec_pb2.ErfFloat80GroundTruthSpec):
    super().__init__()
    del spec
    self._wrapped_ground_truth = float80_ground_truth.ErfFloat80GroundTruth()


class WrappedErfcFloat80GroundTruth(WrappedFloat80GroundTruth):
  """See the `ErfcFloat80GroundTruthSpec` proto."""

  def __init__(self, spec: ground_truth_spec_pb2.ErfcFloat80GroundTruthSpec):
    super().__init__()
    del spec
    self._wrapped_ground_truth = float80_ground_truth.ErfcFloat80GroundTruth()


class WrappedWavyFloat80GroundTruth(WrappedFloat80GroundTruth):
  """See the `WavyFloat80GroundTruthSpec` proto."""

  def __init__(self, spec: ground_truth_spec_pb2.WavyFloat80GroundTruthSpec):
    super().__init__()
    if not spec.HasField("num_periods"):
      raise ValueError("Missing num_periods.")
    self._wrapped_ground_truth = float80_ground_truth.WavyFloat80GroundTruth(
        spec.num_periods
    )


class WrappedAiryFloat80GroundTruth(WrappedFloat80GroundTruth):
  """See the `AiryFloat80GroundTruthSpec` proto."""

  def __init__(self, spec: ground_truth_spec_pb2.AiryFloat80GroundTruthSpec):
    super().__init__()
    if not spec.HasField("input_scaling"):
      raise ValueError("Missing input_scaling.")
    self._wrapped_ground_truth = float80_ground_truth.AiryFloat80GroundTruth(
        spec.input_scaling
    )


class WrappedBesselFloat80GroundTruth(WrappedFloat80GroundTruth):
  """See the `BesselFloat80GroundTruthSpec` proto."""

  def __init__(self, spec: ground_truth_spec_pb2.BesselFloat80GroundTruthSpec):
    super().__init__()
    del spec
    self._wrapped_ground_truth = float80_ground_truth.BesselFloat80GroundTruth()


class Exp2ConvergentGroundTruth(GroundTruth):
  """2^x error calculation. See Exp2ConvergentGroundTruthSpec."""

  def __init__(self, spec: ground_truth_spec_pb2.Exp2ConvergentGroundTruthSpec):
    self._spec = spec
    self.ulp_bounds = [
        float80_ground_truth.Create_UlpBounds(
            min_input=0, max_input=1, min_output=1, max_output=2
        ),
        float80_ground_truth.Create_UlpBounds(
            min_input=1, max_input=2, min_output=2, max_output=4
        ),
    ]
    if self._spec.tolerance_fraction <= 0.0:
      raise ValueError("Missing or invalid tolerance_fraction.")
    self._tolerance_fraction = fractions.Fraction(self._spec.tolerance_fraction)
    if self._spec.start_taylor_terms < 3:
      raise ValueError("start_taylor_terms too small.")
    if self._spec.start_log2_steps < 3:
      raise ValueError("start_log2_steps too small.")
    if not self._spec.HasField("acceptable_error"):
      raise ValueError("Missing acceptable_error.")

  def labels(self, inputs: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError(
        "Not implemented yet. Consider using exp2_float80_ground_truth instead."
    )

  def signed_relative_errors(
      self,
      inputs: jnp.ndarray,
      predictions: jnp.ndarray,
      epsilon: JnpPreciseFloat,
      ulp_dtype: Optional[JnpFloatDType],
      override_ulp_with_value_at: Optional[float],
  ) -> jnp.ndarray:
    """Computes the element-wise error until convergence.

    See `Exp2ConvergentGroundTruthSpec`. Internal calculations are done with
    exact rationals.

    Args:
      inputs: see base class.
      predictions: see base class.
      epsilon: see base class.
      ulp_dtype: see base class.
      override_ulp_with_value_at: see base class.

    Returns:
      The signed errors. The return dtype is float64.
    """
    if override_ulp_with_value_at is not None:
      raise NotImplementedError("Not implemented yet.")
    _check_inputs_and_predictions(inputs, predictions)
    errors: List[float] = []
    for x, y in zip(inputs, predictions):
      errors.append(self._error(float(x), float(y), float(epsilon), ulp_dtype))
    return jnp.array(errors, dtype=JnpPreciseFloat)

  def _error(
      self,
      x: float,
      y: float,
      epsilon: float,
      ulp_dtype: Optional[JnpFloatDType],
  ) -> float:
    """Calculates the error for one scalar.

    Args:
      x: the input.
      y: the prediction.
      epsilon: a regularization term for the denominator.
      ulp_dtype: the dtype to use to determine the value of 1 ULP. If None, the
        result will measure the relative error w.r.t. the label instead of
        w.r.t. 1 ULP of the label.

    Returns:
      The relative error.

    Raises:
      RuntimeError: when there is an error
    """
    if math.isnan(y) or math.isinf(y):
      # Assume the label is not infinite, so this should give infinite error.
      return float("inf")
    rational_x = fractions.Fraction(x)
    rational_y = fractions.Fraction(y)
    epsilon = fractions.Fraction(epsilon)
    unsigned_errors = []  # The errors for each number of terms considered.
    for iter_num in range(self._spec.max_iters):
      label = _calculate_rational_exp2(
          num_steps=self._spec.start_log2_steps + iter_num,
          num_terms=self._spec.start_taylor_terms + iter_num,
          x=rational_x,
      )
      numerator = label - rational_y
      if ulp_dtype is None:
        denominator = abs(label)
      else:
        denominator = fractions.Fraction(
            ulp_util.ulp_float(
                float80_ground_truth.get_ulp_reference_point_for_input(
                    x, self.ulp_bounds
                ),
                dtype=ulp_dtype,
            )
        )
      denominator += epsilon
      error = numerator / denominator
      assert isinstance(error, fractions.Fraction)
      unsigned_errors.append(float(abs(error)))
      if self._has_converged(unsigned_errors):
        return float(error)
    raise RuntimeError("No convergence.")

  def _has_converged(self, errors: List[float]) -> bool:
    if len(errors) < self._spec.tolerance_iters:
      return False
    last_errors = errors[-self._spec.tolerance_iters :]
    lowest_error = min(last_errors)
    highest_error = max(last_errors)
    if highest_error <= self._spec.acceptable_error:
      assert lowest_error <= self._spec.acceptable_error
      return True
    return (
        highest_error - lowest_error
    ) / highest_error <= self._tolerance_fraction


def _calculate_rational_exp2(
    num_steps: int, num_terms: int, x: Fraction
) -> Fraction:
  # Let u = x log 2, then 2^x = e^u. We approximate e^u as a Taylor series
  # and log(2) as a continued fraction.
  log2_const = _calculate_rational_log2_constant(num_steps=num_steps)
  u = x * log2_const
  return _calculate_rational_exp(num_terms=num_terms, x=u)


def _calculate_rational_log2_constant(num_steps: int) -> Fraction:
  """Returns an approximation to log(2) based on a continued fraction."""
  one = Fraction(1, 1)
  two = Fraction(2, 1)
  k = Fraction(num_steps, 1)
  z = one / (two * k + one)
  while k > 0:
    z = one / (k / (k * z + two) + two * k - one)
    k -= one
  return z


def _calculate_rational_exp(num_terms: int, x: Fraction) -> Fraction:
  """Returns an approximation to exp(x) based on a Taylor series."""
  y = Fraction(0, 1)
  for index in range(num_terms):
    term = x**index / _factorial(index)
    y += term
  return y


def _factorial(n: int) -> Fraction:
  r = Fraction(1, 1)
  for i in range(n):
    r *= Fraction(i + 1, 1)
  return r


class RoughTwoXGroundTruth(GroundTruth):
  """2*x error calculation. See RoughTwoXGroundTruthSpec."""

  def __init__(self, spec: ground_truth_spec_pb2.RoughTwoXGroundTruthSpec):
    self._spec = spec

  def labels(self, inputs: jnp.ndarray) -> jnp.ndarray:
    two = jnp.array(2.0, dtype=JnpPreciseFloat)
    inputs = jnp.array(inputs, dtype=JnpPreciseFloat)
    labels = jnp.multiply(inputs, two)
    return labels

  def signed_relative_errors(
      self,
      inputs: jnp.ndarray,
      predictions: jnp.ndarray,
      epsilon: JnpPreciseFloat,
      ulp_dtype: Optional[JnpFloatDType],
      override_ulp_with_value_at: Optional[float],
  ) -> jnp.ndarray:
    """Computes the element-wise error until convergence.

    See `Exp2ConvergentGroundTruthSpec`. Internal calculations are done with
    exact rationals.

    Args:
      inputs: see base class.
      predictions: see base class.
      epsilon: see base class.
      ulp_dtype: see base class.
      override_ulp_with_value_at: see base class.

    Returns:
      The signed errors. The return dtype is float64.
    """
    if override_ulp_with_value_at is not None:
      raise NotImplementedError("Not implemented yet.")
    _check_inputs_and_predictions(inputs, predictions)
    labels = self.labels(inputs)
    predictions = jnp.array(predictions, dtype=JnpPreciseFloat)
    numerators = labels - predictions
    if ulp_dtype is None:
      denominators = jnp.abs(labels)
    else:
      denominators = ulp_util.ulp_array(labels, dtype=ulp_dtype)
    denominators += epsilon
    return jnp.divide(numerators, denominators)


def _check_inputs_and_predictions(
    inputs: jnp.ndarray, predictions: jnp.ndarray
):
  """Checks that the inputs and predictions are consistent."""
  if len(inputs.shape) != 1:
    raise ValueError("Invalid array dims.")
  if len(predictions.shape) != len(inputs.shape):
    raise ValueError("Inconsistent array dims.")
  if inputs.shape[0] == 0:
    raise ValueError("Invalid array size.")
  if inputs.shape[0] != predictions.shape[0]:
    raise ValueError("Inconsistent array size.")
  if predictions.dtype != inputs.dtype and predictions.dtype != JnpPreciseFloat:
    raise ValueError("Unusual predictions dtype.")
