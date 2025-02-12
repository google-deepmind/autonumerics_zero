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

"""JAX-compatible dataset classes.

These are meant to be jit-compiled or auto-differentiated, but are constrained
to float64 or less precision. For more precision, see the `Dataset` classes.
"""

from typing import List, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import jax.random as jrandom
from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.lib.python import rng as rng_lib
from evolution.projects.graphs.joy import bfloat16_util
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import data_pb2
from evolution.projects.graphs.joy import ground_truth as ground_truth_lib
from evolution.projects.graphs.joy import jax_dataset_spec_pb2
from evolution.projects.graphs.joy import ulp_util

JnpKey = data_lib.JnpKey
JnpPreciseFloat = data_lib.JnpPreciseFloat


class JaxDataset:
  """Interface for datasets.

  Subclasses and users must adhere to the following contract when JIT compiling
  methods of this class.

  JIT CONTRACT for users: you can JIT compile (or include in code that is JIT
  compiled) any of the methods in this class, except the constructor,
  the `next_jnp_key` method, and the `reseed_jnp_key` method.

  JIT CONTRACT for subclasses: subclasses cannot hold any mutable state. The
  constructor can set up a state that will never be changed by any of the
  methods. E.g. the constructor can build the inputs and labels and then the
  data method can edit and return them, but the data method cannot modify
  the stored inputs and labels.
  """

  def __init__(self, rng_seed: Optional[int]):
    """Initializes the instance."""
    self._rng_seed = rng_seed
    del rng_seed
    if self._rng_seed is None:
      self._key = None
    else:
      # Generate the initial key for random sampling.
      if self._rng_seed > 0:
        self._key = jrandom.PRNGKey(self._rng_seed)
      else:
        self._key = jrandom.PRNGKey(rng_lib.GenerateRNGSeed())

  @property
  def is_subsampling(self) -> bool:
    """Whether the inputs, labels and label_ulps will be different each time."""
    raise NotImplementedError("Must be implemented by subclass")

  @property
  def inputs_dtype(self) -> data_lib.JnpFloatDType:
    """The dtype of the inputs that are produced by this class.

    Does not refer to the dtype of labels or error computation, which may
    be higher.

    Returns:
      A JNP dtype.
    """
    raise NotImplementedError("Must be implemented by subclass")

  def reseed_jnp_key(self):
    """Re-seeds the random key produced by `next_jnp_key`.

    This function should not be jit-compiled, because it stores a state in the
    class instance. See the "JIT CONTRACT" section of the class-level docstring.
    """
    if self._rng_seed is None:
      return  # No RNG seed, nothing to do.
    if self._rng_seed <= 0:
      return  # No need to reseed as the seeds are random.
    self._key = jrandom.PRNGKey(self._rng_seed)

  def next_jnp_key(
      self, batch_size: Optional[int] = None
  ) -> Optional[Union[JnpKey, jnp.ndarray]]:
    """Provides the next random key to pass to `data`.

    This function should not be jit-compiled, because it stores a state in the
    class instance. See the "JIT CONTRACT" section of the class-level docstring.

    Args:
      batch_size: the size of the batch of keys to return. If `None`, returns a
        single key.

    Returns:
      A jnp key or a batch of jnp keys.
    """
    if self._key is None:
      return None
    else:
      self._key, subkey = jrandom.split(self._key)
      if batch_size is None:
        return subkey
      else:
        return jrandom.split(subkey, batch_size)

  def data(
      self, jnp_key: Optional[JnpKey]
  ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], Optional[List[jnp.ndarray]]]:
    """Returns the labels corresponding to the last set of inputs.

    Args:
      jnp_key: the random key to use for subsampling. Can be `None` if this
        dataset does not use subsampling.

    Returns:
      An (inputs, labels, label_ulps) tuple, where
      inputs: the inputs as a list of arrays. Each item in the list corresponds
        to a graph input. A graph with one input (e.g. "x") needs a list of
        length 1. The elemetns of the array are input values for the different
        examples. Therefore, every array must be of the same length (the number
        of examples).
      labels: the labels as a list of arrays. Each item in the list corresponds
        to a graph output. A graph with one output (e.g. "f") needs a list of
        length 1. The elements of the array are output values for the different
        examples. Therefore, all arrays must be of the same length (the number
        of examples). Each element of the arrays is in correspondence with the
        inputs at the same position. All arrays are of JnpPreciseFloat dtype.
      label_ulps: an array with the same structure as `labels`, where each
        value corresponds to 1 ULP of the corresponding label. All values
        are of the JnpPreciseFloat dtype. Can be `None` if ULP measurement is
        not used.
    """
    raise NotImplementedError("Must be implemented by subclass.")

  def signed_relative_errors(
      self,
      predictions: List[jnp.ndarray],
      labels: List[jnp.ndarray],
      label_ulps: Optional[List[jnp.ndarray]],
  ) -> List[jnp.ndarray]:
    """Calculates the signed relative errors.

    The error is calculated at float64 precision.

    Args:
      predictions: the outputs, generated by evaluating a graph. The fornat is a
        list of arrays. Each item in the list corresponds to a graph output. A
        graph with one output (e.g. "f") needs a list of length 1. The elements
        of the array are output values for the different examples. Therefore,
        all arrays must be of the same length (the number of examples). Each
        element of the arrays is in correspondence with the inputs at the same
        position. Must be of the inputs_dtype.
      labels: the corresponding expected outputs. Must be of the JnpPreciseFloat
        dtype.
      label_ulps: the corresponding label ULPs. Must be of the JnpPreciseFloat
        dtype. Can be `None` if not measuring ULPs.

    Returns:
      The errors arrays, with float64 dtype. The format is the same as that of
      the `predictions` arg.
    """
    raise NotImplementedError("Must be implemented by subclass.")

  def max_relative_error(
      self,
      predictions: List[jnp.ndarray],
      labels: List[jnp.ndarray],
      label_ulps: Optional[List[jnp.ndarray]],
  ) -> JnpPreciseFloat:
    """Calculates the max relative error.

    The labels are those corresponding to the last set of inputs. The labels may
    be calculated at higher than float64 precision, but the error is calculated
    at float64 precision.

    Args:
      predictions: see `signed_relative_errors`.
      labels: see `signed_relative_errors`.
      label_ulps: see `signed_relative_errors`.

    Returns:
      The error.
    """
    raise NotImplementedError("Must be implemented by subclass.")


def build(spec: jax_dataset_spec_pb2.JaxDatasetSpec) -> JaxDataset:
  if spec.HasField("general"):
    return GeneralJaxDataset(spec=spec.general)
  elif spec.HasField("full_bfloat16"):
    return FullBfloat16JaxDataset(spec=spec.full_bfloat16)
  else:
    raise NotImplementedError("Unknown dataset.")


class GeneralJaxDataset(JaxDataset):
  """Please see the base class docstring, especially the JIT CONTRACT note."""

  def __init__(self, spec: jax_dataset_spec_pb2.GeneralJaxDatasetSpec):
    """Initializes the instance."""
    rng_seed = None
    if spec.HasField("input_subsampling"):
      rng_seed = _get_rng_seed(spec.input_subsampling)
    super().__init__(rng_seed=rng_seed)
    self._spec = spec

    # Calculate the fixed inputs.
    if self._spec.num_inputs <= 0:
      raise ValueError("Missing or invalid inputs.")
    if not self._spec.HasField("inputs_min"):
      raise ValueError("Missing inputs_min.")
    if not self._spec.HasField("inputs_max"):
      raise ValueError("Missing inputs_max.")
    if self._spec.inputs_min > self._spec.inputs_max:
      raise ValueError("Inconsistent input limits.")
    if not self._spec.HasField("inputs_dtype"):
      raise ValueError("Missing inputs_dtype.")
    if self._spec.inputs_dtype == data_pb2.BFLOAT16:
      raise ValueError(
          "Incorrect Dataset class used. Please use FullBfloat16Dataset"
      )
    self._inputs_dtype = data_lib.get_dtype(self._spec.inputs_dtype)
    num_inputs = self._spec.num_inputs
    self._inputs = jnp.linspace(
        start=self._spec.inputs_min,
        stop=self._spec.inputs_max,
        num=num_inputs,
        dtype=self._inputs_dtype,
    )
    if not self._spec.inputs_min_inclusive:
      if num_inputs < 2:
        raise ValueError("Too few points.")
      self._inputs = self._inputs[1:]
      num_inputs -= 1
    if not self._spec.inputs_max_inclusive:
      if num_inputs < 2:
        raise ValueError("Too few points.")
      self._inputs = self._inputs[:-1]
      num_inputs -= 1

    # Calculate the fixed labels.
    if not self._spec.HasField("ground_truth"):
      raise ValueError("Missing ground_truth.")
    ground_truth = ground_truth_lib.build(self._spec.ground_truth)
    self._labels = ground_truth.labels(inputs=self._inputs)

    if self._spec.relative_error_epsilon < 0.0:
      raise ValueError("Found negative relative_error_epsilon.")
    if not self._spec.HasField("measure_ulp"):
      raise ValueError("Missing measure_ulp.")
    if self._spec.measure_ulp:
      if self._spec.HasField("override_ulp_dtype"):
        ulp_dtype = data_lib.get_dtype(self._spec.override_ulp_dtype)
      else:
        ulp_dtype = self._inputs_dtype
      if self._spec.HasField("override_ulp_with_value_at"):
        override_ulp_value = ulp_util.ulp_float(
            self._spec.override_ulp_with_value_at, dtype=ulp_dtype
        )
        self._label_ulps = jnp.full_like(
            self._labels, override_ulp_value, dtype=JnpPreciseFloat
        )
      else:
        self._label_ulps = ulp_util.ulp_array(self._labels, dtype=ulp_dtype)
    else:
      assert not self._spec.HasField("override_ulp_dtype")

    self._subsampled_data_fn = self._get_subsampled_data_fn()

  @property
  def spec(self) -> jax_dataset_spec_pb2.GeneralJaxDatasetSpec:
    """Returns the spec."""
    return self._spec

  @property
  def is_subsampling(self) -> bool:
    """See base class."""
    return self._spec.HasField("input_subsampling")

  @property
  def inputs_dtype(self) -> data_lib.JnpFloatDType:
    """See base class."""
    return self._inputs_dtype

  def data(
      self, jnp_key: Optional[JnpKey]
  ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], Optional[List[jnp.ndarray]]]:
    """See base class."""
    if self._spec.HasField("input_subsampling"):
      assert jnp_key is not None
      return self._subsampled_data_fn(jnp_key)
    else:
      label_ulps = [self._label_ulps] if self._spec.measure_ulp else None
      return [self._inputs], [self._labels], label_ulps

  def _get_subsampled_data_fn(self):
    """Returns the compiled data-subsampling function."""

    def subsampled_data_fn(jnp_key):
      """Subsamples the inputs, labels, and label_ulps before returning them.

      Args:
        jnp_key: the jax numpy key for random sampling.

      Returns:
        Subsampled inputs, labels, and label_ulps, with the same structure as
        in the return value of the `data` method.
      """
      if not self._spec.input_subsampling.HasField("num_subsample_inputs"):
        raise ValueError("Missing num_subsample_inputs.")
      if self._spec.input_subsampling.num_subsample_inputs >= len(self._inputs):
        raise ValueError("Invalid num_subsample_inputs.")
      random_indices = jrandom.randint(
          key=jnp_key,
          minval=0,
          maxval=len(self._inputs),
          shape=(self._spec.input_subsampling.num_subsample_inputs,),
      )
      subsampled_inputs = [self._inputs[random_indices]]
      subsampled_labels = [self._labels[random_indices]]
      if self._spec.measure_ulp:
        subsampled_label_ulps = [self._label_ulps[random_indices]]
      else:
        subsampled_label_ulps = None
      return subsampled_inputs, subsampled_labels, subsampled_label_ulps

    subsampled_data_fn = jax.jit(subsampled_data_fn)
    return subsampled_data_fn

  def signed_relative_errors(
      self,
      predictions: List[jnp.ndarray],
      labels: List[jnp.ndarray],
      label_ulps: Optional[List[jnp.ndarray]],
  ) -> List[jnp.ndarray]:
    """See base class."""
    if len(predictions) != 1:
      raise NotImplementedError("Not yet implemented.")
    predictions = predictions[0]
    if len(labels) != 1:
      raise NotImplementedError("Not yet implemented.")
    labels = labels[0]
    if self._spec.measure_ulp:
      if label_ulps is None:
        raise ValueError("label_ulps required.")
      if len(label_ulps) != 1:
        raise NotImplementedError("Not yet implemented.")
      label_ulps = label_ulps[0] if label_ulps is not None else None
    else:
      if label_ulps is not None:
        raise ValueError("Not measuring ULPs.")

    if predictions.shape != labels.shape:
      raise ValueError("Incompatible shapes.")
    if self._spec.measure_ulp:
      assert label_ulps is not None  # For pylint.
      if label_ulps.shape != labels.shape:
        raise ValueError("Incompatible shapes.")
    if predictions.dtype != JnpPreciseFloat:
      predictions = jnp.array(predictions, dtype=JnpPreciseFloat)
    if labels.dtype != JnpPreciseFloat:
      raise ValueError("`labels` must be of JnpPreciseFloat dtype.")
    if self._spec.measure_ulp:
      assert label_ulps is not None  # For pylint.
      if label_ulps.dtype != JnpPreciseFloat:
        raise ValueError("`label_ulps` must be of JnpPreciseFloat dtype.")

    numerators = jnp.subtract(labels, predictions)
    if self._spec.measure_ulp:
      denominators = label_ulps
    else:
      denominators = jnp.abs(labels)
    denominators += JnpPreciseFloat(self._spec.relative_error_epsilon)
    return [jnp.divide(numerators, denominators)]  # pytype: disable=wrong-arg-types  # jnp-types

  def max_relative_error(
      self,
      predictions: List[jnp.ndarray],
      labels: List[jnp.ndarray],
      label_ulps: Optional[List[jnp.ndarray]],
  ) -> JnpPreciseFloat:
    """See base class."""
    errors = self.signed_relative_errors(
        predictions=predictions, labels=labels, label_ulps=label_ulps
    )
    assert len(errors) == 1
    return jnp.max(jnp.abs(errors[0]))


class FullBfloat16JaxDataset(JaxDataset):
  """Please see the base class docstring, especially the JIT CONTRACT note."""

  def __init__(self, spec: jax_dataset_spec_pb2.FullBfloat16JaxDatasetSpec):
    """Initializes the instance."""
    rng_seed = None
    if spec.HasField("input_subsampling"):
      rng_seed = _get_rng_seed(spec.input_subsampling)
    super().__init__(rng_seed=rng_seed)
    self._spec = spec

    # Calculate the fixed inputs.
    if not self._spec.HasField("inputs_min"):
      raise ValueError("Missing inputs_min.")
    if not self._spec.HasField("inputs_max"):
      raise ValueError("Missing inputs_max.")
    if self._spec.inputs_min > self._spec.inputs_max:
      raise ValueError("Inconsistent input limits.")
    self._inputs_dtype = jnp.bfloat16

    self._inputs = bfloat16_util.all_bfloat16_values(
        self._spec.inputs_min,
        self._spec.inputs_max,
        self._spec.skip_every,
        self._spec.inputs_min_inclusive,
        self._spec.inputs_max_inclusive,
    )

    # Calculate the fixed labels.
    if not self._spec.HasField("ground_truth"):
      raise ValueError("Missing ground_truth.")
    ground_truth = ground_truth_lib.build(self._spec.ground_truth)
    self._labels = ground_truth.labels(inputs=self._inputs)

    if self._spec.relative_error_epsilon < 0.0:
      raise ValueError("Found negative relative_error_epsilon.")
    if not self._spec.HasField("measure_ulp"):
      raise ValueError("Missing measure_ulp.")
    if self._spec.measure_ulp:
      self._label_ulps = ulp_util.ulp_array(
          self._labels, dtype=self._inputs_dtype
      )

    self._subsampled_data_fn = self._get_subsampled_data_fn()

  @property
  def spec(self) -> jax_dataset_spec_pb2.FullBfloat16JaxDatasetSpec:
    """Returns the spec."""
    return self._spec

  @property
  def is_subsampling(self) -> bool:
    """See base class."""
    return self._spec.HasField("input_subsampling")

  @property
  def inputs_dtype(self) -> data_lib.JnpFloatDType:
    """See base class."""
    return self._inputs_dtype

  def data(
      self, jnp_key: Optional[JnpKey]
  ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], Optional[List[jnp.ndarray]]]:
    """See base class."""
    if self._spec.HasField("input_subsampling"):
      assert jnp_key is not None
      return self._subsampled_data_fn(jnp_key)
    else:
      label_ulps = [self._label_ulps] if self._spec.measure_ulp else None
      return [self._inputs], [self._labels], label_ulps

  def _get_subsampled_data_fn(self):
    """Returns the compiled data-subsampling function."""

    def subsampled_data_fn(jnp_key):
      """Subsamples the inputs, labels, and label_ulps before returning them.

      Args:
        jnp_key: the jax numpy key for random sampling.

      Returns:
        Subsampled inputs, labels, and label_ulps, with the same structure as
        in the return value of the `data` method.
      """
      if not self._spec.input_subsampling.HasField("num_subsample_inputs"):
        raise ValueError("Missing num_subsample_inputs.")
      if self._spec.input_subsampling.num_subsample_inputs >= len(self._inputs):
        raise ValueError("Invalid num_subsample_inputs.")
      random_indices = jrandom.randint(
          key=jnp_key,
          minval=0,
          maxval=len(self._inputs),
          shape=(self._spec.input_subsampling.num_subsample_inputs,),
      )
      subsampled_inputs = [self._inputs[random_indices]]
      subsampled_labels = [self._labels[random_indices]]
      if self._spec.measure_ulp:
        subsampled_label_ulps = [self._label_ulps[random_indices]]
      else:
        subsampled_label_ulps = None
      return subsampled_inputs, subsampled_labels, subsampled_label_ulps

    subsampled_data_fn = jax.jit(subsampled_data_fn)
    return subsampled_data_fn

  def signed_relative_errors(
      self,
      predictions: List[jnp.ndarray],
      labels: List[jnp.ndarray],
      label_ulps: Optional[List[jnp.ndarray]],
  ) -> List[jnp.ndarray]:
    """See base class."""
    if len(predictions) != 1:
      raise NotImplementedError("Not yet implemented.")
    predictions = predictions[0]
    if len(labels) != 1:
      raise NotImplementedError("Not yet implemented.")
    labels = labels[0]
    if self._spec.measure_ulp:
      if label_ulps is None:
        raise ValueError("label_ulps required.")
      if len(label_ulps) != 1:
        raise NotImplementedError("Not yet implemented.")
      label_ulps = label_ulps[0] if label_ulps is not None else None
    else:
      if label_ulps is not None:
        raise ValueError("Not measuring ULPs.")

    if predictions.shape != labels.shape:
      raise ValueError("Incompatible shapes.")
    if self._spec.measure_ulp:
      assert label_ulps is not None  # For pylint.
      if label_ulps.shape != labels.shape:
        raise ValueError("Incompatible shapes.")
    if predictions.dtype != JnpPreciseFloat:
      predictions = jnp.array(predictions, dtype=JnpPreciseFloat)
    if labels.dtype != JnpPreciseFloat:
      raise ValueError("`labels` must be of JnpPreciseFloat dtype.")
    if self._spec.measure_ulp:
      assert label_ulps is not None  # For pylint.
      if label_ulps.dtype != JnpPreciseFloat:
        raise ValueError("`label_ulps` must be of JnpPreciseFloat dtype.")

    numerators = jnp.subtract(labels, predictions)
    if self._spec.measure_ulp:
      denominators = label_ulps
    else:
      denominators = jnp.abs(labels)
    denominators += JnpPreciseFloat(self._spec.relative_error_epsilon)
    return [jnp.divide(numerators, denominators)]  # pytype: disable=wrong-arg-types  # jnp-types

  def max_relative_error(
      self,
      predictions: List[jnp.ndarray],
      labels: List[jnp.ndarray],
      label_ulps: Optional[List[jnp.ndarray]],
  ) -> JnpPreciseFloat:
    """See base class."""
    errors = self.signed_relative_errors(
        predictions=predictions, labels=labels, label_ulps=label_ulps
    )
    assert len(errors) == 1
    return jnp.max(jnp.abs(errors[0]))


def _get_rng_seed(
    spec: jax_dataset_spec_pb2.InputSubsamplingSpec,
) -> Optional[int]:
  if not spec.HasField("rng_seed"):
    raise ValueError("Missing rng_seed.")
  if spec.rng_seed <= 0 and spec.rng_seed != -1:
    raise ValueError("rng_seed should be positive or -1.")
  return spec.rng_seed
