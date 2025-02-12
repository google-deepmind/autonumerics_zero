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

"""Utilities for evaluation and hashing."""

from typing import List

import jax
from jax import numpy as jnp

from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import dataset as dataset_lib
from evolution.projects.graphs.joy import error_util
from evolution.projects.graphs.joy import interpretation
from evolution.projects.graphs.joy import standard_validator_spec_pb2
from evolution.projects.graphs.joy import validator as validator_lib

LearnableParams = graph_lib.LearnableParams
JnpFloatDType = data_lib.JnpFloatDType
JnpFloat = data_lib.JnpFloat
JnpPreciseFloat = data_lib.JnpPreciseFloat
JnpKey = data_lib.JnpKey


class StandardValidator(validator_lib.Validator):
  """Measures the quality objective(s)."""

  def __init__(self, spec: standard_validator_spec_pb2.StandardValidatorSpec):
    self._spec = spec
    if not self._spec.HasField("dataset"):
      raise ValueError("Missing dataset.")
    self._dataset = dataset_lib.build(self._spec.dataset)

    if not self._spec.HasField("quality_fitness"):
      raise ValueError("Missing quality fitness.")
    self._quality_fitness_fn = error_util.get_quality_fitness_fn(
        self._spec.quality_fitness
    )

  def validate(
      self, graph: graph_lib.Graph, learnable_params: graph_lib.LearnableParams
  ) -> List[JnpPreciseFloat]:
    """Performs the validation, after JIT compiling.

    Args:
      graph: the graph to validate.
      learnable_params: the parameters to use with the graph.

    Returns:
      The quality objective(s) value(s).
    """
    data_lib.check_learnable_params_dtype(learnable_params, JnpPreciseFloat)
    parameterized_fn = interpretation.get_parameterized_fn(
        graph=graph, dtype=self._dataset.inputs_dtype
    )
    finalized_fn = interpretation.get_finalized_fn(
        parameterized_fn=parameterized_fn, learnable_params=learnable_params
    )
    quality_fitnesses = self.validate_finalized_fn(finalized_fn)
    return quality_fitnesses

  def validate_finalized_fn(
      self, finalized_fn: interpretation.FinalizedFn
  ) -> List[JnpPreciseFloat]:
    """Performs the validation.

    Args:
      finalized_fn: the function to validate. It is expected to match the dtype
        provided. If created from a graph, the learnable_parameters must have
        been casted to this dtype.

    Returns:
      The quality objective(s) value(s).

    Raises:
      RuntimeError: if there's an error.
    """
    inputs = self._dataset.inputs()

    # Compile.
    finalized_fn = jax.jit(finalized_fn)
    # Sometimes the uncompiled function evaluates differently, so we ensure
    # JIT is done by forcing one evaluation.
    _ = finalized_fn(inputs).block_until_ready()

    predictions = finalized_fn(inputs)
    if predictions.dtype != self._dataset.inputs_dtype:
      raise RuntimeError("Predictions dtype mismatch.")
    predictions = jnp.array(predictions, dtype=JnpPreciseFloat)
    inputs = jnp.array(inputs, dtype=JnpPreciseFloat)
    quality_fitness = self._predictions_to_quality_fitness(inputs, predictions)
    assert quality_fitness.dtype == JnpPreciseFloat
    return [quality_fitness]

  def _predictions_to_quality_fitness(
      self, inputs: jnp.ndarray, predictions: jnp.ndarray
  ) -> JnpPreciseFloat:
    """Converts predictions to the quality fitness. Useful for mocking."""
    error = self._dataset.max_relative_error(  # pytype: disable=wrong-arg-types  # jax-ndarray
        inputs=inputs, predictions=[predictions]
    )
    assert error.dtype == JnpPreciseFloat
    quality_fitness = self._quality_fitness_fn(error)
    return quality_fitness
