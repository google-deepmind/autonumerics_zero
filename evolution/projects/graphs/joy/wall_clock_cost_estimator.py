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

"""Evaluation of graphs in the toy search space."""

import time
import timeit
from typing import List

import jax
from jax import numpy as jnp

from evolution.lib.python import printing
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import cost_estimator_impl
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import data_pb2
from evolution.projects.graphs.joy import graph_examples
from evolution.projects.graphs.joy import interpretation
from evolution.projects.graphs.joy import wall_clock_cost_estimator_spec_pb2
from evolution.projects.graphs.joy import wall_clock_reference

JnpPreciseFloat = data_lib.JnpPreciseFloat
LearnableParams = graph_lib.LearnableParams
ParameterizedFn = interpretation.ParameterizedFn
FinalizedFn = interpretation.FinalizedFn
WallClockCostEstimatorSpec = (
    wall_clock_cost_estimator_spec_pb2.WallClockCostEstimatorSpec
)


class WallClockCostEstimator(cost_estimator_impl.CostEstimatorImpl):
  """Measures execution time. See WallClockCostEstimatorSpec."""

  def __init__(self, spec: WallClockCostEstimatorSpec):
    """Builds the instance."""
    self._spec = spec
    if self._spec.dataset_dtype == data_pb2.NODTYPE:
      raise ValueError("Unknown dtype.")
    self._dtype = data_lib.get_dtype(self._spec.dataset_dtype)

    # The inputs on which to measure execution time.
    if (
        not self._spec.HasField("dataset_min")
        or not self._spec.HasField("dataset_max")
        or self._spec.dataset_min > self._spec.dataset_max
    ):
      raise ValueError("Invalid dataset limits.")
    if self._spec.dataset_size <= 0:
      raise ValueError("Invalid dataset size.")
    self._inputs = jnp.linspace(
        start=self._spec.dataset_min,
        stop=self._spec.dataset_max,
        num=self._spec.dataset_size,
        dtype=self._dtype,
    )

    # The time to normalize the cost.
    if not self._spec.HasField("reference_cost_type"):
      raise ValueError(
          "Wall-clock cost estimator does not specify a reference_cost_type."
      )
    self._reference = wall_clock_reference.WallClockReference(
        self._spec.reference_cost_type
    )
    printing.print_now(
        "Reference wall-clock cost = %.2f" % self.get_reference_absolute_cost()
    )

  def estimate(
      self, graph: graph_lib.Graph, learnable_params: LearnableParams
  ) -> float:
    """See base class."""
    parameterized_fn = interpretation.get_parameterized_fn(
        graph=graph, dtype=self._dtype
    )
    return self.estimate_parameterized_fn(
        parameterized_fn=parameterized_fn, learnable_params=learnable_params
    )

  def estimate_parameterized_fn(
      self, parameterized_fn: ParameterizedFn, learnable_params: LearnableParams
  ) -> float:
    """Measures the relative/normalized cost of a parameterized function."""
    data_lib.check_learnable_params_dtype(learnable_params, JnpPreciseFloat)
    finalized_fn = interpretation.get_finalized_fn(
        parameterized_fn=parameterized_fn, learnable_params=learnable_params
    )
    return self.estimate_finalized_fn(finalized_fn=finalized_fn)

  def estimate_finalized_fn(self, finalized_fn: FinalizedFn) -> float:
    """See base class."""
    absolute_cost = self._measure_absolute_cost_with_budget(
        finalized_fn=finalized_fn, budget_seconds=self._spec.budget_seconds
    )
    relative_cost = absolute_cost / self.get_reference_absolute_cost()
    if self._spec.HasField("min_cost") and relative_cost < self._spec.min_cost:
      relative_cost = self._spec.min_cost
    return 1.0 / relative_cost  # Speed.

  def _measure_absolute_cost_with_budget(
      self, finalized_fn: FinalizedFn, budget_seconds: float
  ) -> float:
    """Measures the absolute/unnormalized cost (execution time).

    Args:
      finalized_fn: the function to time.
      budget_seconds: a desired amount of time to spent evaluating repeats of
        the function to time.

    Returns:
      The time spent per call in nanoseconds.
    """
    if self._spec.HasField("python_for_stack"):
      jitted_finalized_fn = _stack_and_jit_with_python_for(
          spec=self._spec.python_for_stack,
          finalized_fn=finalized_fn,
          inputs=self._inputs,
      )
      stack_depth = self._spec.python_for_stack.depth
    elif self._spec.HasField("jax_while_stack"):
      jitted_finalized_fn = _stack_and_jit_with_jax_while(
          spec=self._spec.jax_while_stack,
          finalized_fn=finalized_fn,
          inputs=self._inputs,
      )
      stack_depth = self._spec.jax_while_stack.depth
    elif self._spec.HasField("no_stack"):
      jitted_finalized_fn = _stack_and_jit_with_no_stack(
          finalized_fn=finalized_fn,
          inputs=self._inputs,
      )
      stack_depth = 1
    else:
      raise NotImplementedError("Unsupported stack_type.")

    # Measure execution time.
    results = []
    start_time = time.time()
    elapsed_time = 0.0
    if self._spec.min_num_repeats < 1:
      raise ValueError("There must be at least 1 repeat.")
    num_repeats = self._spec.min_num_repeats
    if budget_seconds <= 0.0:
      raise ValueError("Invalid budget.")
    while elapsed_time < budget_seconds:
      results.extend(
          self._measure_absolute_cost_with_repeats(
              jitted_finalized_fn=jitted_finalized_fn,
              num_repeats=num_repeats,
              stack_depth=stack_depth,
          )
      )
      elapsed_time = time.time() - start_time
      if elapsed_time < budget_seconds / 10.0:
        num_repeats *= 2

    results = sorted(results)

    # Return the tightest bound.
    return min(results)

  def _measure_absolute_cost_with_repeats(
      self, jitted_finalized_fn: FinalizedFn, num_repeats: int, stack_depth: int
  ) -> List[float]:
    results = timeit.Timer(
        lambda: jitted_finalized_fn([self._inputs]).block_until_ready()
    ).repeat(repeat=num_repeats, number=1)
    results = [r / float(stack_depth) for r in results]  # Per call.
    results = [r * 1000000000.0 for r in results]  # Nanos.
    return results

  def measure_reference_absolute_cost(self):
    """Returns the absolute cost of the reference.

    This is not used during evolution. Instead, it is used in separate runs
    to calibrate the difference CPUs.
    """
    return self._measure_absolute_cost_with_budget(
        finalized_fn=graph_examples.get_reference8_finalized_fn(
            dtype=self._dtype
        ),
        budget_seconds=self._spec.reference_budget_seconds,
    )

  def get_reference_absolute_cost(self) -> float:
    return self._reference.get_reference_absolute_cost()


def _stack_and_jit_with_python_for(
    spec: wall_clock_cost_estimator_spec_pb2.PythonForStackSpec,
    finalized_fn: FinalizedFn,
    inputs: jnp.ndarray,
) -> FinalizedFn:
  """Stacks the function multiple times to make it more easily measurable."""
  if spec.depth <= 0:
    raise ValueError("Invalid stack depth.")
  if (
      not spec.HasField("min_value")
      or not spec.HasField("max_value")
      or spec.min_value > spec.max_value
  ):
    raise ValueError("Invalid stack limits.")

  def stacked_fn(x):
    assert len(x) == 1
    y = x[0]
    for _ in range(spec.depth):
      y = finalized_fn([y])
      y = jnp.maximum(y, spec.min_value)
      y = jnp.minimum(y, spec.max_value)
    return y

  stacked_fn = jax.jit(stacked_fn)

  # Run once to let JIT do the tracing.
  _ = stacked_fn([inputs]).block_until_ready()

  return stacked_fn


def _stack_and_jit_with_jax_while(
    spec: wall_clock_cost_estimator_spec_pb2.JaxWhileStackSpec,
    finalized_fn: FinalizedFn,
    inputs: jnp.ndarray,
) -> FinalizedFn:
  """Stacks the function multiple times to make it more easily measurable."""
  if spec.depth <= 0:
    raise ValueError("Invalid stack depth.")
  if (
      not spec.HasField("min_value")
      or not spec.HasField("max_value")
      or spec.min_value > spec.max_value
  ):
    raise ValueError("Invalid stack limits.")

  def to_stack_fn(z):
    x = finalized_fn([z["value"]])
    x = jnp.maximum(x, spec.min_value)
    x = jnp.minimum(x, spec.max_value)
    z["value"] = x
    z["step"] = z["step"] + 1
    return z

  def stack_cond_fn(z):
    return z["step"] < spec.depth

  def stacked_fn(x):
    assert len(x) == 1
    z = {"value": x[0], "step": jnp.int64(0)}
    result = jax.lax.while_loop(
        cond_fun=stack_cond_fn, body_fun=to_stack_fn, init_val=z
    )
    return result["value"]

  stacked_fn = jax.jit(stacked_fn)

  # Run once to let JIT do the tracing.
  _ = stacked_fn([inputs]).block_until_ready()

  return stacked_fn


def _stack_and_jit_with_no_stack(
    finalized_fn: FinalizedFn,
    inputs: jnp.ndarray,
) -> FinalizedFn:

  fn = jax.jit(finalized_fn)
  # Run once to let JIT do the tracing.
  _ = fn([inputs]).block_until_ready()
  return fn
