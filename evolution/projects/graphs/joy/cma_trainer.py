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

import time
from typing import Optional
from evojax import algo as evojax_algo
import jax
from jax import numpy as jnp
from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import cma_trainer_spec_pb2
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import early_stopping
from evolution.projects.graphs.joy import error_util
from evolution.projects.graphs.joy import interpretation
from evolution.projects.graphs.joy import jax_dataset as jax_dataset_lib
from evolution.projects.graphs.joy import trainer as trainer_lib

LearnableParams = graph_lib.LearnableParams
JnpFloat = data_lib.JnpFloat
JnpPreciseFloat = data_lib.JnpPreciseFloat
JnpKey = data_lib.JnpKey


class CmaTrainer(trainer_lib.Trainer):
  """Fine-tunes the coefficients using CMA-ES."""

  def __init__(self, spec: cma_trainer_spec_pb2.CmaTrainerSpec):
    self._spec = spec
    if not self._spec.HasField("jax_dataset"):
      raise ValueError("Missing dataset.")
    self._jax_dataset = jax_dataset_lib.build(self._spec.jax_dataset)

    if not self._spec.HasField("quality_fitness"):
      raise ValueError("Missing quality fitness.")
    self._quality_fitness_fn = error_util.get_quality_fitness_fn(
        self._spec.quality_fitness
    )

    if not self._spec.HasField("generations"):
      raise ValueError("Missing generations.")
    if self._spec.population_size < 2:
      raise ValueError("Population size too small.")
    if self._spec.init_scale < 0.0:
      raise ValueError("Init scale too small.")
    if self._spec.HasField("early_stopper"):
      self._early_stopper = CmaEarlyStopper(self._spec.early_stopper)
    elif self._spec.HasField("early_stopping"):
      self._early_stopper = early_stopping.EarlyStopping(
          self._spec.early_stopping
      )
    else:
      self._early_stopper = None
    self._latest_time: Optional[float] = None
    self._latest_executions: Optional[int] = None

  def train(
      self,
      graph: graph_lib.Graph,
      learnable_params: LearnableParams,
      jnp_key: JnpKey,
  ) -> LearnableParams:
    """Optimizes the learnable parameters using a simple ES."""
    parameterized_fn = interpretation.get_parameterized_fn(
        graph=graph, dtype=self._jax_dataset.inputs_dtype
    )
    return self.train_parameterized_fn(
        parameterized_fn=parameterized_fn,
        learnable_params=learnable_params,
        jnp_key=jnp_key,
    )

  def train_parameterized_fn(
      self,
      parameterized_fn: interpretation.ParameterizedFn,
      learnable_params: LearnableParams,
      jnp_key: JnpKey,
  ) -> LearnableParams:
    """Optimizes the learnable parameters using a simple ES."""
    data_lib.check_learnable_params_dtype(learnable_params, JnpPreciseFloat)
    self._latest_time = None
    self._latest_executions = None
    sorted_param_keys = sorted(learnable_params.keys())
    if len(sorted_param_keys) < 2:
      # Cannot use this implementation of CMA-ES to learn only 1 parameter.
      return learnable_params
    measure_fitness_fn = self._get_measure_fitness_fn(
        parameterized_fn=parameterized_fn, sorted_param_keys=sorted_param_keys
    )
    batch_measure_fitness_fn = self._get_batch_measure_fitness_fn(
        measure_fitness_fn=measure_fitness_fn
    )

    learnable_params_array = jnp.array(
        [learnable_params[k] for k in sorted_param_keys], dtype=JnpPreciseFloat
    )
    learnable_params_array = self._cma_es(
        measure_fitness_fn=measure_fitness_fn,
        batch_measure_fitness_fn=batch_measure_fitness_fn,
        init_params_array=learnable_params_array,
    )
    assert learnable_params_array.dtype == JnpPreciseFloat
    learnable_params = {
        k: jnp.array(v, dtype=JnpPreciseFloat)
        for k, v in zip(sorted_param_keys, learnable_params_array)
    }

    return learnable_params

  def retrieve_latest_time(self) -> Optional[float]:
    """See base class."""
    return self._latest_time

  def retrieve_latest_executions(self) -> Optional[int]:
    """See base class."""
    return self._latest_executions

  def _get_measure_fitness_fn(self, parameterized_fn, sorted_param_keys):
    """Returns the function that measures the fitness of a set of params."""

    def measure_fitness_fn(
        params_array: jnp.ndarray, jnp_key: JnpKey
    ) -> data_lib.JnpFloat:
      """Measures the fitnesses with the given parameters."""
      params_dict = {k: v for k, v in zip(sorted_param_keys, params_array)}
      finalized_fn = interpretation.get_finalized_fn(
          parameterized_fn=parameterized_fn, learnable_params=params_dict
      )
      inputs, labels, label_ulps = self._jax_dataset.data(jnp_key)
      predictions = finalized_fn(inputs)
      assert predictions.dtype == self._jax_dataset.inputs_dtype
      predictions = [jnp.array(predictions, dtype=JnpPreciseFloat)]
      error = self._jax_dataset.max_relative_error(
          predictions=predictions, labels=labels, label_ulps=label_ulps
      )
      assert error.dtype == JnpPreciseFloat
      quality_fitness = self._quality_fitness_fn(error)
      assert quality_fitness.dtype == JnpPreciseFloat
      return quality_fitness

    measure_fitness_fn = jax.jit(measure_fitness_fn)
    return measure_fitness_fn

  def _get_batch_measure_fitness_fn(self, measure_fitness_fn):
    """Returns the batched version of function that measures the fitness."""
    batch_measure_fitness_fn = jax.vmap(measure_fitness_fn)
    batch_measure_fitness_fn = jax.jit(batch_measure_fitness_fn)
    return batch_measure_fitness_fn

  def _cma_es(
      self,
      measure_fitness_fn,
      batch_measure_fitness_fn,
      init_params_array: jnp.ndarray,
  ) -> jnp.ndarray:
    """Applies CMA-ES to optimize the parameters.

    Args:
      measure_fitness_fn: a function from parameters (jnp.ndarray) to fitness
        (jnp float). The parameters must be of dtype `JnpPreciseFloat`. This
        function will return a fitness of dtype `JnpPreciseFloat`.
      batch_measure_fitness_fn: a batched version of measure_fitness_fn.
      init_params_array: a jnp.ndarray with the initial parameters. It must be
        of dtype `JnpPreciseFloat`.

    Returns:
      A jnp.ndarray with the optimized parameters. This vector will be of the
      same shape as `init_params` and must be of dtype `JnpPreciseFloat`.
    """
    max_iters = self._spec.generations
    pop_size = self._spec.population_size
    init_stdev = self._spec.init_scale
    init_mean = init_params_array
    param_size = init_params_array.shape[0]

    solver_cls = evojax_algo.CMA_ES_JAX
    solver = solver_cls(
        param_size=param_size,
        pop_size=pop_size,
        mean=init_mean,
        init_stdev=init_stdev,
    )

    # Push calls to astype() into JIT-ed functions, which is important for
    # performance.
    batch_measure_fitness_fn0 = batch_measure_fitness_fn
    batch_measure_fitness_fn = jax.jit(
        lambda p, k: batch_measure_fitness_fn0(p.astype(JnpPreciseFloat), k)
    )
    measure_fitness_fn0 = measure_fitness_fn
    measure_fitness_fn = jax.jit(
        lambda p, k: measure_fitness_fn0(p.astype(JnpPreciseFloat), k)
    )

    @jax.jit
    def _update_best_by_batch(best_x, best_v, xs, vs):
      index = jnp.argmax(vs)
      this_best_x = xs[index]
      this_best_v = vs[index]
      return jax.lax.cond(
          pred=(this_best_v > best_v),
          true_fun=(lambda: (this_best_x, this_best_v)),
          false_fun=(lambda: (best_x, best_v)),
      )

    @jax.jit
    def _update_best(best_x, best_v, x, v):
      return jax.lax.cond(
          pred=(v > best_v),
          true_fun=(lambda: (x, v)),
          false_fun=(lambda: (best_x, best_v)),
      )

    iters = 0
    start_time = self._time_now()
    self._jax_dataset.reseed_jnp_key()
    best_x, best_v = None, None
    if self._early_stopper:
      self._early_stopper.start()
    while iters < max_iters:
      # ask-eval-tell loop.
      xs = solver.ask()
      batch_size = jnp.shape(xs)[0]
      jnp_keys = self._jax_dataset.next_jnp_key(batch_size)
      vs = batch_measure_fitness_fn(xs, jnp_keys)
      solver.tell(vs)

      # update best parameters/values
      if best_v is None:
        best_x, best_v = xs[0], vs[0]
      best_x, best_v = _update_best_by_batch(best_x, best_v, xs, vs)
      x = solver.get_best_params_ref()
      v = measure_fitness_fn(x, self._jax_dataset.next_jnp_key())
      best_x, best_v = _update_best(best_x, best_v, x, v)

      iters += 1

      if self._early_stopper and self._early_stopper.should_stop(iters, best_v):
        break
    self._latest_time = self._time_now() - start_time
    self._latest_executions = iters * pop_size

    optimized_params_array = best_x
    assert optimized_params_array is not None
    optimized_params_array = optimized_params_array.astype(JnpPreciseFloat)
    return optimized_params_array

  def _time_now(self) -> float:
    """Returns the current time, allows mocking in tests."""
    return time.time()


class CmaEarlyStopper:
  """A class to handle the early stopping logic."""

  def __init__(self, spec: cma_trainer_spec_pb2.CmaEarlyStopperSpec):
    self._spec = spec
    self._prev_best_v = None
    self._latest_improved_iters = None

  def start(self):
    self._prev_best_v = None
    self._latest_improved_iters = None

  def should_stop(self, iters: int, best_v: jnp.ndarray) -> bool:
    """Returns whether to stop early."""
    if (
        self._prev_best_v is None
        or best_v > self._prev_best_v + self._spec.required_improvement
    ):
      self._latest_improved_iters = iters
    self._prev_best_v = best_v
    if iters > self._spec.min_iters:
      idle_iters = iters - self._latest_improved_iters
      allowed_idle_iters = iters * self._spec.allowed_idle_iters_fraction
      if idle_iters > allowed_idle_iters:
        return True
    return False


class SimpleTimer:

  def __init__(self):
    self._start_time = None

  def start(self):
    self._start_time = time.time()

  def stop(self) -> float:
    return time.time() - self._start_time
