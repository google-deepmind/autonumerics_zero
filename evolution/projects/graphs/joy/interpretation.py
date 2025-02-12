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

"""Utilities for interpreting graphs as functions."""

from typing import Any, Dict, Callable, List

from jax import numpy as jnp
from jax import random

from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import data as data_lib

JnpPreciseFloat = data_lib.JnpPreciseFloat
LearnableParams = graph_lib.LearnableParams

# A function from (learnable_params, inputs) to outputs that represents a
# parameterized approximation. Can be obtained from interpreting a Graph.
ParameterizedFn = Callable[[LearnableParams, List[jnp.ndarray]], jnp.ndarray]

# The function from inputs to outputs that represents a finalized approximation.
# Any parameters that may have existed have already been bound. Can be optained
# from a ParameterizedFn and a set of learnable_parameters.
FinalizedFn = Callable[[List[jnp.ndarray]], jnp.ndarray]

# A map from vertex ID to the output of that vertex.
ExecutionOutputs = Dict[str, jnp.ndarray]


def init_learnable_params(
    graph: graph_lib.Graph, jnp_key: Any, hashing_round: int
) -> LearnableParams:
  """Initializes a graph's learnable params.

  Args:
    graph: the Graph whose params will be initialized.
    jnp_key: the random key to use to generate random numbers.
    hashing_round: which round of hashing this is or -1 if this is an
      evaluation. In particular, if -1 (typical), then random number generation
      is done in the standard way, producing a new sequence of numbers for the
      learnable parameters of each vertex. If >= 0, the same sequence is used
      each time a vertex uses any given op.

  Returns:
    The initialized parameters.
  """
  learnable_params = {}
  for vertex in graph.vertices.values():
    evolvable_params = None
    if vertex.op.has_evolvable_params:
      evolvable_params = vertex.evolvable_params
    if vertex.op.has_learnable_params:
      if hashing_round >= 0:  # If this is hashing.
        # The op's learnable params are always initialized the same way, even
        # if there are different vertices with the same op. This is useful for
        # hashing.
        jnp_subkey = jnp_key
      elif hashing_round == -1:  # If this is an evaluation.
        # The op's learnable params receive a split key, as is standard. This is
        # the typical way to initialize and it's useful for evaluation.
        jnp_key, jnp_subkey = random.split(jnp_key)
      else:
        raise ValueError("Invalid hashing_round.")
      learnable_params[vertex.vertex_id] = vertex.op.init_learnable_params(
          evolvable_params=evolvable_params,
          jnp_key=jnp_subkey,
          hashing_round=hashing_round,
      )
  data_lib.check_learnable_params_dtype(learnable_params, dtype=JnpPreciseFloat)
  return learnable_params


def get_parameterized_fn(
    graph: graph_lib.Graph, dtype: data_lib.JnpFloatDType
) -> ParameterizedFn:
  """Converts the Graph to its ParameterizedFn.

  The GraphFn is the function represented by the Graph instance. The GraphFn is
  a good object to JIT.

  Args:
    graph: the Graph instance to interpret.
    dtype: the type to use in internal computations when evaluating the function
      value.

  Returns:
    A function of (learnable_params, required inputs) to required outputs.
  """

  def parameterized_fn(
      learnable_params: LearnableParams, inputs: List[jnp.ndarray]
  ) -> jnp.ndarray:
    execution_outputs = execute_graph(
        graph=graph,
        learnable_params=learnable_params,
        inputs=inputs,
        dtype=dtype,
        contributing_only=True,
    )
    output_vertex_id = graph.required_outputs[0]
    predictions = execution_outputs[output_vertex_id]

    # If a constant never interacts with the inputs, it won't be broadcasted to
    # the correct shape. We fix that corner case here.
    inputs_shape = inputs[0].shape
    predictions = jnp.broadcast_to(predictions, inputs_shape)

    return predictions

  return parameterized_fn


def get_finalized_fn(
    parameterized_fn: ParameterizedFn, learnable_params: LearnableParams
) -> FinalizedFn:
  """Binds the parameterized_fn with the given learnable_params."""

  def finalized_fn(inputs: List[jnp.ndarray]) -> jnp.ndarray:
    return parameterized_fn(learnable_params, inputs)

  return finalized_fn


def execute_graph(
    graph: graph_lib.Graph,
    learnable_params: LearnableParams,
    inputs: List[jnp.ndarray],
    dtype: data_lib.JnpFloatDType,
    contributing_only: bool,
) -> ExecutionOutputs:
  """Executes the graph on the given inputs."""
  assert len(graph.required_inputs) == len(inputs)
  execution_inputs = {}
  inputs_shape = inputs[0].shape
  for input_vertex_id, input_tensor in zip(graph.required_inputs, inputs):
    assert input_tensor.shape == inputs_shape
    execution_inputs[input_vertex_id] = input_tensor
  assert len(graph.required_outputs) == 1
  execution_outputs = graph.execute(
      input_dict=execution_inputs,
      learnable_params=learnable_params,
      dtype=dtype,
      contributing_only=contributing_only,
  )
  return execution_outputs
