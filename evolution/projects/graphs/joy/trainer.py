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

"""Trainer interface."""

from jax import numpy as jnp
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import interpretation

LearnableParams = graph_lib.LearnableParams
ParameterizedFn = interpretation.ParameterizedFn


class Trainer(object):
  """Base class for trainers."""

  def train(
      self,
      graph: graph_lib.Graph,
      learnable_params: LearnableParams,
      jnp_key: jnp.ndarray,
  ) -> LearnableParams:
    """Train the learnable parameters of a graph.

    Args:
      graph:
      learnable_params:
      jnp_key:

    Returns:
      The trained parameters.
    """
    raise NotImplementedError("Must be implemented by subclass.")

  def train_parameterized_fn(
      self,
      parameterized_fn: ParameterizedFn,
      learnable_params: LearnableParams,
      jnp_key: jnp.ndarray,
  ) -> LearnableParams:
    """Train the learnable parameters of a parameterized function.

    Args:
      parameterized_fn: the parameterized function.
      learnable_params: the untrained learnable parameters.
      jnp_key: a jax random key.

    Returns:
      The trained parameters.
    """
    raise NotImplementedError("Must be implemented by subclass.")
