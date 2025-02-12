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
from typing import List, Optional
from jax import random
from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.lib.python import rng as rng_lib
from evolution.projects.graphs import evaluator_interface
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import graph_transformer
from evolution.projects.graphs import learnable_params as learnable_params_lib
from evolution.projects.graphs.joy import constant_vertex_collapser
from evolution.projects.graphs.joy import cost_estimator as cost_estimator_lib
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import eval_metadata as eval_metadata_lib
from evolution.projects.graphs.joy import evaluator_spec_pb2
from evolution.projects.graphs.joy import evaluator_util
from evolution.projects.graphs.joy import interpretation
from evolution.projects.graphs.joy import trainers
from evolution.projects.graphs.joy import validators

EvalMetadata = eval_metadata_lib.EvalMetadata
JnpKey = data_lib.JnpKey
JnpPreciseFloat = data_lib.JnpPreciseFloat
LearnableParams = learnable_params_lib.LearnableParams
GraphEvalResult = evaluator_interface.GraphEvalResult


class JoyEvaluator(evaluator_interface.Evaluator):
  """A class to evaluate joy search space graphs.

  Evaluates the quality of a graph by how well it can approximate the target
  function at a set of points. Before assessing the error, it will
  optimize with SGD any learnable constants the graph has. Uses JAX.

  See the BUILD file for context.

  Raises:
    ValueError: if a value is invalid.
  """

  def __init__(self, spec: evaluator_spec_pb2.JoyEvaluatorSpec, rng_seed: int):
    self._spec = spec
    if rng_seed <= 0:
      self._rng = rng_lib.RNG(rng_lib.GenerateRNGSeed())
    else:
      self._rng = rng_lib.RNG(rng_seed)

    self._graph_transformer = None
    if self._spec.HasField("graph_transformer"):
      self._graph_transformer = graph_transformer.build_graph_transformer(
          spec=self._spec.graph_transformer, op_init_params=None
      )

    if self._spec.HasField("constant_vertex_collapser"):
      self._constant_vertex_collapser = (
          constant_vertex_collapser.ConstantVertexCollapser(
              self._spec.constant_vertex_collapser
          )
      )
    else:
      self._constant_vertex_collapser = None

    self._jnp_key = random.PRNGKey(self._rng.UniformRNGSeed())

    self._trainers = []
    for trainer_spec in self._spec.trainers:
      self._trainers.append(trainers.build(trainer_spec))

    if not self._spec.HasField("validator"):
      raise ValueError("Validator is required.")
    self._validator = validators.build(self._spec.validator)

    if self._spec.HasField("cost_estimator"):
      self._cost_estimator = cost_estimator_lib.CostEstimator(
          self._spec.cost_estimator
      )
    else:
      self._cost_estimator = None

  def evaluate(
      self,
      graph: graph_lib.Graph,
      serialized_eval_metadata: Optional[bytes] = None,
      serialized_eval_transient_metadata: Optional[bytes] = None,
  ) -> GraphEvalResult:
    """See base class."""
    # Transform the graph if requested.
    if self._graph_transformer is not None:
      graph = self._graph_transformer.transform(graph)
    if self._constant_vertex_collapser:
      graph = self._constant_vertex_collapser.collapse(graph)

    # Get the metadata.
    eval_metadata = EvalMetadata()
    eval_metadata.deserialize(
        serialized_eval_metadata, serialized_eval_transient_metadata, graph
    )
    self._maybe_generate_random_keys(eval_metadata)
    self._maybe_calculate_learnable_params(graph, eval_metadata)

    fitnesses = self._validator.validate(
        graph=graph, learnable_params=eval_metadata.learnable_params
    )
    for fitness in fitnesses:
      assert fitness.dtype == JnpPreciseFloat
    fitnesses = [float(f) for f in fitnesses]
    fitnesses = self._maybe_measure_cost(
        graph=graph, eval_metadata=eval_metadata, fitnesses=fitnesses
    )
    fitnesses = self._postprocess(fitnesses)

    (
        reserialized_eval_metadata,
        reserialized_eval_transient_metadata,
    ) = eval_metadata.serialize(graph)

    if self._spec.delay > 0.0:
      time.sleep(self._spec.delay)
    return GraphEvalResult(
        fitnesses,
        reserialized_eval_metadata,
        reserialized_eval_transient_metadata,
        eval_metadata.learnable_params,
    )

  def _maybe_measure_cost(
      self,
      graph: graph_lib.Graph,
      eval_metadata: EvalMetadata,
      fitnesses: List[float],
  ) -> List[float]:
    """If requested, appends a cost estimate to the fitnesses.

    Args:
      graph: the graph in question.
      eval_metadata: that graph's metadata.
      fitnesses: a list of fitnesses, not including the cost estimate.

    Returns:
      The list of fitnesses, possibly with an additional entry at the end
      indicating the cost estimate.

    Raises:
      RuntimeError: if an error occurs.
    """
    if self._cost_estimator is not None:
      cost_fitness = self._cost_estimator.estimate(
          graph=graph, learnable_params=eval_metadata.learnable_params
      )
      fitnesses.append(cost_fitness)
    return fitnesses

  def _postprocess(self, fitnesses: List[float]) -> List[float]:
    if self._spec.HasField("no_objective_postprocessing"):
      pass  # Nothing to do.
    elif self._spec.HasField("reduce_stratified_objective_postprocessing"):
      fitnesses = evaluator_util.reduce_stratified_objective_postprocessing(
          fitnesses, self._spec.reduce_stratified_objective_postprocessing
      )
    else:
      raise ValueError("Missing required oneof `objective_postprocessing`.")
    return fitnesses

  def _split_jnp_key(self) -> JnpKey:
    self._jnp_key, jnp_subkey = random.split(self._jnp_key)
    return jnp_subkey

  def _maybe_generate_random_keys(self, eval_metadata: EvalMetadata):
    """Adds random keys to the metadata, if they are not there already."""
    if eval_metadata.init_jnp_key is None:
      eval_metadata.init_jnp_key = self._split_jnp_key()
    if eval_metadata.train_jnp_key is None:
      eval_metadata.train_jnp_key = self._split_jnp_key()
    if eval_metadata.valid_jnp_key is None:
      eval_metadata.valid_jnp_key = self._split_jnp_key()
    if eval_metadata.finetune_jnp_key is None:
      eval_metadata.finetune_jnp_key = self._split_jnp_key()

  def _maybe_calculate_learnable_params(
      self, graph: graph_lib.Graph, eval_metadata: EvalMetadata
  ):
    """Add learnable parameters to metadata, if they are not already there."""
    if eval_metadata.learnable_params is None:  # If params are missing.
      eval_metadata.learnable_params = interpretation.init_learnable_params(
          graph=graph, jnp_key=eval_metadata.init_jnp_key, hashing_round=-1
      )
      if eval_metadata.learnable_params:  # If there are params.
        for trainer in self._trainers:
          eval_metadata.learnable_params = trainer.train(
              graph=graph,
              learnable_params=eval_metadata.learnable_params,
              jnp_key=eval_metadata.train_jnp_key,
          )
