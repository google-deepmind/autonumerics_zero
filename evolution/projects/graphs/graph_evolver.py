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

"""Evolver class."""

import time
from typing import Any, List, NamedTuple, Optional, Tuple

import numpy as np

from evolution.lib import individual_pb2
from evolution.lib import population_client_manager_spec_pb2
from evolution.lib import search_algorithm_stats_pb2
from evolution.lib.python import log_tracker
from evolution.lib.python import population_client_manager
from evolution.lib.python import printing
from evolution.lib.python import rng as rng_lib
from evolution.lib.python import timed_measurement
from evolution.projects.graphs import evaluator_interface
from evolution.projects.graphs import generators
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import graph_evolver_spec_pb2
from evolution.projects.graphs import hasher_interface
from evolution.projects.graphs import meta_validator_interface
from evolution.projects.graphs import metadata_util
from evolution.projects.graphs import mutators

print_now = printing.print_now


class _EvolverResult(NamedTuple):
  """Contains graph evaluation result."""

  fitnesses: List[float]
  eval_metadata: Optional[bytes]
  eval_transient_metadata: Optional[bytes]
  was_evaluated: bool = False
  cache_key: Optional[bytes] = None


PopulationClientManagerSpec = (
    population_client_manager_spec_pb2.PopulationClientManagerSpec
)

_NANOS_PER_SECOND = 1000000000
_META_VALIDATION_ID_LENGTH = 32


class GraphEvolver:
  """A class to evolve a graph.

  Please see the documentation in the GraphEvolverSpec for details.
  """

  def __init__(
      self,
      spec: graph_evolver_spec_pb2.GraphEvolverSpec,
      population_client_manager_spec: PopulationClientManagerSpec,
      worker_id: int,
      evaluator: evaluator_interface.Evaluator,
      hasher: Optional[hasher_interface.Hasher],
      rng_seed: int,
      meta_validator: Optional[meta_validator_interface.MetaValidator] = None,
      op_init_params: Any = None,
      custom_generator: Optional[generators.Generator] = None,
      custom_mutator: Optional[mutators.Mutator] = None,
  ):
    """Initializes the instance.

    Args:
      spec: the specification with how to build this object. Required.
      population_client_manager_spec: the specification for how to build the
        population client manager. Required.
      worker_id: a unique identifier for the worker.
      evaluator: an evaluator tailored to the search space being used.
      hasher: a hasher tailored to the search space being used.
      rng_seed: a random seed for the RNG. A positive seed will be respected
        literally and can be used within tests for determinism. Typically, this
        is non-positive, in which case it gets replaced by a randomly generated
        seed.
      meta_validator: DEPRECATED. an optional meta-validator.
      op_init_params: search-space-tailored parameters to initialize Ops. Use
        `None` if not needed.
      custom_generator: if set, this generator will be used to produce new seed
        candidates. Otherwise, a default generator will be produced using the
        configuration in the `spec`. Do not provide both.
      custom_mutator: if set, this mutator will be used to produce new
        candidates from parents. Otherwise, a default mutator will be produced
        using the configuration in the `spec`. Do not provide both.
    """
    self._spec = spec
    self._evaluator = evaluator
    self._hasher = hasher
    self._init_meta_validation(meta_validator, worker_id)
    if self._spec.HasField("fec") and not self._hasher:
      raise ValueError("FEC requires a hasher.")

    if rng_seed <= 0:
      rng_seed = rng_lib.GenerateRNGSeed()
    self._rng = rng_lib.RNG(rng_seed)
    self._np_rng = np.random.RandomState(self._rng.UniformRNGSeed())
    self._op_init_params = op_init_params

    self._local_fec_misses = 0
    self._local_fec_hits = 0
    self._local_fec_hash_collisions = 0

    self._manager = population_client_manager.PopulationClientManager(
        spec=population_client_manager_spec,
        worker_id=worker_id,
        rng_seed=self._rng.UniformRNGSeed(),
    )

    if not self._spec.HasField("identity_rate"):
      raise ValueError("Identity rate is required.")

    self._use_generator_or_build_standard(custom_generator)
    self._use_mutator_or_build_standard(custom_mutator)

    self._best_local_fitness0 = float("-inf")
    self._best_local_graph = None

  @property
  def hasher(self):
    if self._hasher is None:
      raise ValueError("Hasher is not initialized.")
    return self._hasher

  def _use_generator_or_build_standard(
      self, custom_generator: generators.Generator
  ):
    """Helper to the constructor."""
    if custom_generator:
      if self._spec.HasField("generator"):
        raise ValueError("Both standard and custom generators were specified.")
      self._generator = custom_generator
    else:
      if not self._spec.HasField("generator"):
        raise ValueError("Did not specify any generator.")
      self._generator = generators.build_standard_generator(
          spec=self._spec.generator,
          op_init_params=self._op_init_params,
          rng=self._np_rng,
      )

  def _use_mutator_or_build_standard(self, custom_mutator: mutators.Mutator):
    """Helper to the constructor."""
    if custom_mutator:
      if self._spec.HasField("mutator"):
        raise ValueError("Both standard and custom mutators were specified.")
      self._mutator = custom_mutator
    else:
      if not self._spec.HasField("mutator"):
        raise ValueError("Did not specify any mutator.")
      self._mutator = mutators.build_standard_mutator(
          spec=self._spec.mutator,
          op_init_params=self._op_init_params,
          rng=self._np_rng,
          hasher=self._hasher,
      )

  def evolve(self):
    """Runs evolution."""
    experiment_start_time = time.time()
    stats = self._get_population_stats()
    fitness0_tracker = log_tracker.LogTracker(
        log_message_prefix_fn=lambda: "Fitness0 @%d: " % experiment_size(stats),
        log_every_secs=self._spec.report_every_secs,
        capacity=self._spec.report_over_last,
    )

    while experiment_size(stats) < self._spec.experiment_size:
      parent_individuals, stats = self._manager.Communicate([], 2)
      if experiment_size(stats) < self._spec.init_population_size:
        stats = self._do_seeding_cycle()
      else:
        stats = self._do_evolution_cycle(
            parent_individuals=parent_individuals,
            stats=stats,
            fitness0_tracker=fitness0_tracker,
        )

    experiment_end_time = time.time()

    print_now("Evolution complete.")
    fitness0_tracker.log("Latest fitness @%d: " % experiment_size(stats))
    if self._spec.verbose:
      experiment_elapsed_time = experiment_end_time - experiment_start_time
      print_now("Total experiment time: %f\n" % experiment_elapsed_time)
      if self._best_local_graph is not None:
        print_now("Best local fitness0: %f\n" % self._best_local_fitness0)
        print_now(
            "Best local graph by fitness0: \n%s"
            % self._best_local_graph.debug_string()
        )

    self._maybe_run_meta_validation()

  def _do_seeding_cycle(
      self,
  ) -> search_algorithm_stats_pb2.SearchAlgorithmStats:
    """Adds one seed individual to the population."""
    with timed_measurement.measure("generation_time"):
      graph = self._generator.generate()
    with timed_measurement.new_timer() as total_timer:
      evolver_result = self._evaluate(
          graph=graph,
          prev_eval_metadata=None,
          prev_eval_transient_metadata=None,
      )
    stats = self._add_to_population(
        graph=graph,
        fitnesses=evolver_result.fitnesses,
        eval_metadata=evolver_result.eval_metadata,
        eval_transient_metadata=evolver_result.eval_transient_metadata,
        parent_individuals=None,
        total_time=total_timer.result,
        was_evaluated=evolver_result.was_evaluated,
        cache_key=evolver_result.cache_key,
    )
    return stats

  def _do_evolution_cycle(
      self,
      parent_individuals: List[individual_pb2.Individual],
      stats: search_algorithm_stats_pb2.SearchAlgorithmStats,
      fitness0_tracker: log_tracker.LogTracker,
  ) -> search_algorithm_stats_pb2.SearchAlgorithmStats:
    """Do one round of selection and mutation.

    Args:
      parent_individuals: the two parents.
      stats: the stats object.
      fitness0_tracker: a tracker for the 0th fitness. Only used to write
        progress to logs (for debugging, etc).

    Returns:
      The stats object.

    Raises:
      RuntimeError: if there is one.
    """
    if len(parent_individuals) != 2:
      raise RuntimeError(
          "Expected to be receiving parents by now. Is the "
          "init_population_size correct? Does it match the server spec?"
      )
    is_identity = self._rng.UniformProbability() < self._spec.identity_rate
    parent0 = self._extract_graph(parent_individuals[0])
    parent0_eval_metadata = metadata_util.get_additional_binary_metadata(
        key=metadata_util.EVAL_METADATA_KEY,
        individual=parent_individuals[0],
        transient=False,
    )
    parent0_eval_tr_metadata = metadata_util.get_additional_binary_metadata(
        key=metadata_util.EVAL_TRANSIENT_METADATA_KEY,
        individual=parent_individuals[0],
        transient=True,
    )
    parent1 = self._extract_graph(parent_individuals[1])
    parent1_eval_metadata = metadata_util.get_additional_binary_metadata(
        key=metadata_util.EVAL_METADATA_KEY,
        individual=parent_individuals[1],
        transient=False,
    )
    parent1_eval_tr_metadata = metadata_util.get_additional_binary_metadata(
        key=metadata_util.EVAL_TRANSIENT_METADATA_KEY,
        individual=parent_individuals[1],
        transient=True,
    )
    if is_identity:
      child = parent0
      prev_eval_metadata = parent0_eval_metadata
      prev_eval_transient_metadata = parent0_eval_tr_metadata
    else:
      with timed_measurement.measure("mutation_time"):
        child = self._mutator.mutate(
            parent=parent0,
            serialized_parent_eval_metadata=parent0_eval_metadata,
            serialized_parent_eval_transient_metadata=parent0_eval_tr_metadata,
            parent1=parent1,
            serialized_parent1_eval_metadata=parent1_eval_metadata,
            serialized_parent1_eval_transient_metadata=parent1_eval_tr_metadata,
            stats=stats,
        )
        prev_eval_metadata, prev_eval_transient_metadata = None, None
    with timed_measurement.new_timer() as timer:
      evolver_result = self._evaluate(
          graph=child,
          prev_eval_metadata=prev_eval_metadata,
          prev_eval_transient_metadata=prev_eval_transient_metadata,
      )
    stats = self._add_to_population(
        graph=child,
        fitnesses=evolver_result.fitnesses,
        eval_metadata=evolver_result.eval_metadata,
        eval_transient_metadata=evolver_result.eval_transient_metadata,
        parent_individuals=parent_individuals,
        total_time=timer.result,
        was_evaluated=evolver_result.was_evaluated,
        cache_key=evolver_result.cache_key,
    )
    fitness0_tracker.track(evolver_result.fitnesses[0])
    return stats

  @property
  def best_local_fitness0(self):
    """Returns the best fitnesses[0] seen locally. Useful for unit tests."""
    return self._best_local_fitness0

  def _evaluate(
      self,
      graph: graph_lib.Graph,
      prev_eval_metadata: Optional[bytes],
      prev_eval_transient_metadata: Optional[bytes],
  ) -> _EvolverResult:
    """Evaluates the graph.

    Args:
      graph: the graph to evaluate.
      prev_eval_metadata: metadata from a previous evaluation if known (i.e. if
        this is an identity).
      prev_eval_transient_metadata: transient metadata from a previous
        evaluation if known (i.e. if this is an identity).

    Returns:
      fitnesses: the list of fitnesses resulting from the evaluation.
      eval_metadata: the evaluation metadata.
      was_evaluated: whether a full evaluation took place (as opposed to a
        cache hit).
      cache_key: the hash of the graph, used for FEC. Populated only if hasher
        exists.

    Raises:
      ValueError: if there is one.
    """
    if self._spec.HasField("fec"):
      if self._spec.fec.HasField("caching"):
        evolver_result = self._evaluate_fec_caching(
            graph=graph,
            prev_eval_metadata=prev_eval_metadata,
            prev_eval_transient_metadata=prev_eval_transient_metadata,
        )
      elif self._spec.fec.HasField("aggregation"):
        evolver_result = self._evaluate_fec_aggregation(
            graph=graph,
            prev_eval_metadata=prev_eval_metadata,
            prev_eval_transient_metadata=prev_eval_transient_metadata,
        )
      elif self._spec.fec.HasField("counterfactual"):
        evolver_result = self._evaluate_fec_counterfactual(
            graph=graph,
            prev_eval_metadata=prev_eval_metadata,
            prev_eval_transient_metadata=prev_eval_transient_metadata,
        )
      else:
        raise ValueError("Unknown FEC usage type.")
    else:
      # No caching. Evaluate the model.
      evolver_result = self._evaluate_no_fec(
          graph=graph,
          prev_eval_metadata=prev_eval_metadata,
          prev_eval_transient_metadata=prev_eval_transient_metadata,
      )
    return evolver_result

  def _evaluate_fec_caching(
      self,
      graph: graph_lib.Graph,
      prev_eval_metadata: Optional[bytes],
      prev_eval_transient_metadata: Optional[bytes],
  ) -> _EvolverResult:
    """Helper to _evaluate. Evaluates the graph using FEC."""
    if prev_eval_metadata is not None:
      raise RuntimeError(
          "Metadata not supported with FEC, because the "
          "metadata is not cached yet."
      )
    if prev_eval_transient_metadata is not None:
      raise RuntimeError(
          "Transient metadata not supported with FEC, because the "
          "metadata is not cached yet."
      )
    fitnesses = None
    eval_transient_metadata = None
    with timed_measurement.measure("hash_time"):
      cache_key = self.hasher.hash(graph)
    lookup_result = None
    if lookup_result is None:
      with timed_measurement.measure("evaluation_time"):
        fitnesses, eval_metadata, eval_transient_metadata = (
            self._evaluate_graph(
                graph=graph,
                prev_eval_metadata=None,
                prev_eval_transient_metadata=None,
            )
        )
        if eval_metadata is not None:
          raise RuntimeError(
              "Metadata not supported with FEC, because the "
              "metadata i`s not cached yet."
          )
        if eval_transient_metadata is not None:
          raise RuntimeError(
              "Transient metadata not supported with FEC, because the "
              "metadata is not cached yet."
          )
        if cache_key == hasher_interface.UNHASHABLE:
          fitnesses = _empty_fitnesses(fitnesses)
      self._local_fec_misses += 1
      was_evaluated = True
    else:
      # fitnesses = lookup_result.latest_fitnesses
      eval_metadata = None
      self._local_fec_hits += 1
      was_evaluated = False

    if fitnesses[0] > self._best_local_fitness0:
      self._best_local_fitness0 = fitnesses[0]
      self._best_local_graph = graph.clone()
    return _EvolverResult(
        fitnesses=fitnesses,
        eval_metadata=eval_metadata,
        eval_transient_metadata=eval_transient_metadata,
        was_evaluated=was_evaluated,
        cache_key=cache_key,
    )

  def _evaluate_fec_aggregation(
      self,
      prev_eval_metadata: Optional[bytes],
      prev_eval_transient_metadata: Optional[bytes],
      graph: graph_lib.Graph,
  ) -> _EvolverResult:
    """Helper to _evaluate. Evaluates the graph using FEC with aggregation."""
    if prev_eval_metadata is not None:
      raise RuntimeError(
          "Metadata not supported with FEC aggregation, because the "
          "metadata is not cached yet."
      )
    if prev_eval_transient_metadata is not None:
      raise RuntimeError(
          "Transient metadata not supported with FEC aggregation, because the "
          "metadata is not cached yet."
      )
    assert self._spec.HasField("fec")
    assert self._spec.fec.HasField("aggregation")

    # Look up in cache.
    mean_fitnesses = None
    eval_count = 0
    with timed_measurement.measure("hash_time"):
      cache_key = self.hasher.hash(graph)

    # If it has not been evaluated enough times, evaluate it once more.
    with timed_measurement.measure("evaluation_time"):
      new_fitnesses, new_eval_metadata, new_eval_transient_metadata = (
          self._evaluate_graph(
              graph=graph,
              prev_eval_metadata=None,
              prev_eval_transient_metadata=None,
          )
      )
      if new_eval_metadata is not None:
        raise RuntimeError(
            "Metadata not supported with FEC aggregation, because the "
            "metadata is not cached yet."
        )
      if new_eval_transient_metadata is not None:
        raise RuntimeError(
            "Transient metadata not supported with FEC aggregation, because "
            "the metadata is not cached yet."
        )
      if cache_key == hasher_interface.UNHASHABLE:
        new_fitnesses = _empty_fitnesses(new_fitnesses)
    self._local_fec_misses += 1
    mean_fitnesses, _ = _update_fitnesses(
        mean_fitnesses, new_fitnesses, eval_count
    )
    eval_metadata = new_eval_metadata
    eval_transient_metadata = new_eval_transient_metadata

    assert mean_fitnesses
    if mean_fitnesses[0] > self._best_local_fitness0:
      self._best_local_fitness0 = mean_fitnesses[0]
      self._best_local_graph = graph.clone()
    return _EvolverResult(
        fitnesses=mean_fitnesses,
        eval_metadata=eval_metadata,
        eval_transient_metadata=eval_transient_metadata,
        was_evaluated=True,
        cache_key=cache_key,
    )

  def _evaluate_fec_counterfactual(
      self,
      graph: graph_lib.Graph,
      prev_eval_metadata: Optional[bytes],
      prev_eval_transient_metadata: Optional[bytes],
  ) -> _EvolverResult:
    """Helper to _evaluate. Evaluates the graph using FEC counterfactual."""
    assert self._spec.HasField("fec")
    assert self._spec.fec.HasField("counterfactual")
    with timed_measurement.measure("hash_time"):
      cache_key = self.hasher.hash(graph)
    with timed_measurement.measure("evaluation_time"):
      new_fitnesses, new_eval_metadata, new_eval_transient_metadata = (
          self._evaluate_graph(
              graph=graph,
              prev_eval_metadata=prev_eval_metadata,
              prev_eval_transient_metadata=prev_eval_transient_metadata,
          )
      )
      if cache_key == hasher_interface.UNHASHABLE:
        new_fitnesses = _empty_fitnesses(new_fitnesses)

    self._local_fec_misses += 1

    if new_fitnesses[0] > self._best_local_fitness0:
      self._best_local_fitness0 = new_fitnesses[0]
      self._best_local_graph = graph.clone()
    return _EvolverResult(
        fitnesses=new_fitnesses,
        eval_metadata=new_eval_metadata,
        eval_transient_metadata=new_eval_transient_metadata,
        was_evaluated=True,
        cache_key=cache_key,
    )

  def _evaluate_no_fec(
      self,
      graph: graph_lib.Graph,
      prev_eval_metadata: Optional[bytes],
      prev_eval_transient_metadata: Optional[bytes],
  ) -> _EvolverResult:
    """Evaluates the graph.

    Args:
      graph: the graph to evaluate.
      prev_eval_metadata: the metadata from a previous evaluation or `None` if
        not known.
      prev_eval_transient_metadata: the transient metadata from a previous
        evaluation or `None` if not known.

    Returns:
      fitnesses: the list of fitnesses resulting from the evaluation.
      eval_metadata: the evaluation metadata.
      eval_transient_metadata: the evaluation transient metadata.
    """
    # If it has not been evaluated enough times, evaluate it once more.
    with timed_measurement.measure("evaluation_time"):
      fitnesses, eval_metadata, eval_transient_metadata = self._evaluate_graph(
          graph=graph,
          prev_eval_metadata=prev_eval_metadata,
          prev_eval_transient_metadata=prev_eval_transient_metadata,
      )
    assert fitnesses
    if fitnesses[0] > self._best_local_fitness0:
      self._best_local_fitness0 = fitnesses[0]
      self._best_local_graph = graph.clone()
    return _EvolverResult(
        fitnesses=fitnesses,
        eval_metadata=eval_metadata,
        eval_transient_metadata=eval_transient_metadata,
        was_evaluated=True,
        cache_key=None,
    )

  def _evaluate_graph(
      self,
      graph: graph_lib.Graph,
      prev_eval_metadata: Optional[bytes],
      prev_eval_transient_metadata: Optional[bytes],
  ) -> Tuple[List[float], Optional[bytes], Optional[bytes]]:
    """Evaluates the graph and updates the evolvable params (if needed)."""
    graph_eval_result = self._evaluator.evaluate(
        graph, prev_eval_metadata, prev_eval_transient_metadata
    )
    for vertex_id, vertex in graph.vertices.items():
      # Update the evolvable params for this vertex.
      if vertex_id in graph_eval_result.learnable_params:
        learnable_params = graph_eval_result.learnable_params[vertex_id]
      else:
        learnable_params = None
      vertex.evolvable_params = vertex.op.update_evolvable_params(
          evolvable_params=vertex.evolvable_params,
          learnable_params=learnable_params,
      )
    return (
        graph_eval_result.fitnesses,
        graph_eval_result.eval_metadata,
        graph_eval_result.eval_transient_metadata,
    )

  def _get_population_stats(
      self,
  ) -> search_algorithm_stats_pb2.SearchAlgorithmStats:
    """Finds out the most recent population stats.

    Note that _add_to_population also gets the stats, while doing other things
    too, so prefer to call that.

    Returns:
      The stats object.
    """
    _, stats = self._manager.Communicate([], 0)
    return stats

  def _add_to_population(
      self,
      graph: graph_lib.Graph,
      fitnesses: List[float],
      eval_metadata: Optional[bytes],
      eval_transient_metadata: Optional[bytes],
      parent_individuals: Optional[List[individual_pb2.Individual]],
      total_time: float,
      was_evaluated: bool,
      cache_key: Optional[bytes],
  ) -> search_algorithm_stats_pb2.SearchAlgorithmStats:
    """Adds a graph to the evolving population.

    Args:
      graph: the graph to add.
      fitnesses: the fitnesses resulting from evaluating this graph.
      eval_metadata: the metadata resulting from evaluating this graph.
      eval_transient_metadata: The metadata from evaluating this graph that
        shouldn't be saved in spanner.
      parent_individuals: the parents of this graph. Use `None` if this is a
        seed individual.
      total_time: how long it took to evaluated this individual.
      was_evaluated: whether the graph was evaluated (i.e. it was a cache miss).
      cache_key: the graph hash used for FEC.

    Returns:
      The most recent known stats object.
    """
    if not fitnesses:
      raise ValueError("Must have at least 1 fitness.")
    data = individual_pb2.IndividualData()
    data.serialized = graph.serialize()
    data.fitnesses.extend(fitnesses)
    data.individual_progress.append(1.0)  # Number of individuals.
    data.individual_progress.append(total_time)
    data.individual_progress.append(
        1.0 if was_evaluated else 0.0
    )  # Number of evaluations.
    if cache_key is not None:
      data.hashes.append(cache_key)

    # Metadata.
    if parent_individuals:  # If it is not a seed individual.
      data.parent_ids.extend([p.individual_id for p in parent_individuals])
      data.earliest_algorithm_state_nanos = min(
          [p.time_nanos for p in parent_individuals]
      )
      data.latest_algorithm_state_nanos = max(
          [p.time_nanos for p in parent_individuals]
      )
    else:  # If it is a seed individual.
      # Irrelevant for seed individuals, but must be present.
      data.earliest_algorithm_state_nanos = int(time.time() * _NANOS_PER_SECOND)
      data.latest_algorithm_state_nanos = int(time.time() * _NANOS_PER_SECOND)
    metadata_util.put_additional_binary_metadata(
        key=metadata_util.EVAL_METADATA_KEY,
        value=eval_metadata,
        individual_data=data,
        transient=False,
    )
    metadata_util.put_additional_binary_metadata(
        key=metadata_util.EVAL_TRANSIENT_METADATA_KEY,
        value=eval_transient_metadata,
        individual_data=data,
        transient=True,
    )

    _, stats = self._manager.Communicate([data], 0)
    return stats

  def _extract_graph(
      self, individual: individual_pb2.Individual
  ) -> graph_lib.Graph:
    """Extracts the graph associated with this individual."""
    graph = graph_lib.Graph()
    graph.deserialize(
        individual.data.serialized, op_init_params=self._op_init_params
    )
    return graph

  def _init_meta_validation(
      self,
      meta_validator: meta_validator_interface.MetaValidator,
      worker_id: int,
  ):
    """Initializes the meta-validator if needed."""
    if (
        self._spec.HasField("meta_validation")
        and self._spec.meta_validation.worker_id == worker_id
    ):
      self._meta_validator = meta_validator
      assert self._meta_validator is not None
    else:
      self._meta_validator = None

  def _maybe_run_meta_validation(self):
    """Runs offline meta-validation according to the MetaValidationSpec.

    This meta-validation takes place at the end of the experiment.
    """
    if self._meta_validator is None:
      return
    print_now("Starting meta-validation...")
    if self._manager.NumClients() != 1:
      raise NotImplementedError(
          "Meta validation with multiple-servers not yet supported."
      )
    population, stats = self._manager.Communicate([], -1)
    individuals = self._meta_validator.filter(population)
    all_meta_validations = []
    for individual in individuals:
      print_now(
          "Meta-validating individual %s..." % str(individual.individual_id)
      )
      graph = self._extract_graph(individual)
      meta_validation = self._meta_validator.meta_validate(graph)
      meta_validation.individual_id = individual.individual_id
      meta_validation.meta_validation_id = self._rng.UniformString(
          _META_VALIDATION_ID_LENGTH
      )
      meta_validation.stats.CopyFrom(stats)
      meta_validation.individual.CopyFrom(individual)
      all_meta_validations.append(meta_validation)
    print_now("Meta-validation complete.")


def experiment_size(
    stats: search_algorithm_stats_pb2.SearchAlgorithmStats,
) -> int:
  """Returns the approximate number of individuals in the experiment so far."""
  if stats.experiment_progress:
    return round(stats.experiment_progress[0])
  else:
    return 0


def _update_fitnesses(
    mean_fitnesses: Optional[List[float]],
    new_fitnesses: List[float],
    eval_count: int,
) -> Tuple[List[float], int]:
  """Computes the average of previous fitnesses and the new ones."""
  if mean_fitnesses is None:
    assert eval_count == 0
    return new_fitnesses, 1
  assert len(mean_fitnesses) == len(new_fitnesses)
  updated_mean_fitnesses = []
  a = float(eval_count) / float(eval_count + 1)
  b = 1.0 / float(eval_count + 1)
  for previous, new in zip(mean_fitnesses, new_fitnesses):
    updated_mean_fitnesses.append(a * previous + b * new)
  return updated_mean_fitnesses, eval_count + 1


def _empty_fitnesses(fitnesses: List[float]) -> List[float]:
  return [evaluator_interface.MIN_FITNESS for _ in fitnesses]
