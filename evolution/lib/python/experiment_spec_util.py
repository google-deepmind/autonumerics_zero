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

"""Utilities to parse an ExperimentSpec proto."""

from evolution.lib import experiment_spec_pb2
from evolution.lib.python import types


def unpack_worker_spec(worker_spec_str):
  """Unpacks a WorkerSpec proto.

  Args:
    worker_spec_str: the text-format worker spec proto.

  Returns:
    project_worker_spec: a project_worker_spec_cls instance.
    population_client_manager_spec: PopulationClientManagerSpec.
    historical_fitness_client_manager_spec: a HistoricalFitnessClientManagerSpec
      or None.
  """
  assert worker_spec_str
  worker_spec = types.parse_text_format(
      experiment_spec_pb2.WorkerSpec, types.nonempty_or_die(worker_spec_str)
  )
  assert len(worker_spec.Extensions) == 1
  project_worker_spec = worker_spec.Extensions[list(worker_spec.Extensions)[0]]

  assert worker_spec.HasField("client_manager")
  population_client_manager_spec = worker_spec.client_manager

  return (
      project_worker_spec,
      population_client_manager_spec,
      worker_spec,
  )
