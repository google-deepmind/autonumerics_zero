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

"""Trainer building."""

from evolution.projects.graphs.joy import cma_trainer
from evolution.projects.graphs.joy import trainer as trainer_lib
from evolution.projects.graphs.joy import trainer_spec_pb2


def build(spec: trainer_spec_pb2.TrainerSpec) -> trainer_lib.Trainer:
  """Builds the trainer."""
  if spec.HasField("cma"):
    return cma_trainer.CmaTrainer(spec.cma)
  else:
    raise ValueError("Unknown or missing trainer.")
