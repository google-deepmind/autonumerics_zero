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

from typing import List
from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs.joy import evaluator_util_pb2


def reduce_stratified_objective_postprocessing(
    objectives: List[float],
    spec: evaluator_util_pb2.ReduceStratifiedObjectivePostprocessing,
) -> List[float]:
  """Postprocesses the objectives into fitnesses.

  See ReduceStratifiedObjectivePostprocessing proto.

  Args:
    objectives: the list of objectives to postprocess.
    spec: the ReduceStratifiedObjectivePostprocessing spec.

  Returns:
    A list containing the fitnesses.

  Raises:
    RuntimeError: when the list of objectives does not contain exactly two.
  """
  if not spec.HasField("quality_threshold"):
    raise ValueError("Missing required field `quality_threshold`.")
  if len(objectives) != 2:
    raise RuntimeError("Requires exactly two objectives.")
  if objectives[1] < 0.0:
    raise RuntimeError("Found negative cost.")
  if objectives[0] < spec.quality_threshold:
    return [objectives[0] - spec.quality_threshold]
  else:
    return [objectives[1]]
