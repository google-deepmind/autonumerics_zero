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

r"""Interface for evaluators.

All evaluators must derive from this base class. Evaluators should be in the
appropriate subdirectory (e.g. graphs/auto_rl).
"""

from typing import Any, List, NamedTuple, Optional

from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import learnable_params as learnable_params_lib

MIN_FITNESS = 0.0


class GraphEvalResult(NamedTuple):
  """Contains graph evaluation result."""

  fitnesses: List[float]
  eval_metadata: Optional[bytes]
  eval_transient_metadata: Optional[bytes]
  learnable_params: learnable_params_lib.LearnableParams
  pred: Optional[Any] = None  # Output of graph. Usually a np or jax array.


class Evaluator(object):
  """Base class for objects that evaluate graphs."""

  def evaluate(
      self,
      graph: graph_lib.Graph,
      prev_eval_metadata: Optional[bytes],
      prev_eval_transient_metadata: Optional[bytes],
  ) -> GraphEvalResult:
    """Performs the evaluation.

    Args:
      graph: the graph to evaluate.
      prev_eval_metadata: metadata from a previous evaluation or `None` if
        unknown.
      prev_eval_transient_metadata: transient metadata (which shouldn't be
        stored in spanner) from a previous evaluation or `None` if unknown.

    Returns:
      A tuple (fitnesses, eval_metadata, learnable_params), where:
      -fitnesses: a list of fitnesses resulting from the evaluation.
      -eval_metadata: a serialized metadata bytestring. This metadata
        cannot be too long as it will be stored in Spanner. Other than that,
        the metadata can be anything. If search spaces don't need metadata,
        they can return `None` instead of the bytestring.
      -eval_transient_metadata: a serialized metadata bytestring. This metadata
        will not be stored in Spanner. If search spaces don't need metadata,
        they can return `None` instead of the bytestring.
      -learnable_params: a dictionary from vertex ID to the learnable params of
        that vertex. If there are no learnable params in the search space, can
        just return an empty dictionary.
    """
    raise NotImplementedError("Must be implemented by the subclass.")
