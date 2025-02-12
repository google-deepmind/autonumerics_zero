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

"""Validator interface."""

from typing import List

from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import interpretation

JnpPreciseFloat = data_lib.JnpPreciseFloat


class Validator(object):
  """Base class for validators."""

  def validate(
      self, graph: graph_lib.Graph, learnable_params: graph_lib.LearnableParams
  ) -> List[JnpPreciseFloat]:
    """Performs the validation, after JIT compiling.

    Args:
      graph: the graph to validate.
      learnable_params: the parameters to use with the graph.

    Returns:
      The quality objective(s) value(s).
    """
    raise NotImplementedError("Must be implemented by subclass.")

  def validate_finalized_fn(
      self, finalized_fn: interpretation.FinalizedFn
  ) -> List[JnpPreciseFloat]:
    """Performs the validation.

    Args:
      finalized_fn: the function to validate. It is expected to match the dtype
        provided. If created from a graph, the learnable_parameters must have
        been casted to this dtype.

    Returns:
      The quality objective(s) value(s).

    Raises:
      RuntimeError: if there's an error.
    """
    raise NotImplementedError("Must be implemented by subclass.")
