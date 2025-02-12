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

r"""Base class for meta-validators.

All meta-validators must derive from this base class. Meta-validators should
be in the appropriate subdirectory (e.g. graphs/meta_pg). Meta-validators are
run at the end of the experiment by selected workers.
"""

from typing import List

from evolution.lib import individual_pb2
from evolution.lib import meta_validation_pb2
from evolution.projects.graphs import graph as graph_lib


class MetaValidator(object):
  """Base class for objects that evaluate a graph on the meta-validation set.

  See the MetaValidatorSpec for details.
  """

  def filter(
      self, population: List[individual_pb2.Individual]
  ) -> List[individual_pb2.Individual]:
    """Filters population to keep only those individuals to meta-validate.

    This could select the top-1 if there is one fitness or a pareto-front if
    there are multiple fitnesses, for example.

    Args:
      population: the population at the end of the experiment.

    Returns:
      The individuals kept by this filter.
    """
    raise NotImplementedError("Must be implemented by the subclass.")

  def meta_validate(
      self, graph: graph_lib.Graph
  ) -> meta_validation_pb2.MetaValidation:
    """Performs the evaluation.

    Args:
      graph: the graph to evaluate.

    Returns:
      The results of the meta-validation as a MetaValidation proto.
      The implementation of this method must create the MetaValidation proto and
      fill it appropriately, but must leave the following fields blank, to be
      filled by the evolution infra: individual_id, meta_validation_id, and
      stats.
    """
    raise NotImplementedError("Must be implemented by the subclass.")
