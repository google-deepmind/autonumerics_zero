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

"""Reference time retrieval.

To measure the reference times, see
`WallClockCostEstimator.measure_reference_absolute_cost`.
"""

from absl import flags
from evolution.lib.python import printing
from evolution.projects.graphs.joy import wall_clock_reference_spec_pb2

_OVERRIDE_REFERENCE_ABSOLUTE_COST = flags.DEFINE_float(
    "override_reference_absolute_cost",
    -1.0,
    "Can be used in unit tests to override the absolute cost. This avoids "
    "a crash if the CPU is not supported. Negative means no override. ",
)


class WallClockReference:
  """Determines the reference time to use for time cost normalization."""

  def __init__(
      self, reference_cost_type: wall_clock_reference_spec_pb2.ReferenceCostType
  ):
    """Builds the instance."""
    self._reference_cost_type = reference_cost_type

    # The time to normalize the cost.
    self._reference_absolute_cost = self.load_reference_absolute_cost()
    printing.print_now(
        "Reference wall-clock cost = %.2f" % self._reference_absolute_cost
    )

  def get_reference_absolute_cost(self):
    return self._reference_absolute_cost

  def load_reference_absolute_cost(self):
    """Returns the absolute cost of the reference."""
    if _OVERRIDE_REFERENCE_ABSOLUTE_COST.value > 0.0:
      return _OVERRIDE_REFERENCE_ABSOLUTE_COST.value
    if (
        self._reference_cost_type
        == wall_clock_reference_spec_pb2.ReferenceCostType.NONE
    ):
      return self._load_no_reference_absolute_cost()
    raise RuntimeError(
        "reference_cost_type must be set in the spec to retrieve the "
        "reference absolute cost."
    )

  def _load_no_reference_absolute_cost(self):
    """Disables normalization."""
    return 1.0
