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

"""Parsing an experiment into a data structure useful for analysis.

This is for experiments based on data in the Snapshots table.

This is for experiments that were configured to write to the Snapshots table.
For experiments that wrote to the Individuals table, please see
experiment_individuals_parsing.py instead.
"""

from typing import Any, Dict, List
from evolution.lib import snapshot_pb2

Snapshot = snapshot_pb2.Snapshot


class ParsedExperimentSnapshots(object):
  """Holds experiment results in a form amenable to analysis.

  Create this class using the methods `parse_from_spanner` or
  `parse_from_dumped_experiment`, depending on where your experiment is stored.

  This class is built by reading the Spanner Snapshots table, so it most likely
  will not contain all the individuals in the experiment.

  Attributes:
    experiment_id: the experiment ID.
    snapshots: dict from snapshot ID to Snapshot proto.
    ordered_snapshots: a list of all the snapshots loaded, in chronological
      order.
    order: a map from snapshot ID to the position of this snapshot, in
      chronological order, among all the snapshots. It is the "inverse" of
      `ordered_snapshots`.
    experiment_progress: a dict from snapshot ID to the cumulative experiment
      progress for that snapshot. This is like the "time" from the start of the
      experiment. More precisely, progress is a list where each entry is one of
      the progress metrics (see `individual_progress` in the Individual proto).
      For example, if the 2nd progress metric is compute time, then
      `experiment_progress["abc"][2]` contains the total compute time from the
      beginning of the experiment up to, and including, the point when the
      snapshot with ID "abc" was taken.
    annotations: a dictionary from annotation ID to an annotation. Each
      anotation can be anything. It is useful to store the result of a
      computation on an experiment or an intermediate quantity (e.g.: you can
      store the diversity of each snapshot in order to later make a plot). The
      keys are any useful identifying strings.
  """

  def __init__(self, experiment_id: str, snapshots: List[Snapshot]):
    """Initializes the instance.

    Args:
      experiment_id: the experiment ID.
      snapshots: all the snapshots in this experiment.
    """
    self.experiment_id = experiment_id
    self.snapshots: Dict[str, Snapshot] = {s.snapshot_id: s for s in snapshots}
    self.ordered_snapshots: List[Snapshot] = sorted(
        snapshots, key=lambda s: s.stats.time_nanos
    )
    self.order: Dict[str, int] = {
        s.snapshot_id: pos for pos, s in enumerate(self.ordered_snapshots)
    }

    # Experiment progress.
    self.experiment_progress: Dict[str, List[float]] = {}
    for snapshot in self.ordered_snapshots:
      self.experiment_progress[snapshot.snapshot_id] = list(
          snapshot.stats.experiment_progress
      )

    self.annotations: Dict[str, Any] = {}

  def purge(self):
    """Purges the original data in the experiment.

    This can be useful if annotations have been filled with results, we are
    going to use this object as a convenient temporary storage for those
    results, and we want to reduce the memory footprint of this object.
    """
    del self.snapshots
    del self.ordered_snapshots
    del self.order
    del self.experiment_progress
