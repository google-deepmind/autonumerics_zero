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

"""Utils to print simple stats to logs."""

import collections
import time
from typing import Callable, Union

import numpy as np

from evolution.lib.python import printing

print_now = printing.print_now


class LogTracker(object):
  """A class to track the min, max, and mean of a value."""

  def __init__(
      self,
      log_message_prefix_fn: Callable[[], str],
      log_every_secs: float,
      capacity: int,
  ):
    """Initializes the instance.

    Args:
      log_message_prefix_fn: a function that returns a string with a message
        with which to prefix the log printout.
      log_every_secs: the interval to use for automatic logging, in seconds.
      capacity: consider up to these many values. If more than these have been
        tracked since the last printout, the earlier ones are ignored.
    """
    self._log_message_prefix_fn = log_message_prefix_fn
    self._log_every_secs = log_every_secs
    self._capacity = capacity
    self._buffer = collections.deque([], capacity)
    self._last_log_time = None

  def track(self, value: Union[float, int]):
    """Note a sample of the quantity we are interested in tracking.

    Call repeatedly. Occasionally, a call to `Track` will have the side-effect
    of printing the stats to the log.

    Args:
      value: the sample.
    """
    self._buffer.append(value)
    if (
        self._last_log_time is None
        or time.time() > self._last_log_time + self._log_every_secs
    ):
      self.log(self._log_message_prefix_fn())

  def log(self, message_prefix: str):
    """Print the statistics of the tracked quantity to the log."""
    if not self._buffer:
      print_now("no data.")
    else:
      print_now(
          "%smean=%s, min=%s, max=%s"
          % (
              message_prefix,
              str(np.mean(self._buffer)),
              str(min(self._buffer)),
              str(max(self._buffer)),
          )
      )
    self._last_log_time = time.time()
