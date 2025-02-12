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

"""Utilities to help with measurements."""

import contextlib
import time


def define(
    quantity_name_prefix: str,
):
  """Defines a set of quantities for the timing of a section of code.

  Args:
    quantity_name_prefix: the quantity name prefix. E.g. "evaluation_time" to if
      the code is performing an evaluation. Any short string is valid.

  Intended to be used with `measure`. Please see there.
  """
  assert quantity_name_prefix


def try_define(
    quantity_name_prefix: str,
):
  """Like `define` but does not fail if already defined."""
  assert quantity_name_prefix


@contextlib.contextmanager
def measure(
    quantity_name_prefix: str,
):
  """Saves call timing and count for everything performed within the context.

  Intended to be used with `define`, as follows:
    from ... import timed_measurement
    saver = MeasurementsSaver(...)
    timed_measurement.define("evaluation", saver)
    for ...:  // Evolution loop (should maintain an up-to-date stats object).
      ...
      with timed_measurement.measure("evaluation", stats, saver) as timer:
        ... execute evaluation ...
      // At this point the relevant measurements have been saved to Spanner.
      // Also, the time it took can be accessed through `timer.result`.

  Args:
    quantity_name_prefix: a prefix for the quantity names that will be used to
      identity the measurements. Must match the arg to
      `define_time_measurement`.

  Yields:
    A timer object that can be used to consult the result.
  """
  assert quantity_name_prefix
  with new_timer() as timer:
    yield timer


@contextlib.contextmanager
def new_timer():
  """A context to time a piece of code.

  Use on its own or, for measurements, use with `timed_measurements`.

  Yields:
    The Timer object.
  """
  timer = Timer()
  timer.start()

  try:
    yield timer
  except GeneratorExit:
    print("good cm exit")
  timer.stop()


class Timer(object):
  """An object to time a piece of code. Use with `new_timer`."""

  def __init__(self):
    self._start = None
    self._result = None

  @property
  def result(self):
    assert self._result is not None
    return self._result

  def start(self):
    self._start = time.time()

  def stop(self):
    assert self._start is not None
    self._result = time.time() - self._start
