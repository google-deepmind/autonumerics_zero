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

"""Simple graph hasher."""

from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import hasher_interface
from evolution.projects.graphs.joy import simple_hasher_spec_pb2


class SimpleHasher(hasher_interface.Hasher):
  """Hashes search space graphs based on strict sturctural identity.

  See SimpleHasherSpec for details.

  Raises:
    ValueError: if a value is invalid.
  """

  def __init__(self, spec: simple_hasher_spec_pb2.SimpleHasherSpec):
    self._spec = spec

  def hash(self, graph: graph_lib.Graph) -> bytes:
    """See base class."""
    return graph.serialize()
