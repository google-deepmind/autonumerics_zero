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

"""Validator building."""

from evolution.projects.graphs.joy import standard_validator
from evolution.projects.graphs.joy import validator as validator_lib
from evolution.projects.graphs.joy import validator_spec_pb2


def build(spec: validator_spec_pb2.ValidatorSpec) -> validator_lib.Validator:
  if spec.HasField("standard"):
    return standard_validator.StandardValidator(spec.standard)
  else:
    raise ValueError("Unknown or missing validator.")
