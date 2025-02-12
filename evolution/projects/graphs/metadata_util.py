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

r"""Evolver class."""

from typing import Optional

from evolution.lib import individual_pb2

EVAL_METADATA_KEY = "eval"
EVAL_TRANSIENT_METADATA_KEY = "eval-transient"


def put_additional_binary_metadata(
    key: str,
    value: Optional[bytes],
    individual_data: individual_pb2.IndividualData,
    transient: bool,
) -> None:
  """Adds the passed metadata into the provided individual.

  Args:
    key: The key the data is stored under in the metadata field
    value: The metadata
    individual_data: The individual the metadata needs to be added to
    transient: If the data should be saved to spanner (true: not saved)

  Raises:
    ValueError: in the case of a duplicate key
  """

  metadata = (
      individual_data.additional_binary_meta_data
      if not transient
      else individual_data.additional_transient_binary_meta_data
  )
  for key_value in metadata:
    if key_value.key == key:
      raise ValueError("Key already exists.")
  if value is not None:
    metadata.append(individual_pb2.BytesKeyValuePair(key=key, value=value))


def get_additional_binary_metadata(
    key: str, individual: individual_pb2.Individual, transient: bool
) -> Optional[bytes]:
  """Returns the metadata stored with the provided key in the passed individual.

  Args:
    key: The key the data is stored under in the metadata field
    individual: The individual that has the metadata
    transient: If we should retrieve the transient or regular metadata

  Returns:
    The metadata corresponding to the passed key

  Raises:
    RuntimeError: in the case there are multiple instances of the key
  """

  value: Optional[bytes] = None
  for key_value in (
      individual.data.additional_binary_meta_data
      if not transient
      else individual.data.additional_transient_binary_meta_data
  ):
    if key_value.key == key:
      if value is None:
        value = key_value.value
      else:
        raise RuntimeError("Duplicated key.")
  return value
