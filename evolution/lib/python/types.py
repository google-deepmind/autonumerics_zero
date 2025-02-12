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

"""Types, conversions, and simple inline checks."""

from google.protobuf import text_format


def parse_serialized(proto_cls, serialized: bytes):
  """Parses a text-format proto of the given class.

  Args:
    proto_cls: the proto class.
    serialized: a binary string, typically obtained with
      proto_cls.SerializeAsString().

  Returns:
    An instance of the proto class.
  """
  proto = proto_cls()
  proto.ParseFromString(serialized)
  return proto


def parse_text_format(proto_cls, text):
  """Parses a text-format proto of the given class.

  Args:
    proto_cls: the proto class.
    text: a string with the proto in text-format.

  Returns:
    An instance of the proto class.
  """
  proto = proto_cls()
  text_format.Parse(text, proto)
  return proto


def try_parse_text_format(proto_cls, text):
  """Parses a text-format proto of the given class.

  Args:
    proto_cls: the proto class.
    text: a string with the proto in text-format.

  Returns:
    An instance of the proto class if success or `None` if failure.
  """
  proto = proto_cls()
  try:
    text_format.Parse(text, proto)
  except text_format.ParseError:
    return None
  return proto


def parse_text_format_extension(proto_cls, extension_cls, text):
  """Parses a text-format extended proto.

  Args:
    proto_cls: the base proto class.
    extension_cls: the extension of the proto class.
    text: a string with the proto, with extension, in text-format.

  Returns:
    An instance of the extension class.
  """
  proto = proto_cls()
  text_format.Parse(text, proto)
  return proto.Extensions[extension_cls.ext]


def print_text_format(proto):
  """Inverse of `parse_text_format`."""
  text = text_format.MessageToString(proto)

  # These are necessary to pass the text as a flag in launcher scripts.
  text = text.replace("\n", " ")
  text = text.replace("\t", " ")
  text = text.replace("\r", " ")

  return text


def nonempty_or_die(value):
  """Dies if value is empty (or None)."""
  if value:
    return value
  else:
    raise ValueError("Expected nonempty value.")


def positive_or_die(value):
  """Dies unless value is positive."""
  if value > 0:
    return value
  else:
    raise ValueError("Expected positive value.")


def nonnegative_or_die(value):
  """Dies unless value is non-negative."""
  if value >= 0:
    return value
  else:
    raise ValueError("Expected nonnegative value.")
