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

"""Data-related classes and methods."""

from typing import Any, List, Optional, Union

from jax import numpy as jnp


from evolution.lib.python import printing  # pylint: disable=unused-import
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import data_pb2


# To use in PyType expressions.
JnpFloat = Union[jnp.bfloat16, jnp.float32, jnp.float64]
JnpFloatDType = Any
JnpKey = jnp.ndarray
JnpPreciseFloat = jnp.float64


JNP_FLOAT_LIST = [jnp.bfloat16, jnp.float32, jnp.float64]
DTYPE_KEY = "dtype"


def get_dtype(spec: data_pb2.FloatDTypeSpec) -> Optional[JnpFloatDType]:
  """Returns the JAX dtype corresponding to the given proto enum.

  Args:
    spec: the specification for the data type. See float_dtype.proto.

  Returns:
    The JAX dtype.

  Raises:
    AttributeError: if spec is not known.
  """

  if spec == data_pb2.NODTYPE:
    return None
  elif spec == data_pb2.FLOAT64:
    return jnp.float64
  elif spec == data_pb2.FLOAT32:
    return jnp.float32
  elif spec == data_pb2.BFLOAT16:
    return jnp.bfloat16
  else:
    raise AttributeError(
        "Requested cast data type invalid. Did you set the dtype field in "
        "the config?"
    )


def get_dtype_spec(dtype: Optional[JnpFloatDType]) -> data_pb2.FloatDTypeSpec:
  """Returns the proto enum corresponding to the given JAX dtype.

  Args:
    dtype: the JAX dtype.

  Returns:
    The specification for the data type. See float_dtype.proto.

  Raises:
    AttributeError: if spec is not known.
  """
  if dtype is None:
    return data_pb2.NODTYPE
  elif dtype == jnp.float64:
    return data_pb2.FLOAT64
  elif dtype == jnp.float32:
    return data_pb2.FLOAT32
  elif dtype == jnp.bfloat16:
    return data_pb2.BFLOAT16
  else:
    raise AttributeError("Requested jnp dtype invalid.")


def check_labels_dtype(labels: List[jnp.ndarray], dtype: JnpFloatDType):
  for labels_tensor in labels:
    if labels_tensor.dtype != dtype:
      raise RuntimeError("Label dtype mismatch.")


def check_fitnesses_dtype(fitnesses: List[JnpFloat], dtype: JnpFloatDType):
  for fitness in fitnesses:
    if fitness.dtype != dtype:
      raise RuntimeError("Fitness dtype mismatch.")


def check_learnable_params_dtype(
    learnable_params: graph_lib.LearnableParams, dtype: JnpFloatDType
):
  for param in learnable_params.values():
    if param.dtype != dtype:
      raise RuntimeError(
          f"Learnable params dtype mismatch. {param.dtype} {dtype}"
      )


def cast_learnable_params(
    learnable_params: graph_lib.LearnableParams, dtype: JnpFloatDType
) -> graph_lib.LearnableParams:
  return {k: jnp.array(v, dtype=dtype) for k, v in learnable_params.items()}
