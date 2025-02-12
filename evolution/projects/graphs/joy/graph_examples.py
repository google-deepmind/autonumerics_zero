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

"""Hard-coded graphs.

Useful for unit testing and for reference.

Freezing tag conventions.
  FV: freeze vertex (cannot be deleted)
  FAIE: freeze all input edges (the edges cannot be disconnected at either end).
  FIE_: where `_` is the index of the edge to freeze, e.g. FIE2.
  FAOE: freeze all output edges.
"""

import itertools
from typing import Dict, List, Optional, Tuple

from jax import numpy as jnp
import numpy as np

from evolution.lib.python import rng as rng_lib
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs import op as op_lib
from evolution.projects.graphs.joy import baselines
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import interpretation


JnpFloatDType = data_lib.JnpFloatDType
JnpPreciseFloat = data_lib.JnpPreciseFloat


def identity() -> Tuple[graph_lib.Graph, Dict[str, JnpPreciseFloat]]:
  """Returns a graph connecting input to output directly."""
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[("x", "ProduceXOp"), ("f", "ConsumeFOp")],
      desired_edges=[(["x"], "f")],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
  )
  return graph, {}


def zeroth_order_taylor() -> Tuple[graph_lib.Graph, Dict[str, JnpPreciseFloat]]:
  """Returns the Taylor series for cos(x+y) to zeroth order."""
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("1", "RandomInitVariableOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[(["1"], "f")],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
  )
  learnable_params = {"1": JnpPreciseFloat("1.0")}
  return graph, learnable_params


def second_order_taylor() -> Tuple[graph_lib.Graph, Dict[str, JnpPreciseFloat]]:
  """Returns the Taylor series for cos(x+y) to second order."""
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("1", "RandomInitVariableOp"),
          ("-1/2", "RandomInitVariableOp"),
          ("x^2", "MultOp"),
          ("-1/2*x^2", "MultOp"),
          ("1-1/2*x^2", "AddOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[
          (["x", "x"], "x^2"),
          (["-1/2", "x^2"], "-1/2*x^2"),
          (["1", "-1/2*x^2"], "1-1/2*x^2"),
          (["1-1/2*x^2"], "f"),
      ],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
  )
  learnable_params = {
      "1": JnpPreciseFloat("1.0"),
      "-1/2": JnpPreciseFloat("-0.5"),
  }
  return graph, learnable_params


def fourth_order_taylor() -> Tuple[graph_lib.Graph, Dict[str, JnpPreciseFloat]]:
  """Returns the Taylor series for cos(x+y) to fourth order."""
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("1", "RandomInitVariableOp"),
          ("-1/2", "RandomInitVariableOp"),
          ("1/24", "RandomInitVariableOp"),
          ("x^2", "MultOp"),
          ("-1/2*x^2", "MultOp"),
          ("x^4", "MultOp"),
          ("1/24*x^4", "MultOp"),
          ("1-1/2*x^2", "AddOp"),
          ("1-1/2*x^2+1/24*x^4", "AddOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[
          (["x", "x"], "x^2"),
          (["-1/2", "x^2"], "-1/2*x^2"),
          (["x^2", "x^2"], "x^4"),
          (["1/24", "x^4"], "1/24*x^4"),
          (["1", "-1/2*x^2"], "1-1/2*x^2"),
          (["1-1/2*x^2", "1/24*x^4"], "1-1/2*x^2+1/24*x^4"),
          (["1-1/2*x^2+1/24*x^4"], "f"),
      ],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
  )
  learnable_params = {
      "1": JnpPreciseFloat("1.0"),
      "-1/2": JnpPreciseFloat("-0.5"),
      "1/24": JnpPreciseFloat("0.041667"),
  }
  return graph, learnable_params


def two_x() -> Tuple[graph_lib.Graph, Dict[str, JnpPreciseFloat]]:
  """Returns c * x to second order, with c is learnable and set to 2."""
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("c", "RandomInitVariableOp"),
          ("*", "MultOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[(["x", "c"], "*"), (["*"], "f")],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
  )
  learnable_params = {"c": JnpPreciseFloat("2.0")}
  return graph, learnable_params


def small_angle_baseline() -> (
    Tuple[graph_lib.Graph, Dict[str, JnpPreciseFloat]]
):
  """Returns a popular method for computing sin(x) for small angles.

  This baseline is a compact way of writing the Taylor series up and including
  the 11th order term. We consider small angles those in the interval
  [1.49011612e-8, 0.126].

  Returns:
    The graph and dict of constants.
  """
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("c1", "RandomInitVariableOp"),
          ("c2", "RandomInitVariableOp"),
          ("c3", "RandomInitVariableOp"),
          ("c4", "RandomInitVariableOp"),
          ("c5", "RandomInitVariableOp"),
          ("x2", "MultOp"),
          ("x3", "MultOp"),
          ("v0", "MultOp"),
          ("v1", "SubOp"),
          ("v2", "MultOp"),
          ("v3", "AddOp"),
          ("v4", "MultOp"),
          ("v5", "SubOp"),
          ("v6", "MultOp"),
          ("v7", "AddOp"),
          ("v8", "MultOp"),
          ("v9", "SubOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[
          (["x", "x"], "x2"),
          (["x2", "x"], "x3"),
          (["c5", "x2"], "v0"),
          (["v0", "c4"], "v1"),
          (["v1", "x2"], "v2"),
          (["v2", "c3"], "v3"),
          (["v3", "x2"], "v4"),
          (["v4", "c2"], "v5"),
          (["v5", "x2"], "v6"),
          (["v6", "c1"], "v7"),
          (["v7", "x3"], "v8"),
          (["x", "v8"], "v9"),
          (["v9"], "f"),
      ],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
  )
  learnable_params = {
      "c1": JnpPreciseFloat("1") / JnpPreciseFloat("6"),  # 1/3!
      "c2": JnpPreciseFloat("1") / JnpPreciseFloat("120"),  # 1/5!
      "c3": JnpPreciseFloat("1") / JnpPreciseFloat("5040"),  # 1/7!
      "c4": JnpPreciseFloat("1") / JnpPreciseFloat("362880"),  # 1/9!
      "c5": JnpPreciseFloat("1") / JnpPreciseFloat("39916800"),
  }  # 1/11!
  return graph, learnable_params


def constant_small_angle_baseline():
  """Returns a popular method for computing sin(x) for small angles.

  This baseline is a compact way of writing the Taylor series up and including
  the 11th order term with learnable constants. We consider small angles those
  in the interval [1.49011612e-8, 0.126].

  Returns:
    The graph and dict of constants.
  """
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("c1", "ConstantOp"),
          ("c2", "ConstantOp"),
          ("c3", "ConstantOp"),
          ("c4", "ConstantOp"),
          ("c5", "ConstantOp"),
          ("x2", "MultOp"),
          ("x3", "MultOp"),
          ("v0", "MultOp"),
          ("v1", "SubOp"),
          ("v2", "MultOp"),
          ("v3", "AddOp"),
          ("v4", "MultOp"),
          ("v5", "SubOp"),
          ("v6", "MultOp"),
          ("v7", "AddOp"),
          ("v8", "MultOp"),
          ("v9", "SubOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[
          (["x", "x"], "x2"),
          (["x2", "x"], "x3"),
          (["c5", "x2"], "v0"),
          (["v0", "c4"], "v1"),
          (["v1", "x2"], "v2"),
          (["v2", "c3"], "v3"),
          (["v3", "x2"], "v4"),
          (["v4", "c2"], "v5"),
          (["v5", "x2"], "v6"),
          (["v6", "c1"], "v7"),
          (["v7", "x3"], "v8"),
          (["x", "v8"], "v9"),
          (["v9"], "f"),
      ],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
      evolvable_params={
          "c1": JnpPreciseFloat(1.0) / JnpPreciseFloat(6.0),  # 1/3!
          "c2": JnpPreciseFloat(1.0) / JnpPreciseFloat(120.0),  # 1/5!
          "c3": JnpPreciseFloat(1.0) / JnpPreciseFloat(5040.0),  # 1/7!
          "c4": JnpPreciseFloat(1.0) / JnpPreciseFloat(362880.0),  # 1/9!
          "c5": JnpPreciseFloat(1.0) / JnpPreciseFloat(39916800.0),
      },  # 1/11!
  )
  learnable_params = {}
  return graph, learnable_params


def zeroed_constant_small_angle_baseline():
  """Returns a popular method for computing sin(x) for small angles.

  This baseline is a compact way of writing the Taylor series up and including
  the 11th order term with learnable constants. We consider small angles those
  in the interval [1.49011612e-8, 0.126].

  The constants have been zeroed.

  Returns:
    The graph and dict of constants.
  """
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("c1", "ConstantOp"),
          ("c2", "ConstantOp"),
          ("c3", "ConstantOp"),
          ("c4", "ConstantOp"),
          ("c5", "ConstantOp"),
          ("x2", "MultOp"),
          ("x3", "MultOp"),
          ("v0", "MultOp"),
          ("v1", "SubOp"),
          ("v2", "MultOp"),
          ("v3", "AddOp"),
          ("v4", "MultOp"),
          ("v5", "SubOp"),
          ("v6", "MultOp"),
          ("v7", "AddOp"),
          ("v8", "MultOp"),
          ("v9", "SubOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[
          (["x", "x"], "x2"),
          (["x2", "x"], "x3"),
          (["c5", "x2"], "v0"),
          (["v0", "c4"], "v1"),
          (["v1", "x2"], "v2"),
          (["v2", "c3"], "v3"),
          (["v3", "x2"], "v4"),
          (["v4", "c2"], "v5"),
          (["v5", "x2"], "v6"),
          (["v6", "c1"], "v7"),
          (["v7", "x3"], "v8"),
          (["x", "v8"], "v9"),
          (["v9"], "f"),
      ],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
      evolvable_params={
          "c1": JnpPreciseFloat(0.0),
          "c2": JnpPreciseFloat(0.0),
          "c3": JnpPreciseFloat(0.0),
          "c4": JnpPreciseFloat(0.0),
          "c5": JnpPreciseFloat(0.0),
      },
  )
  learnable_params = {}
  return graph, learnable_params


def constant_small_angle_baseline_order9():
  """Returns the 9th order Tarylor expansion of sin(x) for small angles.

  This baseline is a compact way of writing the Taylor series up and including
  the 9th order term with learnable constants. We consider small angles those
  in the interval [1.49011612e-8, 0.126].

  Returns:
    The graph and dict of constants.
  """
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("c1", "ConstantOp"),
          ("c2", "ConstantOp"),
          ("c3", "ConstantOp"),
          ("c4", "ConstantOp"),
          ("x2", "MultOp"),
          ("x3", "MultOp"),
          ("v2", "MultOp"),
          ("v3", "SubOp"),
          ("v4", "MultOp"),
          ("v5", "SubOp"),
          ("v6", "MultOp"),
          ("v7", "AddOp"),
          ("v8", "MultOp"),
          ("v9", "SubOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[
          (["x", "x"], "x2"),
          (["x2", "x"], "x3"),
          (["c4", "x2"], "v2"),
          (["c3", "v2"], "v3"),
          (["v3", "x2"], "v4"),
          (["v4", "c2"], "v5"),
          (["v5", "x2"], "v6"),
          (["v6", "c1"], "v7"),
          (["v7", "x3"], "v8"),
          (["x", "v8"], "v9"),
          (["v9"], "f"),
      ],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
      evolvable_params={
          "c1": JnpPreciseFloat(1.0) / JnpPreciseFloat(6.0),  # 1/3!
          "c2": JnpPreciseFloat(1.0) / JnpPreciseFloat(120.0),  # 1/5!
          "c3": JnpPreciseFloat(1.0) / JnpPreciseFloat(5040.0),  # 1/7!
          "c4": JnpPreciseFloat(1.0) / JnpPreciseFloat(362880.0),
      },  # 1/9!
  )
  learnable_params = {}
  return graph, learnable_params


def constant_small_angle_baseline_order7():
  """Returns the 9th order Tarylor expansion of sin(x) for small angles.

  This baseline is a compact way of writing the Taylor series up and including
  the 7th order term with learnable constants. We consider small angles those
  in the interval [1.49011612e-8, 0.126].

  Returns:
    The graph and dict of constants.
  """
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("c1", "ConstantOp"),
          ("c2", "ConstantOp"),
          ("c3", "ConstantOp"),
          ("x2", "MultOp"),
          ("x3", "MultOp"),
          ("v4", "MultOp"),
          ("v5", "SubOp"),
          ("v6", "MultOp"),
          ("v7", "AddOp"),
          ("v8", "MultOp"),
          ("v9", "SubOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[
          (["x", "x"], "x2"),
          (["x2", "x"], "x3"),
          (["c3", "x2"], "v4"),
          (["v4", "c2"], "v5"),
          (["v5", "x2"], "v6"),
          (["v6", "c1"], "v7"),
          (["v7", "x3"], "v8"),
          (["x", "v8"], "v9"),
          (["v9"], "f"),
      ],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
      evolvable_params={
          "c1": JnpPreciseFloat(1.0) / JnpPreciseFloat(6.0),  # 1/3!
          "c2": JnpPreciseFloat(1.0) / JnpPreciseFloat(120.0),  # 1/5!
          "c3": JnpPreciseFloat(1.0) / JnpPreciseFloat(5040.0),
      },  # 1/7!
  )
  learnable_params = {}
  return graph, learnable_params


def erf_large_regime_leading_seed() -> (
    Tuple[graph_lib.Graph, Dict[str, JnpPreciseFloat]]
):
  """Returns a graph that can be used to seed the evolution of erf.

  Provides the leading asymptotic behavior. Namely, this graph encodes:
    f(x) = 1 - exp(-x^2) / x / sqrt(pi) * G(x),
  where G(x) = 1.

  In order to only allow G(x) to be evolved, freeze these vertices:
    "decay", "one", "mult", "sub",
  and these edges:
    ("x", "decay"), ("decay", "mult"), ("one", "sub"), ("mult", "sub"),
    ("sub", "f").
  """
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("decay", "ErfcAsymptoticBehaviorOp"),
          ("one", "OneOp"),
          ("mult", "MultOp"),
          ("sub", "SubOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[
          (["x"], "decay"),
          (["decay", "one"], "mult"),
          (["one", "mult"], "sub"),
          (["sub"], "f"),
      ],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=[
          "f",
      ],
      op_init_params=None,
  )
  learnable_params = {}
  return graph, learnable_params


def erf_large_regime_minimal_seed() -> (
    Tuple[graph_lib.Graph, Dict[str, JnpPreciseFloat]]
):
  """Returns a graph that can be used to seed the evolution of erf.

  Provides the exponential needed for correct asymptotic behavior but not hints
  on how to use it. Namely, this graph encodes:
    f(x) = G(x, exp(-x^2))
  where G(x, y) = 0.

  In order to only allow G(x) to be evolved, freeze these vertices:
    "exp"
  and these edges:
    ("x", "exp").
  """
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("exp", "ErfcExponentialOp"),
          ("zero", "ZeroOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[(["x"], "exp"), (["zero"], "f")],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=[
          "f",
      ],
      op_init_params=None,
  )
  learnable_params = {}
  return graph, learnable_params


def zero_root_graph() -> graph_lib.Graph:
  """A graph to provide the root input to continued fractions.

  This graph has no inputs and only one output that always produces zero.

  For example, can be used as the graph R in the continued representation
  H(x, G^N(x, R)), for example when approximating erf(x).

  Returns:
    The graph.
  """
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[("0", "ZeroOp"), ("f", "ConsumeFOp")],
      desired_edges=[(["0"], "f")],
      required_input_vertex_ids=[],
      required_output_vertex_ids=["f"],
      op_init_params=None,
  )
  return graph


def erf_following_graph() -> graph_lib.Graph:
  """Can be composed with a graph to provide the asymptotic behavior of erf.

  This graph is H(x, y) = 1 - exp(-x^2) / x / sqrt(pi) * y.

  Therefore this graph can be used to transform a graph G into H(x, G(x))
  before evaluation, allowing evolving G with arithmetic operations.

  It can also be used in a continued fraction representation H(x, G^N(x, R)) as
  the graph H.

  Returns:
    The graph H in the description above.
  """
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("y", "ProduceYOp"),
          ("decay", "ErfcAsymptoticBehaviorOp"),
          ("mult", "MultOp"),
          ("one", "OneOp"),
          ("sub", "SubOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[
          (
              [
                  "x",
              ],
              "decay",
          ),
          (["decay", "y"], "mult"),
          (["one", "mult"], "sub"),
          (["sub"], "f"),
      ],
      required_input_vertex_ids=["x", "y"],
      required_output_vertex_ids=["f"],
      op_init_params=None,
  )
  return graph


def fp_erf_following_graph() -> graph_lib.Graph:
  """Can be composed with a graph to provide the asymptotic behavior of erf.

  This graph is H(x, y) = 1 - exp(-x^2) / x / sqrt(pi) * y.

  Therefore this graph can be used to transform a graph G into H(x, G(x))
  before evaluation, allowing evolving G with arithmetic operations. It can also
  be used in a continued fraction representation H(x, G^N(x, R)) as the graph H.

  This graph belongs in the double-word search space, but uses fp ops
  only.

  Returns:
    The graph H in the description above.
  """
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceDwXOp"),
          ("y", "ProduceDwYOp"),
          ("decay", "ErfcFpAsymptoticBehaviorOp"),
          ("mult", "FpMultFpOp"),
          ("one", "DwOneOp"),
          ("sub", "FpSubFpOp"),
          ("f", "ConsumeDwFOp"),
      ],
      desired_edges=[
          (
              [
                  "x",
              ],
              "decay",
          ),
          (["decay", "y"], "mult"),
          (["one", "mult"], "sub"),
          (["sub"], "f"),
      ],
      required_input_vertex_ids=["x", "y"],
      required_output_vertex_ids=["f"],
      op_init_params=None,
  )
  return graph


def erf_continued_fraction_module() -> graph_lib.Graph:
  """Graph G in H(x, G^N(x, R)) for the continued fraction of erf(x)."""
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("y", "ProduceYOp"),
          ("a", "RandomInitVariableOp"),
          ("b", "RandomInitVariableOp"),
          ("c", "RandomInitVariableOp"),
          ("mult", "MultOp"),
          ("add1", "AddOp"),
          ("div1", "DivOp"),
          ("add2", "AddOp"),
          ("div2", "DivOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[
          (["x", "x"], "mult"),
          (["mult", "y"], "add1"),
          (["a", "add1"], "div1"),
          (["b", "div1"], "add2"),
          (["c", "add2"], "div2"),
          (["div2"], "f"),
      ],
      required_input_vertex_ids=["x", "y"],
      required_output_vertex_ids=["f"],
      op_init_params=None,
  )
  return graph


def get_reference5_finalized_fn(
    dtype: JnpFloatDType,
) -> interpretation.FinalizedFn:
  """Returns the reference function used to normalize the execution times.

  Originally obtained with Sollya with 32-bit coefficients as a 5th order
  polynomial to approximate 2^x. However, we use it for any dtype with the
  purpose of measuring speed.

  Args:
    dtype: the desired dtype for the calculations.

  Returns:
    The finalized function.
  """

  def finalized_fn(x):
    """A Horner-scheme 8th-order polynomial."""
    assert x[0].dtype == dtype
    coeffs = [
        0.999999940395355224609375,
        0.693152725696563720703125,
        0.2401557266712188720703125,
        5.58212064206600189208984375e-2,
        8.994684554636478424072265625e-3,
        1.875563524663448333740234375e-3,
    ]
    coeffs = [jnp.array(coeff, dtype=dtype) for coeff in coeffs]
    f = coeffs.pop()
    assert len(x) == 1
    for a in reversed(coeffs):
      f = jnp.add(a, jnp.multiply(x[0], f))
    return f

  return finalized_fn


def get_reference7_finalized_fn(
    dtype: JnpFloatDType,
) -> interpretation.FinalizedFn:
  """Returns the reference function used to normalize the execution times.

  Originally obtained with Sollya with 32-bit coefficients as a 7th order
  polynomial to approximate 2^x. However, we use it for any dtype with the
  purpose of measuring speed.

  Args:
    dtype: the desired dtype for the calculations.

  Returns:
    The finalized function.
  """

  def finalized_fn(x):
    """A Horner-scheme 8th-order polynomial."""
    assert x[0].dtype == dtype
    coeffs = [
        0.693147182464599609375,
        0.24022643268108367919921875,
        5.550487339496612548828125e-2,
        9.61467809975147247314453125e-3,
        1.341356779448688030242919921875e-3,
        1.44149686093442142009735107421875e-4,
        2.13267994695343077182769775390625e-5,
    ]
    coeffs = [jnp.array(coeff, dtype=dtype) for coeff in coeffs]
    f = coeffs.pop()
    assert len(x) == 1
    for a in reversed(coeffs):
      f = jnp.add(a, jnp.multiply(x[0], f))
    return f

  return finalized_fn


def get_reference8_finalized_fn(
    dtype: JnpFloatDType,
) -> interpretation.FinalizedFn:
  """Returns the reference function used to normalize the execution times.

  Originally obtained with Sollya with 32-bit coefficients as a 8th order
  polynomial to approximate 2^x. However, we use it for any dtype with the
  purpose of measuring speed.

  Args:
    dtype: the desired dtype for the calculations.

  Returns:
    The finalized function.
  """

  def finalized_fn(x):
    """A Horner-scheme 8th-order polynomial."""
    assert x[0].dtype == dtype
    coeffs = [
        1.0,
        0.693147182464599609375,
        0.24022646248340606689453125,
        5.5504463613033294677734375e-2,
        9.616785682737827301025390625e-3,
        1.336037297733128070831298828125e-3,
        1.5121899195946753025054931640625e-4,
        1.657833854551427066326141357421875e-5,
        1.27110934045049361884593963623046875e-6,
    ]
    coeffs = [jnp.array(coeff, dtype=dtype) for coeff in coeffs]
    f = coeffs.pop()
    assert len(x) == 1
    for a in reversed(coeffs):
      f = jnp.add(a, jnp.multiply(x[0], f))
    return f

  return finalized_fn


def n_order_poly_horner_ratio(
    n: int,
    d: int,
    numerator_coefficient_op_ids: Optional[List[str]] = None,
    denominator_coefficient_op_ids: Optional[List[str]] = None,
    numerator_coefficients: Optional[List[data_lib.JnpFloatDType]] = None,
    denominator_coefficients: Optional[List[data_lib.JnpFloatDType]] = None,
    rng=np.random.RandomState(rng_lib.GenerateRNGSeed()),
) -> Tuple[graph_lib.Graph, Optional[Dict[str, JnpPreciseFloat]]]:
  """Returns the ratio of P(n)/Q(d).

  P and Q are Horner-scheme nth-order and dth-order polynomials.

  Args:
    n: the degree of the numerator polynomial P.
    d: the degree of the denominator polynomial Q.
    numerator_coefficient_op_ids: the op_ids to use to generate the coefficient
      vertices. If set the list must either contain a single element (which will
      be the op_id for all coefficients) or n+1 elements.
    denominator_coefficient_op_ids: the op_ids to use to generate the
      coefficient vertices. If set the list must either contain a single element
      (which will be the op_id for all coefficients) or n+1 elements.
    numerator_coefficients: the coefficients to use with the numerator
      polynomial. Optional.
    denominator_coefficients: the coefficients to use with the denominator
      polynomial. Optional.
    rng: numpy random state.

  Returns:
    A graph representing a P(n)/Q(d) ratio.
  """
  if n < 0 or d < 0:
    raise ValueError("n should have a non-negative value.")
  if (denominator_coefficients or numerator_coefficients) and not (
      denominator_coefficients and numerator_coefficients
  ):
    raise ValueError(
        "coefficients should be set for both "
        "numerator and denominator or not at all."
    )
  if (
      numerator_coefficients is not None
      and len(numerator_coefficients) != n + 1
  ):
    raise ValueError("there should be n + 1 numerator coefficients.")
  if (
      denominator_coefficients is not None
      and len(denominator_coefficients) != d + 1
  ):
    raise ValueError("there should be d + 1 denominator coefficients.")
  numerator_coefficient_op_ids = numerator_coefficient_op_ids or [
      "RandomInitVariableOp"
  ]
  denominator_coefficient_op_ids = denominator_coefficient_op_ids or [
      "RandomInitVariableOp"
  ]

  if len(numerator_coefficient_op_ids) not in [1, n + 1]:
    raise ValueError("there should be 1 or n + 1 numerator coefficient op ids.")
  if len(denominator_coefficient_op_ids) not in [1, d + 1]:
    raise ValueError("there should be d + 1 denominator coefficient op ids.")

  numerator_edges, numerator_vertices = n_horner_edges_and_vertices(
      numerator_coefficient_op_ids, n, "p"
  )
  denominator_edges, denominator_vertices = n_horner_edges_and_vertices(
      denominator_coefficient_op_ids, d, "q"
  )

  graph = graph_lib.Graph()
  numerator_coefficient_ops = [
      op_lib.Op.build_op(op_id, None) for op_id in numerator_coefficient_op_ids
  ]
  denominator_coefficient_ops = [
      op_lib.Op.build_op(op_id, None)
      for op_id in denominator_coefficient_op_ids
  ]

  evolvable_params = {}
  if numerator_coefficients and denominator_coefficients:
    assert all(
        op.has_evolvable_params
        for op in itertools.chain(
            numerator_coefficient_ops, denominator_coefficient_ops
        )
    )
    evolvable_params = {
        "p" + str(i): numerator_coefficients[i] for i in range(n + 1)
    }
    evolvable_params.update(
        {"q" + str(i): denominator_coefficients[i] for i in range(d + 1)}
    )
  else:
    # If coefficients are not set we need to verify whether the chosen op_id
    # needs evolvable_params and initialize them if it does
    for i, op in zip(range(n + 1), itertools.cycle(numerator_coefficient_ops)):
      if op.has_evolvable_params:
        evolvable_params["p" + str(i)] = op.create_evolvable_params(rng)
    for i, op in zip(
        range(n + 1), itertools.cycle(denominator_coefficient_ops)
    ):
      if op.has_evolvable_params:
        evolvable_params["q" + str(i)] = op.create_evolvable_params(rng)

  desired_vertices = [("x", "ProduceXOp"), ("f", "ConsumeFOp")]
  desired_vertices = (
      desired_vertices + numerator_vertices + denominator_vertices
  )
  desired_vertices.append(("P/Q", "DivOp"))
  desired_edges = numerator_edges + denominator_edges

  numerator_vertex = "p0" if n == 0 else "p0+..."
  denominator_vertex = "q0" if n == 0 else "q0+..."
  desired_edges.append(([numerator_vertex, denominator_vertex], "P/Q"))

  desired_edges.append((["P/Q"], "f"))

  graph.parse(
      desired_vertices=desired_vertices,
      desired_edges=desired_edges,
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
      evolvable_params=evolvable_params,
  )

  return graph, None


def n_order_poly_horner(
    n: int,
    coefficient_op_ids: Optional[List[str]] = None,
    coefficients: Optional[List[JnpFloatDType]] = None,
    rng=np.random.RandomState(rng_lib.GenerateRNGSeed()),
) -> Tuple[graph_lib.Graph, Optional[Dict[str, JnpPreciseFloat]]]:
  """Returns the Horner-scheme nth-order polynomial."""
  coefficient_op_ids = coefficient_op_ids or ["RandomInitVariableOp"]

  if n < 0:
    raise ValueError("n should have a non-negative value.")
  if coefficients is not None and len(coefficients) != n + 1:
    raise ValueError("there should be n+1 coefficients.")

  if len(coefficient_op_ids) not in [1, n + 1]:
    raise ValueError("there should be 1 or n + 1 coefficient op ids.")

  desired_edges, desired_vertices = n_horner_edges_and_vertices(
      coefficient_op_ids, n, "a"
  )
  desired_vertices += [("x", "ProduceXOp"), ("f", "ConsumeFOp")]

  if n == 0:
    desired_edges.append((["a0"], "f"))
  else:
    desired_edges.append((["a0+..."], "f"))

  graph = graph_lib.Graph()
  coefficient_ops = [
      op_lib.Op.build_op(coefficient_op_id, None)
      for coefficient_op_id in coefficient_op_ids
  ]
  if coefficients:
    assert all(
        coefficient_op.has_evolvable_params
        for coefficient_op in coefficient_ops
    )
    evolvable_params = {"a" + str(i): coefficients[i] for i in range(n + 1)}
  else:
    # If coefficients are not set we need to verify whether the chosen op_id
    # needs evolvable_params and initialize them if it does
    evolvable_params = {}
    for i, coefficient_op in zip(
        range(n + 1), itertools.cycle(coefficient_ops)
    ):
      if coefficient_op.has_evolvable_params:
        evolvable_params["a" + str(i)] = coefficient_op.create_evolvable_params(
            rng
        )

  graph.parse(
      desired_vertices=desired_vertices,
      desired_edges=desired_edges,
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
      evolvable_params=evolvable_params,
  )

  return graph, None


def n_horner_edges_and_vertices(
    coefficient_op_ids: List[str], n: int, vertex_prefix: str = "a"
) -> Tuple[List[Tuple[List[str], str]], List[Tuple[str, str]]]:
  """Generates edges and vertices for an n-th order Horner-scheme polynomial.

  Args:
    coefficient_op_ids: op_ids of the coefficient vertices.
    n: the degree of the polynomial.
    vertex_prefix: the prefix that will be used to name the vertices. For each
      coefficient the corresponding vertex will have name
      "{vertex_prefix}{exponent}".

  Returns:
    A list of tuples representing edges and a list of tuples representing
    vertices that can be used as argument to graph.parse.
  """

  # Adding the different vertices. For Horner scheme polynomial of order n,
  # there would "n+1" constants, "n" multiply ops and "n" addition ops. We will
  # also have one ProduceOp and a ConsumeOp.
  desired_vertices = []
  for i, coefficient_op_id in zip(
      range(n + 1), itertools.cycle(coefficient_op_ids)
  ):
    desired_vertices.append((f"{vertex_prefix}" + str(i), coefficient_op_id))
  for i in range(1, n + 1):
    desired_vertices.append(
        (f"x*({vertex_prefix}" + str(i) + "+...)", "MultOp")
    )
  for i in range(0, n):
    desired_vertices.append((f"{vertex_prefix}" + str(i) + "+...", "AddOp"))
  # Adding the edges. For a horner scheme polynomial representation, in our
  # graph, we will have "n" edges for each of the "n" multiply ops and another
  # "n" edges for "n" addition ops. We will also need to include a final edge
  # to the ConsumeOp "f".
  desired_edges = []
  for i in range(1, n + 1):
    source_vertex1 = "x"
    if i == n:
      source_vertex2 = f"{vertex_prefix}" + str(i)
    else:
      source_vertex2 = f"{vertex_prefix}" + str(i) + "+..."
    destination_vertex = f"x*({vertex_prefix}" + str(i) + "+...)"
    desired_edges.append(([source_vertex1, source_vertex2], destination_vertex))
  for i in range(0, n):
    source_vertex1 = f"{vertex_prefix}" + str(i)
    source_vertex2 = f"x*({vertex_prefix}" + str(i + 1) + "+...)"
    destination_vertex = f"{vertex_prefix}" + str(i) + "+..."
    desired_edges.append(([source_vertex1, source_vertex2], destination_vertex))
  return desired_edges, desired_vertices


def exp2_float64_sollya_baselines_graphs() -> List[graph_lib.Graph]:
  """Creates a list of Graphs, each representing a sollya exp2 64b baseline."""

  coefficient_functions = [
      baselines.exp64_poly_sollya_5_coefficients,
      baselines.exp64_poly_sollya_6_coefficients,
      baselines.exp64_poly_sollya_7_coefficients,
      baselines.exp64_poly_sollya_8_coefficients,
      baselines.exp64_poly_sollya_9_coefficients,
      baselines.exp64_poly_sollya_10_coefficients,
      baselines.exp64_poly_sollya_11_coefficients,
      baselines.exp64_poly_sollya_12_coefficients,
      baselines.exp64_poly_sollya_13_coefficients,
      baselines.exp64_poly_sollya_14_coefficients,
      baselines.exp64_poly_sollya_15_coefficients,
      baselines.exp64_poly_sollya_16_coefficients,
      baselines.exp64_poly_sollya_17_coefficients,
      baselines.exp64_poly_sollya_18_coefficients,
      baselines.exp64_poly_sollya_19_coefficients,
      baselines.exp64_poly_sollya_20_coefficients,
      baselines.exp64_poly_sollya_21_coefficients,
      baselines.exp64_poly_sollya_22_coefficients,
      baselines.exp64_poly_sollya_23_coefficients,
  ]

  graphs = []
  for coefficient_function in coefficient_functions:
    coefficients = coefficient_function()
    n = len(coefficients) - 1
    graph, _ = n_order_poly_horner(
        n,
        coefficient_op_ids=["PositiveScale16AnchoredVariableOp"],
        coefficients=coefficients,
    )
    graphs.append(graph)
  return graphs
