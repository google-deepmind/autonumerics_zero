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

"""JAX baselines.

Notes:
-All Sollya polynomials have coefficients constrained to the target precision.
-All polynomials are in Horner's scheme.
-The numerators and denominators of rational functions are in Horner's scheme.
-Coefficients of 0, 1 and terms of 0 have been simplified away.
"""

from typing import List, Optional, Tuple
import jax
from jax import numpy as jnp
from evolution.projects.graphs import graph as graph_lib
from evolution.projects.graphs.joy import data as data_lib
from evolution.projects.graphs.joy import interpretation

FinalizedFn = interpretation.FinalizedFn
JnpFloat = data_lib.JnpFloat
JnpPreciseFloat = data_lib.JnpPreciseFloat


def baselines(baseline_id: str):
  """Returns a table of baselines.

  Args:
    baseline_id: which function to approximate.

  Returns:
    A list of (name, finalized_fn).
  """
  if baseline_id == "table_exp2_16bits":
    results = table_exp2_16bits_baselines()
  elif baseline_id == "table_exp2_32bits":
    results = table_exp2_32bits_baselines()
  elif baseline_id == "table_log2_32bits":
    results = table_log2_32bits_baselines()
  elif baseline_id == "table_erf_32bits_0to1":
    results = table_erf_32bits_0to1_baselines()
  elif baseline_id == "table_erf_32bits_1to5":
    results = table_erf_32bits_1to5_baselines()
  elif baseline_id == "table_exp2_64bits":
    results = table_exp2_64bits_baselines()
  elif baseline_id == "table_log2_64bits":
    results = table_log2_64bits_baselines()
  elif baseline_id == "table_erf_64bits_0to1":
    results = table_erf_64bits_0to1_baselines()
  elif baseline_id == "table_erf_64bits_1to7":
    results = table_erf_64bits_1to7_baselines()
  elif baseline_id == "correct_on_the_fly_expe_32bits":
    results = correct_on_the_fly_expe_32bits_baselines()
  elif baseline_id == "correct_on_the_fly_loge_32bits":
    results = correct_on_the_fly_loge_32bits_baselines()
  else:
    raise NotImplementedError("Missing baselines for target.")
  for name, _ in results:
    if len(name) > 32:
      raise ValueError("Name won't fit in individual_id Spanner field.")
  return results


def table_exp2_16bits_baselines() -> List[Tuple[str, FinalizedFn]]:
  """Baselines for bfloat16 2^x on [0, 1].

  Returns:
    A list of (name, finalized_fn).
  """
  return [  # pytype: disable=bad-return-type  # jax-ndarray
      ("JAX Numpy", exp16_jnp),
      ("Polynomial, Sollya, 2nd order", exp16_poly_sollya_2),
      ("Polynomial, Sollya, 3rd order", exp16_poly_sollya_3),
      ("Polynomial, Sollya, 4th order", exp16_poly_sollya_4),
      ("Polynomial, Sollya, 5th order", exp16_poly_sollya_5),
      ("Polynomial, Sollya, 6th order", exp16_poly_sollya_6),
      ("Polynomial, Sollya, 7th order", exp16_poly_sollya_7),
      ("Polynomial, Sollya, 8th order", exp16_poly_sollya_8),
      ("Polynomial, Sollya, 9th order", exp16_poly_sollya_9),
      ("Polynomial, Sollya, 10th order", exp16_poly_sollya_10),
      ("Taylor expansion, 4th order", exp16_poly_taylor_4),
      ("Taylor expansion, 5th order", exp16_poly_taylor_5),
      ("Taylor expansion, 6th order", exp16_poly_taylor_6),
      ("Taylor expansion, 7th order", exp16_poly_taylor_7),
      ("Rational, Minimax, (2,2)-order", exp16_rational_minimax_2_2),
      ("Rational, Minimax, (3,3)-order", exp16_rational_minimax_3_3),
      ("Rational, Minimax, (4,4)-order", exp16_rational_minimax_4_4),
      ("Rational, Minimax, (5,5)-order", exp16_rational_minimax_5_5),
      ("Rational, Minimax, (6,6)-order", exp16_rational_minimax_6_6),
      ("Pade approximant, (1,1)-order", exp16_rational_pade_1_1),
      ("Pade approximant, (2,2)-order", exp16_rational_pade_2_2),
      ("Pade approximant, (3,3)-order", exp16_rational_pade_3_3),
      ("Pade approximant, (4,4)-order", exp16_rational_pade_4_4),
  ]


def exp16_jnp(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns JAX NumPy approximation."""
  return jnp.exp2(x[0])


def exp16_poly_sollya_2(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 5-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-07-03.exp16_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(
      x[0],
      [
          jnp.bfloat16(float.fromhex("0x1p0")),
          jnp.bfloat16(float.fromhex("0x1.54p-1")),
          jnp.bfloat16(float.fromhex("0x1.54p-2")),
      ],
  )


def exp16_poly_sollya_3(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 5-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-07-03.exp16_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(
      x[0],
      [
          jnp.bfloat16(float.fromhex("0x1p0")),
          jnp.bfloat16(float.fromhex("0x1.64p-1")),
          jnp.bfloat16(float.fromhex("0x1.dp-3")),
          jnp.bfloat16(float.fromhex("0x1.4p-4")),
      ],
  )


def exp16_poly_sollya_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 4-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-07-03.exp16_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(
      x[0],
      [
          jnp.bfloat16(float.fromhex("0x1p0")),
          jnp.bfloat16(float.fromhex("0x1.62p-1")),
          jnp.bfloat16(float.fromhex("0x1p-2")),
          jnp.bfloat16(float.fromhex("0x1.3cp-5")),
          jnp.bfloat16(float.fromhex("0x1.48p-6")),
      ],
  )


def exp16_poly_sollya_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 5-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-07-03.exp16_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(
      x[0],
      [
          jnp.bfloat16(float.fromhex("0x1p0")),
          jnp.bfloat16(float.fromhex("0x1.62p-1")),
          jnp.bfloat16(float.fromhex("0x1.06p-2")),
          jnp.bfloat16(float.fromhex("0x1.4ap-7")),
          jnp.bfloat16(float.fromhex("0x1p-4")),
          jnp.bfloat16(float.fromhex("-0x1.46p-6")),
      ],
  )


def exp16_poly_sollya_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 6-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-07-03.exp16_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(
      x[0],
      [
          jnp.bfloat16(float.fromhex("0x1p0")),
          jnp.bfloat16(float.fromhex("0x1.62p-1")),
          jnp.bfloat16(float.fromhex("0x1.0cp-2")),
          jnp.bfloat16(float.fromhex("-0x1.22p-5")),
          jnp.bfloat16(float.fromhex("0x1.74p-3")),
          jnp.bfloat16(float.fromhex("-0x1.3p-3")),
          jnp.bfloat16(float.fromhex("0x1.92p-5")),
      ],
  )


def exp16_poly_sollya_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 7-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-07-03.exp16_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(
      x[0],
      [
          jnp.bfloat16(float.fromhex("0x1p0")),
          jnp.bfloat16(float.fromhex("0x1.62p-1")),
          jnp.bfloat16(float.fromhex("0x1.12p-2")),
          jnp.bfloat16(float.fromhex("-0x1.a4p-4")),
          jnp.bfloat16(float.fromhex("0x1.ccp-2")),
          jnp.bfloat16(float.fromhex("-0x1.44p-1")),
          jnp.bfloat16(float.fromhex("0x1.d4p-2")),
          jnp.bfloat16(float.fromhex("-0x1.0ap-3")),
      ],
  )


def exp16_poly_sollya_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 8-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-07-03.exp16_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(
      x[0],
      [
          jnp.bfloat16(float.fromhex("0x1p0")),
          jnp.bfloat16(float.fromhex("0x1.62p-1")),
          jnp.bfloat16(float.fromhex("0x1.1cp-2")),
          jnp.bfloat16(float.fromhex("-0x1.dp-3")),
          jnp.bfloat16(float.fromhex("0x1.0ep0")),
          jnp.bfloat16(float.fromhex("-0x1.0ep1")),
          jnp.bfloat16(float.fromhex("0x1.3p1")),
          jnp.bfloat16(float.fromhex("-0x1.66p0")),
          jnp.bfloat16(float.fromhex("0x1.58p-2")),
      ],
  )


def exp16_poly_sollya_9(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 9-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-07-03.exp16_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(
      x[0],
      [
          jnp.bfloat16(float.fromhex("0x1p0")),
          jnp.bfloat16(float.fromhex("0x1.62p-1")),
          jnp.bfloat16(float.fromhex("0x1.1ep-2")),
          jnp.bfloat16(float.fromhex("-0x1.2ap-2")),
          jnp.bfloat16(float.fromhex("0x1.a8p0")),
          jnp.bfloat16(float.fromhex("-0x1.28p2")),
          jnp.bfloat16(float.fromhex("0x1.f8p2")),
          jnp.bfloat16(float.fromhex("-0x1.fcp2")),
          jnp.bfloat16(float.fromhex("0x1.16p2")),
          jnp.bfloat16(float.fromhex("-0x1.fcp-1")),
      ],
  )


def exp16_poly_sollya_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 10-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-07-03.exp16_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(
      x[0],
      [
          jnp.bfloat16(float.fromhex("0x1p0")),
          jnp.bfloat16(float.fromhex("0x1.62p-1")),
          jnp.bfloat16(float.fromhex("0x1.2cp-2")),
          jnp.bfloat16(float.fromhex("-0x1.bp-2")),
          jnp.bfloat16(float.fromhex("0x1.cp0")),
          jnp.bfloat16(float.fromhex("-0x1.cp0")),
          jnp.bfloat16(float.fromhex("-0x1.6p2")),
          jnp.bfloat16(float.fromhex("0x1.3p4")),
          jnp.bfloat16(float.fromhex("-0x1.8p4")),
          jnp.bfloat16(float.fromhex("0x1.c8p3")),
          jnp.bfloat16(float.fromhex("-0x1.a8p1")),
      ],
  )


def exp16_poly_taylor_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 4-th order Taylor polynomial about x=0."""
  return _exp16_poly_taylor(order=4, x=x)


def exp16_poly_taylor_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 5-th order Taylor polynomial about x=0."""
  return _exp16_poly_taylor(order=5, x=x)


def exp16_poly_taylor_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 6-th order Taylor polynomial about x=0."""
  return _exp16_poly_taylor(order=6, x=x)


def exp16_poly_taylor_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 7-th order Taylor polynomial about x=0."""
  return _exp16_poly_taylor(order=7, x=x)


def _exp16_poly_taylor(order: int, x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the Taylor polynomial about x=0 of the given order."""
  coeffs = _exp2_16_poly_taylor_coeffs()
  coeffs = [jnp.bfloat16(c) for c in coeffs]
  if order >= len(coeffs):
    raise NotImplementedError("Missing coefficients.")
  coeffs = coeffs[: order + 1]
  return _horner_scheme(x[0], coeffs)


def _exp2_16_poly_taylor_coeffs() -> List[jnp.bfloat16]:
  return [
      jnp.bfloat16(1.0000000000000000000000e0),
      jnp.bfloat16(6.9314718055994530941723e-1),
      jnp.bfloat16(2.4022650695910071233355e-1),
      jnp.bfloat16(5.5504108664821579953142e-2),
      jnp.bfloat16(9.6181291076284771619791e-3),
      jnp.bfloat16(1.3333558146428443423412e-3),
      jnp.bfloat16(1.5403530393381609954437e-4),
      jnp.bfloat16(1.5252733804059840280025e-5),
      jnp.bfloat16(1.3215486790144309488404e-6),
      jnp.bfloat16(1.0178086009239699727490e-7),
      jnp.bfloat16(7.0549116208011233298754e-9),
      jnp.bfloat16(4.4455382718708114975964e-10),
      jnp.bfloat16(2.5678435993488205141995e-11),
      jnp.bfloat16(1.3691488853904128880892e-12),
      jnp.bfloat16(6.7787263548225456334491e-14),
      jnp.bfloat16(3.1324367070884286216349e-15),
      jnp.bfloat16(1.3570247948755147193113e-16),
      jnp.bfloat16(5.5330465324582420434855e-18),
      jnp.bfloat16(2.1306753354891179960204e-19),
      jnp.bfloat16(7.7730084288573564190890e-21),
      jnp.bfloat16(2.6939194384655834169729e-22),
      jnp.bfloat16(8.8918222068002391716486e-24),
      jnp.bfloat16(2.8015188603108621554653e-25),
      jnp.bfloat16(8.4428908665651537891497e-27),
      jnp.bfloat16(2.4384024999728957391126e-28),
      jnp.bfloat16(6.7606872717061392013239e-30),
      jnp.bfloat16(1.8023658927040843470438e-31),
      jnp.bfloat16(4.6270549513527591374399e-33),
      jnp.bfloat16(1.1454393192236071063434e-34),
      jnp.bfloat16(2.7377863262839532041765e-36),
      jnp.bfloat16(6.3256295767976421810542e-38),
      jnp.bfloat16(1.4143846149754470061566e-39),
      jnp.bfloat16(3.0636772128049840388260e-41),
  ]


def exp16_rational_minimax_2_2(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (2, 2)th order rational approximation with minimax coeffs.

  Computed in 2023-07-03.rational_function_minimax_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0000004340345412482396911664949127338014686592115),
          jnp.bfloat16(0.36648083737656894587904405715753434885667190664788),
          jnp.bfloat16(0.047732290961851645674926681934254819604785083240237),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0),
          jnp.bfloat16(-0.32664473852531631611764106373825365218713724774378),
          jnp.bfloat16(0.033751826620697830871997163260564962450377124492229),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp16_rational_minimax_3_3(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (3, 3)th order rational approximation with minimax coeffs.

  Computed in 2023-07-03.rational_function_minimax_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(0.99999999990690473717505909830363350027664634388131),
          jnp.bfloat16(0.35857295387049270758870663293763299992633595381372),
          jnp.bfloat16(0.052335357780515512510714408445474101067283481237042),
          jnp.bfloat16(0.0033052508151820915209509737017296612609358819222018),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0),
          jnp.bfloat16(-0.33457423579158266928350341277985789477295952904675),
          jnp.bfloat16(0.044018182177239523068827078151998606550202697590210),
          jnp.bfloat16(-0.0023371652649376210213394248938490759253629263417203),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp16_rational_minimax_4_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (4, 4)th order rational approximation with minimax coeffs.

  Computed in 2023-07-03.rational_function_minimax_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0000000000000110932314930916403358964074451096810),
          jnp.bfloat16(0.35514943227000784164247291260110350981327842758030),
          jnp.bfloat16(0.054504397443498413634232494565692466546621667343988),
          jnp.bfloat16(0.0043961581146562074591196827074891917120033491565777),
          jnp.bfloat16(0.00016357454492149283435106367082339055580415290918081),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0),
          jnp.bfloat16(-0.33799774828814290230468112423520427769911103170492),
          jnp.bfloat16(0.048560076698570416362649009417815080291534980798979),
          jnp.bfloat16(-0.0035712118938156366801419356548733184499178255759202),
          jnp.bfloat16(0.00011566466994349112222271042040971889865223845557821),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp16_rational_minimax_5_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (5, 5)th order rational approximation with minimax coeffs.

  Computed in 2023-07-03.rational_function_minimax_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(0.99999999999999999915881045863163938868878769156443),
          jnp.bfloat16(0.35324495940459823413109142991204531932810581877484),
          jnp.bfloat16(0.055725984269045596406224024203560093365801610502792),
          jnp.bfloat16(0.0049793691590860342706272498424577252623262349121666),
          jnp.bfloat16(0.00025695208314218783764187336804480239598674605238496),
          jnp.bfloat16(6.2974572229969972894339519146031177118648559017555e-6),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0),
          jnp.bfloat16(-0.33990222115534727866136044818791876171823435818021),
          jnp.bfloat16(0.051101743569844926084658605665749829020028189809642),
          jnp.bfloat16(-0.0042922456872064277233627541001421827734975553582289),
          jnp.bfloat16(0.00020395743396291748707697098117168179490316632132612),
          jnp.bfloat16(-4.4529747066133809809683687992333491018786710064949e-6),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp16_rational_minimax_6_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (6, 6)th order rational approximation with minimax coeffs.

  Computed in 2023-07-03.rational_function_minimax_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0000000000000000000000441599721678342879445406787),
          jnp.bfloat16(0.35203245589268587800834721529829750482601260661364),
          jnp.bfloat16(0.056507868835118329308353468509810112347424392310025),
          jnp.bfloat16(0.0053439007055788682531793669868282447124238885655518),
          jnp.bfloat16(0.00031770881056990130630117428505603535708559831774140),
          jnp.bfloat16(
              0.000011429747188738066107408801566885768885594797377839
          ),
          jnp.bfloat16(1.9838195333385935474055030850130918642773613567649e-7),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0),
          jnp.bfloat16(-0.34111472466725943139396894833701788077845821406477),
          jnp.bfloat16(0.052724071526610519393286798673206867198647950024721),
          jnp.bfloat16(-0.0047609507064225453880903568085425761994317947890137),
          jnp.bfloat16(0.00026716847148572337991204927871765626355580452847803),
          jnp.bfloat16(-8.9237150912089954334673589763357096151576653352675e-6),
          jnp.bfloat16(1.4027722446740516891757040671335835834876884135408e-7),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp16_rational_pade_1_1(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (1, 1)th order rational approximation with Pade coeffs.

  Computed in 2023-07-03.pade_approximants_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0000000000000000000000000000000000000000000000000),
          jnp.bfloat16(0.34657359027997265470861606072908828403775006718013),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0000000000000000000000000000000000000000000000000),
          jnp.bfloat16(-0.34657359027997265470861606072908828403775006718013),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp16_rational_pade_2_2(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (2, 2)th order rational approximation with Pade coeffs.

  Computed in 2023-07-03.pade_approximants_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0000000000000000000000000000000000000000000000000),
          jnp.bfloat16(0.34657359027997265470861606072908828403775006718013),
          jnp.bfloat16(0.040037751159850118722258543860555414310879412632879),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0000000000000000000000000000000000000000000000000),
          jnp.bfloat16(-0.34657359027997265470861606072908828403775006718013),
          jnp.bfloat16(0.040037751159850118722258543860555414310879412632879),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp16_rational_pade_3_3(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (3, 3)th order rational approximation with Pade coeffs.

  Computed in 2023-07-03.pade_approximants_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0000000000000000000000000000000000000000000000000),
          jnp.bfloat16(0.34657359027997265470861606072908828403775006718013),
          jnp.bfloat16(0.048045301391820142466710252632666497173055295159455),
          jnp.bfloat16(0.0027752054332410789976571131884310878679677207112384),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0000000000000000000000000000000000000000000000000),
          jnp.bfloat16(-0.34657359027997265470861606072908828403775006718013),
          jnp.bfloat16(0.048045301391820142466710252632666497173055295159455),
          jnp.bfloat16(-0.0027752054332410789976571131884310878679677207112384),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp16_rational_pade_4_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (4, 4)th order rational approximation with Pade coeffs.

  Computed in 2023-07-03.pade_approximants_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0000000000000000000000000000000000000000000000000),
          jnp.bfloat16(0.34657359027997265470861606072908828403775006718013),
          jnp.bfloat16(0.051477108634093009785760984963571246971130673385130),
          jnp.bfloat16(0.0039645791903443985680815902691872683828110295874834),
          jnp.bfloat16(0.00013740184439469253088541530819512664971328350747532),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.bfloat16(1.0000000000000000000000000000000000000000000000000),
          jnp.bfloat16(-0.34657359027997265470861606072908828403775006718013),
          jnp.bfloat16(0.051477108634093009785760984963571246971130673385130),
          jnp.bfloat16(-0.0039645791903443985680815902691872683828110295874834),
          jnp.bfloat16(0.00013740184439469253088541530819512664971328350747532),
      ],
  )
  return jnp.divide(numerator, denominator)


################################################################################
# 32-bit baselines for f=e^x on [-log(2)/2, log(2)/2]. #########################
################################################################################


def expe32_jnp(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns JAX NumPy approximation."""
  return jnp.exp(x[0])


def correct_on_the_fly_expe_32bits_baselines() -> List[Tuple[str, FinalizedFn]]:
  """Baselines for 32-bit e^x on [-log(2)/2, log(2)/2].

  Returns:
    A list of (name, finalized_fn).
  """
  return [  # pytype: disable=bad-return-type  # jax-ndarray
      ("JAX Numpy", expe32_jnp),
  ]


################################################################################
# 32-bit baselines for f=loge(x) on [0.5, 1]. ##################################
################################################################################


def loge32_jnp(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns JAX NumPy approximation."""
  return jnp.exp(x[0])


def correct_on_the_fly_loge_32bits_baselines() -> List[Tuple[str, FinalizedFn]]:
  """Baselines for 32-bit loge(x) on [0.5, 1].

  Returns:
    A list of (name, finalized_fn).
  """
  return [  # pytype: disable=bad-return-type  # jax-ndarray
      ("JAX Numpy", loge32_jnp),
  ]


################################################################################
# 32-bit baselines for f=2^x on [0, 1]. ########################################
################################################################################


def table_exp2_32bits_baselines() -> List[Tuple[str, FinalizedFn]]:
  """Baselines for 32-bit 2^x on [0, 1].

  Returns:
    A list of (name, finalized_fn).
  """
  return [  # pytype: disable=bad-return-type  # jax-ndarray
      ("JAX Numpy", exp32_jnp),
      ("Polynomial, Sollya, 5th order", exp32_poly_sollya_5),
      ("Polynomial, Sollya, 6th order", exp32_poly_sollya_6),
      ("Polynomial, Sollya, 7th order", exp32_poly_sollya_7),
      ("Polynomial, Sollya, 8th order", exp32_poly_sollya_8),
      ("Polynomial, Sollya, 9th order", exp32_poly_sollya_9),
      ("Polynomial, Sollya, 10th order", exp32_poly_sollya_10),
      ("Polynomial, Sollya, 11th order", exp32_poly_sollya_11),
      ("Polynomial, Sollya, 12th order", exp32_poly_sollya_12),
      ("Polynomial, Sollya, 13th order", exp32_poly_sollya_13),
      ("Polynomial, Sollya, 14th order", exp32_poly_sollya_14),
      ("Polynomial, Sollya, 15th order", exp32_poly_sollya_15),
      ("Polynomial, Sollya, 16th order", exp32_poly_sollya_16),
      ("Polynomial, Sollya, 17th order", exp32_poly_sollya_17),
      ("Polynomial, Sollya, 18th order", exp32_poly_sollya_18),
      ("Polynomial, Sollya, 19th order", exp32_poly_sollya_19),
      ("Polynomial, Sollya, 20th order", exp32_poly_sollya_20),
      ("Polynomial, Sollya, 21th order", exp32_poly_sollya_21),
      ("Polynomial, Sollya, 22th order", exp32_poly_sollya_22),
      ("Polynomial, Sollya, 23th order", exp32_poly_sollya_23),
      ("Taylor expansion, 6th order", exp32_poly_taylor_6),
      ("Taylor expansion, 7th order", exp32_poly_taylor_7),
      ("Taylor expansion, 8th order", exp32_poly_taylor_8),
      ("Taylor expansion, 9th order", exp32_poly_taylor_9),
      ("Taylor expansion, 10th order", exp32_poly_taylor_10),
      ("Taylor expansion, 11th order", exp32_poly_taylor_11),
      ("Taylor expansion, 12th order", exp32_poly_taylor_12),
      ("Rational, Minimax, (2,2)-order", exp32_rational_minimax_2_2),
      ("Rational, Minimax, (3,3)-order", exp32_rational_minimax_3_3),
      ("Rational, Minimax, (4,4)-order", exp32_rational_minimax_4_4),
      ("Rational, Minimax, (5,5)-order", exp32_rational_minimax_5_5),
      ("Rational, Minimax, (6,6)-order", exp32_rational_minimax_6_6),
      ("Pade approximant, (2,2)-order", exp32_rational_pade_2_2),
      ("Pade approximant, (3,3)-order", exp32_rational_pade_3_3),
      ("Pade approximant, (4,4)-order", exp32_rational_pade_4_4),
      ("Pade approximant, (6,6)-order", exp32_rational_pade_6_6),
      ("Pade approximant, (8,8)-order", exp32_rational_pade_8_8),
      ("Pade approximant, (10,10)-order", exp32_rational_pade_10_10),
  ]


def exp32_jnp(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns JAX NumPy approximation."""
  return jnp.exp2(x[0])


# Sollya


def exp32_poly_sollya_5_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 5-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1.fffffep-1")),  # 0.999999940395355224609375
      jnp.float32(float.fromhex("0x1.62e4eap-1")),  # 0.693152725696563720703125
      jnp.float32(
          float.fromhex("0x1.ebd6c4p-3")
      ),  # 0.2401557266712188720703125
      jnp.float32(
          float.fromhex("0x1.c9498ep-5")
      ),  # 5.58212064206600189208984375e-2
      jnp.float32(
          float.fromhex("0x1.26bce2p-7")
      ),  # 8.994684554636478424072265625e-3
      jnp.float32(
          float.fromhex("0x1.ebaafp-10")
      ),  # 1.875563524663448333740234375e-3
  ]


def exp32_poly_sollya_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 5-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_5_coefficients())


def exp32_poly_sollya_6_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 6-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e42cp-1")),  # 0.69314706325531005859375
      jnp.float32(float.fromhex("0x1.ebfd38p-3")),  # 0.240229070186614990234375
      jnp.float32(
          float.fromhex("0x1.c68b2ep-5")
      ),  # 5.54862879216670989990234375e-2
      jnp.float32(
          float.fromhex("0x1.3cfb6p-7")
      ),  # 9.67352092266082763671875e-3
      jnp.float32(
          float.fromhex("0x1.4748aep-10")
      ),  # 1.248489017598330974578857421875e-3
      jnp.float32(
          float.fromhex("0x1.c41242p-13")
      ),  # 2.15564403333701193332672119140625e-4
  ]


def exp32_poly_sollya_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 6-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_6_coefficients())


def exp32_poly_sollya_7_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 7-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbd6p-3")
      ),  # 0.24022643268108367919921875
      jnp.float32(
          float.fromhex("0x1.c6b228p-5")
      ),  # 5.550487339496612548828125e-2
      jnp.float32(
          float.fromhex("0x1.3b0dc4p-7")
      ),  # 9.61467809975147247314453125e-3
      jnp.float32(
          float.fromhex("0x1.5fa0eep-10")
      ),  # 1.341356779448688030242919921875e-3
      jnp.float32(
          float.fromhex("0x1.2e4dc6p-13")
      ),  # 1.44149686093442142009735107421875e-4
      jnp.float32(
          float.fromhex("0x1.65cde8p-16")
      ),  # 2.13267994695343077182769775390625e-5
  ]


def exp32_poly_sollya_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 7-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_7_coefficients())


def exp32_poly_sollya_8_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 8-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbdap-3")
      ),  # 0.24022646248340606689453125
      jnp.float32(
          float.fromhex("0x1.c6b14cp-5")
      ),  # 5.5504463613033294677734375e-2
      jnp.float32(
          float.fromhex("0x1.3b1f72p-7")
      ),  # 9.616785682737827301025390625e-3
      jnp.float32(
          float.fromhex("0x1.5e3bf2p-10")
      ),  # 1.336037297733128070831298828125e-3
      jnp.float32(
          float.fromhex("0x1.3d2114p-13")
      ),  # 1.5121899195946753025054931640625e-4
      jnp.float32(
          float.fromhex("0x1.16236cp-16")
      ),  # 1.657833854551427066326141357421875e-5
      jnp.float32(
          float.fromhex("0x1.5535f8p-20")
      ),  # 1.27110934045049361884593963623046875e-6
  ]


def exp32_poly_sollya_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 8-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_8_coefficients())


def exp32_poly_sollya_9_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 9-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbdap-3")
      ),  # 0.24022646248340606689453125
      jnp.float32(
          float.fromhex("0x1.c6b162p-5")
      ),  # 5.55045045912265777587890625e-2
      jnp.float32(float.fromhex("0x1.3b1b8p-7")),  # 9.616315364837646484375e-3
      jnp.float32(
          float.fromhex("0x1.5ec8dcp-10")
      ),  # 1.33813708089292049407958984375e-3
      jnp.float32(
          float.fromhex("0x1.333588p-13")
      ),  # 1.464887172915041446685791015625e-4
      jnp.float32(
          float.fromhex("0x1.7635ep-16")
      ),  # 2.230468089692294597625732421875e-5
      jnp.float32(
          float.fromhex("-0x1.32f41p-19")
      ),  # -2.286980816279537975788116455078125e-6
      jnp.float32(
          float.fromhex("0x1.deacb6p-21")
      ),  # 8.9160101879315334372222423553466796875e-7
  ]


def exp32_poly_sollya_9(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 9-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_9_coefficients())


def exp32_poly_sollya_10_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 10-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(float.fromhex("0x1.ebfbd8p-3")),  # 0.240226447582244873046875
      jnp.float32(
          float.fromhex("0x1.c6b204p-5")
      ),  # 5.5504806339740753173828125e-2
      jnp.float32(
          float.fromhex("0x1.3b074ep-7")
      ),  # 9.613907895982265472412109375e-3
      jnp.float32(
          float.fromhex("0x1.616edep-10")
      ),  # 1.348240184597671031951904296875e-3
      jnp.float32(
          float.fromhex("0x1.fe83e6p-14")
      ),  # 1.217163153341971337795257568359375e-4
      jnp.float32(
          float.fromhex("0x1.ef755cp-15")
      ),  # 5.90632480452768504619598388671875e-5
      jnp.float32(
          float.fromhex("-0x1.23f368p-15")
      ),  # -3.4803248126991093158721923828125e-5
      jnp.float32(
          float.fromhex("0x1.17c60ep-16")
      ),  # 1.6675809092703275382518768310546875e-5
      jnp.float32(
          float.fromhex("-0x1.b2689cp-19")
      ),  # -3.23659651257912628352642059326171875e-6
  ]


def exp32_poly_sollya_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 10-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_10_coefficients())


def exp32_poly_sollya_11_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 11-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbd6p-3")
      ),  # 0.24022643268108367919921875
      jnp.float32(
          float.fromhex("0x1.c6b2d4p-5")
      ),  # 5.5505193769931793212890625e-2
      jnp.float32(
          float.fromhex("0x1.3ae604p-7")
      ),  # 9.60993953049182891845703125e-3
      jnp.float32(
          float.fromhex("0x1.67157ap-10")
      ),  # 1.369796344079077243804931640625e-3
      jnp.float32(
          float.fromhex("0x1.b4993p-15")
      ),  # 5.204658373259007930755615234375e-5
      jnp.float32(
          float.fromhex("0x1.a31be2p-13")
      ),  # 1.99846705072559416294097900390625e-4
      jnp.float32(
          float.fromhex("-0x1.c1c284p-13")
      ),  # -2.1446219761855900287628173828125e-4
      jnp.float32(
          float.fromhex("0x1.4a11f4p-13")
      ),  # 1.5738970250822603702545166015625e-4
      jnp.float32(
          float.fromhex("-0x1.109f9p-14")
      ),  # -6.49984576739370822906494140625e-5
      jnp.float32(
          float.fromhex("0x1.8655aap-17")
      ),  # 1.16328783406061120331287384033203125e-5
  ]


def exp32_poly_sollya_11(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 11-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_11_coefficients())


def exp32_poly_sollya_12_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 12-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbd4p-3")
      ),  # 0.2402264177799224853515625
      jnp.float32(
          float.fromhex("0x1.c6b3d4p-5")
      ),  # 5.5505670607089996337890625e-2
      jnp.float32(
          float.fromhex("0x1.3ab38cp-7")
      ),  # 9.60392318665981292724609375e-3
      jnp.float32(
          float.fromhex("0x1.71b562p-10")
      ),  # 1.410326105542480945587158203125e-3
      jnp.float32(
          float.fromhex("-0x1.d84d7p-14")
      ),  # -1.126056886278092861175537109375e-4
      jnp.float32(
          float.fromhex("0x1.48fbd8p-11")
      ),  # 6.2748673371970653533935546875e-4
      jnp.float32(
          float.fromhex("-0x1.eda122p-11")
      ),  # -9.415234089829027652740478515625e-4
      jnp.float32(
          float.fromhex("0x1.f905e8p-11")
      ),  # 9.6325506456196308135986328125e-4
      jnp.float32(
          float.fromhex("-0x1.483f3cp-11")
      ),  # -6.26081484369933605194091796875e-4
      jnp.float32(
          float.fromhex("0x1.eba9bcp-13")
      ),  # 2.3444319958798587322235107421875e-4
      jnp.float32(
          float.fromhex("-0x1.42ea72p-15")
      ),  # -3.849456334137357771396636962890625e-5
  ]


def exp32_poly_sollya_12(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 12-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_12_coefficients())


def exp32_poly_sollya_13_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 13-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbd2p-3")
      ),  # 0.24022640287876129150390625
      jnp.float32(
          float.fromhex("0x1.c6b506p-5")
      ),  # 5.55062405765056610107421875e-2
      jnp.float32(
          float.fromhex("0x1.3a6b68p-7")
      ),  # 9.5953233540058135986328125e-3
      jnp.float32(
          float.fromhex("0x1.83f8aap-10")
      ),  # 1.479993225075304508209228515625e-3
      jnp.float32(
          float.fromhex("-0x1.deb5c6p-12")
      ),  # -4.5653348206542432308197021484375e-4
      jnp.float32(
          float.fromhex("0x1.c5bc02p-10")
      ),  # 1.730859396047890186309814453125e-3
      jnp.float32(
          float.fromhex("-0x1.b2aedap-9")
      ),  # -3.31636820919811725616455078125e-3
      jnp.float32(
          float.fromhex("0x1.21e9c2p-8")
      ),  # 4.4237230904400348663330078125e-3
      jnp.float32(
          float.fromhex("-0x1.05da34p-8")
      ),  # -3.995549865067005157470703125e-3
      jnp.float32(
          float.fromhex("0x1.31fe64p-9")
      ),  # 2.3345467634499073028564453125e-3
      jnp.float32(
          float.fromhex("-0x1.a16b1p-11")
      ),  # -7.961620576679706573486328125e-4
      jnp.float32(
          float.fromhex("0x1.f8c018p-14")
      ),  # 1.2034186511300504207611083984375e-4
  ]


def exp32_poly_sollya_13(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 13-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_13_coefficients())


def exp32_poly_sollya_14_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 14-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(float.fromhex("0x1.ebfbdp-3")),  # 0.24022638797760009765625
      jnp.float32(
          float.fromhex("0x1.c6b67p-5")
      ),  # 5.55069148540496826171875e-2
      jnp.float32(
          float.fromhex("0x1.3a06b6p-7")
      ),  # 9.583319537341594696044921875e-3
      jnp.float32(
          float.fromhex("0x1.a222f4p-10")
      ),  # 1.59506429918110370635986328125e-3
      jnp.float32(
          float.fromhex("-0x1.292cb6p-10")
      ),  # -1.133631332777440547943115234375e-3
      jnp.float32(
          float.fromhex("0x1.1d1b38p-8")
      ),  # 4.35037724673748016357421875e-3
      jnp.float32(
          float.fromhex("-0x1.4f57a8p-7")
      ),  # -1.02338381111621856689453125e-2
      jnp.float32(
          float.fromhex("0x1.184ad4p-6")
      ),  # 1.71076841652393341064453125e-2
      jnp.float32(
          float.fromhex("-0x1.4a0b6p-6")
      ),  # -2.01443135738372802734375e-2
      jnp.float32(
          float.fromhex("0x1.0bbe14p-6")
      ),  # 1.63417048752307891845703125e-2
      jnp.float32(
          float.fromhex("-0x1.1cd476p-7")
      ),  # -8.692319504916667938232421875e-3
      jnp.float32(
          float.fromhex("0x1.65a3a4p-9")
      ),  # 2.7285707183182239532470703125e-3
      jnp.float32(
          float.fromhex("-0x1.91b698p-12")
      ),  # -3.83103615604341030120849609375e-4
  ]


def exp32_poly_sollya_14(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 14-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_14_coefficients())


def exp32_poly_sollya_15_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 15-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbcep-3")
      ),  # 0.24022637307643890380859375
      jnp.float32(
          float.fromhex("0x1.c6b812p-5")
      ),  # 5.55076934397220611572265625e-2
      jnp.float32(
          float.fromhex("0x1.398028p-7")
      ),  # 9.5672793686389923095703125e-3
      jnp.float32(
          float.fromhex("0x1.d0ff8ep-10")
      ),  # 1.773827592842280864715576171875e-3
      jnp.float32(
          float.fromhex("-0x1.35ff52p-9")
      ),  # -2.36509204842150211334228515625e-3
      jnp.float32(
          float.fromhex("0x1.471d9cp-7")
      ),  # 9.98277775943279266357421875e-3
      jnp.float32(
          float.fromhex("-0x1.cba8d4p-6")
      ),  # -2.80553884804248809814453125e-2
      jnp.float32(
          float.fromhex("0x1.d2e47ap-5")
      ),  # 5.69937117397785186767578125e-2
      jnp.float32(
          float.fromhex("-0x1.57263ep-4")
      ),  # -8.3776704967021942138671875e-2
      jnp.float32(
          float.fromhex("0x1.698222p-4")
      ),  # 8.8258869946002960205078125e-2
      jnp.float32(
          float.fromhex("-0x1.0a07d6p-4")
      ),  # -6.4948879182338714599609375e-2
      jnp.float32(
          float.fromhex("0x1.03a3fep-5")
      ),  # 3.16944085061550140380859375e-2
      jnp.float32(
          float.fromhex("-0x1.2de892p-7")
      ),  # -9.213515557348728179931640625e-3
      jnp.float32(
          float.fromhex("0x1.3c8706p-10")
      ),  # 1.207456341944634914398193359375e-3
  ]


def exp32_poly_sollya_15(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 15-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_15_coefficients())


def exp32_poly_sollya_16_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 16-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbccp-3")
      ),  # 0.2402263581752777099609375
      jnp.float32(
          float.fromhex("0x1.c6b9eap-5")
      ),  # 5.55085726082324981689453125e-2
      jnp.float32(float.fromhex("0x1.38d38p-7")),  # 9.546697139739990234375e-3
      jnp.float32(
          float.fromhex("0x1.0adfccp-9")
      ),  # 2.0360886119306087493896484375e-3
      jnp.float32(
          float.fromhex("-0x1.234fe4p-8")
      ),  # -4.445069469511508941650390625e-3
      jnp.float32(
          float.fromhex("0x1.58893cp-6")
      ),  # 2.10288129746913909912109375e-2
      jnp.float32(
          float.fromhex("-0x1.1af342p-4")
      ),  # -6.9079644978046417236328125e-2
      jnp.float32(
          float.fromhex("0x1.5497b6p-3")
      ),  # 0.16630499064922332763671875
      jnp.float32(float.fromhex("-0x1.2e7fap-2")),  # -0.295408725738525390625
      jnp.float32(
          float.fromhex("0x1.8baba6p-2")
      ),  # 0.3863969743251800537109375
      jnp.float32(
          float.fromhex("-0x1.782c1ap-2")
      ),  # -0.3673557341098785400390625
      jnp.float32(
          float.fromhex("0x1.f96b06p-3")
      ),  # 0.24678616225719451904296875
      jnp.float32(
          float.fromhex("-0x1.c6a562p-4")
      ),  # -0.110997565090656280517578125
      jnp.float32(
          float.fromhex("0x1.eb3476p-6")
      ),  # 2.998076938092708587646484375e-2
      jnp.float32(
          float.fromhex("-0x1.e1cdb4p-9")
      ),  # -3.6758692003786563873291015625e-3
  ]


def exp32_poly_sollya_16(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 16-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_16_coefficients())


def exp32_poly_sollya_17_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 17-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbcap-3")
      ),  # 0.24022634327411651611328125
      jnp.float32(
          float.fromhex("0x1.c6bbfap-5")
      ),  # 5.55095560848712921142578125e-2
      jnp.float32(
          float.fromhex("0x1.37fa84p-7")
      ),  # 9.52083058655261993408203125e-3
      jnp.float32(
          float.fromhex("0x1.3ba2fp-9")
      ),  # 2.40811519324779510498046875e-3
      jnp.float32(
          float.fromhex("-0x1.fecc88p-8")
      ),  # -7.79417343437671661376953125e-3
      jnp.float32(float.fromhex("0x1.52ccep-5")),  # 4.1357457637786865234375e-2
      jnp.float32(
          float.fromhex("-0x1.3fbe72p-3")
      ),  # -0.15612496435642242431640625
      jnp.float32(float.fromhex("0x1.bf2d04p-2")),  # 0.436695158481597900390625
      jnp.float32(float.fromhex("-0x1.d4447p-1")),  # -0.914584636688232421875
      jnp.float32(float.fromhex("0x1.703592p0")),  # 1.43831741809844970703125
      jnp.float32(float.fromhex("-0x1.b07794p0")),  # -1.6893246173858642578125
      jnp.float32(float.fromhex("0x1.757f2ap0")),  # 1.45897161960601806640625
      jnp.float32(
          float.fromhex("-0x1.cc474ap-1")
      ),  # -0.898981392383575439453125
      jnp.float32(float.fromhex("0x1.7edd18p-2")),  # 0.37389028072357177734375
      jnp.float32(float.fromhex("-0x1.81294p-4")),  # -9.40334796905517578125e-2
      jnp.float32(
          float.fromhex("0x1.61df1cp-7")
      ),  # 1.079930178821086883544921875e-2
  ]


def exp32_poly_sollya_17(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 17-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_17_coefficients())


def exp32_poly_sollya_18_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 18-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbc6p-3")
      ),  # 0.24022631347179412841796875
      jnp.float32(
          float.fromhex("0x1.c6c048p-5")
      ),  # 5.551160871982574462890625e-2
      jnp.float32(
          float.fromhex("0x1.36163ep-7")
      ),  # 9.463100694119930267333984375e-3
      jnp.float32(
          float.fromhex("0x1.b1fd38p-9")
      ),  # 3.311074338853359222412109375e-3
      jnp.float32(
          float.fromhex("-0x1.12747ep-6")
      ),  # -1.675140671432018280029296875e-2
      jnp.float32(
          float.fromhex("0x1.a1a5dcp-4")
      ),  # 0.10196481645107269287109375
      jnp.float32(
          float.fromhex("-0x1.cb755ap-2")
      ),  # -0.4486898481845855712890625
      jnp.float32(float.fromhex("0x1.793ab8p0")),  # 1.473552227020263671875
      jnp.float32(float.fromhex("-0x1.d4902p1")),  # -3.660648345947265625
      jnp.float32(float.fromhex("0x1.bb1cecp2")),  # 6.92364025115966796875
      jnp.float32(float.fromhex("-0x1.3eee8ep3")),  # -9.96662044525146484375
      jnp.float32(float.fromhex("0x1.5ab2b2p3")),  # 10.83431339263916015625
      jnp.float32(float.fromhex("-0x1.17b3b4p3")),  # -8.7406864166259765625
      jnp.float32(float.fromhex("0x1.448316p2")),  # 5.070500850677490234375
      jnp.float32(float.fromhex("-0x1.ffa858p0")),  # -1.998662471771240234375
      jnp.float32(float.fromhex("0x1.eaa43cp-2")),  # 0.479142129421234130859375
      jnp.float32(
          float.fromhex("-0x1.afd54cp-5")
      ),  # -5.2714012563228607177734375e-2
  ]


def exp32_poly_sollya_18(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 18-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_18_coefficients())


def exp32_poly_sollya_19_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 19-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbc4p-3")
      ),  # 0.2402262985706329345703125
      jnp.float32(
          float.fromhex("0x1.c6c2fap-5")
      ),  # 5.55128939449787139892578125e-2
      jnp.float32(
          float.fromhex("0x1.34a868p-7")
      ),  # 9.4194896519184112548828125e-3
      jnp.float32(float.fromhex("0x1.0de68p-8")),  # 4.1183531284332275390625e-3
      jnp.float32(
          float.fromhex("-0x1.ac3288p-6")
      ),  # -2.6135094463825225830078125e-2
      jnp.float32(float.fromhex("0x1.68862p-3")),  # 0.1760370731353759765625
      jnp.float32(
          float.fromhex("-0x1.bb2d4ap-1")
      ),  # -0.865579903125762939453125
      jnp.float32(float.fromhex("0x1.99a7d8p1")),  # 3.20043468475341796875
      jnp.float32(float.fromhex("-0x1.212088p3")),  # -9.035221099853515625
      jnp.float32(float.fromhex("0x1.3a60fp4")),  # 19.6486663818359375
      jnp.float32(float.fromhex("-0x1.081b5ep5")),  # -33.013362884521484375
      jnp.float32(float.fromhex("0x1.55d6bp5")),  # 42.729827880859375
      jnp.float32(float.fromhex("-0x1.5181fp5")),  # -42.188446044921875
      jnp.float32(float.fromhex("0x1.f2d97p4")),  # 31.1780853271484375
      jnp.float32(float.fromhex("-0x1.0b060ep4")),  # -16.6889781951904296875
      jnp.float32(float.fromhex("0x1.86ee7cp2")),  # 6.10830593109130859375
      jnp.float32(float.fromhex("-0x1.5dfe2ap0")),  # -1.36715948581695556640625
      jnp.float32(
          float.fromhex("0x1.20f9acp-3")
      ),  # 0.1411012113094329833984375
  ]


def exp32_poly_sollya_19(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 19-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_19_coefficients())


def exp32_poly_sollya_20_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 20-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(float.fromhex("0x1.ebfbcp-3")),  # 0.240226268768310546875
      jnp.float32(
          float.fromhex("0x1.c6c876p-5")
      ),  # 5.55155090987682342529296875e-2
      jnp.float32(
          float.fromhex("0x1.319a32p-7")
      ),  # 9.326242841780185699462890625e-3
      jnp.float32(
          float.fromhex("0x1.87033cp-8")
      ),  # 5.966379307210445404052734375e-3
      jnp.float32(
          float.fromhex("-0x1.94d94ap-5")
      ),  # -4.94200177490711212158203125e-2
      jnp.float32(float.fromhex("0x1.823b3cp-2")),  # 0.377179086208343505859375
      jnp.float32(float.fromhex("-0x1.0eb63ep1")),  # -2.1149365901947021484375
      jnp.float32(float.fromhex("0x1.1ebac4p3")),  # 8.9602985382080078125
      jnp.float32(float.fromhex("-0x1.d2ba26p4")),  # -29.1704463958740234375
      jnp.float32(float.fromhex("0x1.270d06p6")),  # 73.76271820068359375
      jnp.float32(float.fromhex("-0x1.235cd4p7")),  # -145.681304931640625
      jnp.float32(float.fromhex("0x1.c1ae14p7")),  # 224.839996337890625
      jnp.float32(float.fromhex("-0x1.0dde7ap8")),  # -269.869049072265625
      jnp.float32(float.fromhex("0x1.f22a1p7")),  # 249.0821533203125
      jnp.float32(float.fromhex("-0x1.5a875cp7")),  # -173.264373779296875
      jnp.float32(float.fromhex("0x1.5f440ep6")),  # 87.81645965576171875
      jnp.float32(float.fromhex("-0x1.e9751ep4")),  # -30.5910930633544921875
      jnp.float32(float.fromhex("0x1.a2f448p2")),  # 6.5461597442626953125
      jnp.float32(float.fromhex("-0x1.4c0b28p-1")),  # -0.6485226154327392578125
  ]


def exp32_poly_sollya_20(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 20-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_20_coefficients())


def exp32_poly_sollya_21_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 21-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbbep-3")
      ),  # 0.24022625386714935302734375
      jnp.float32(
          float.fromhex("0x1.c6cbcap-5")
      ),  # 5.55170960724353790283203125e-2
      jnp.float32(
          float.fromhex("0x1.2f69a8p-7")
      ),  # 9.2594213783740997314453125e-3
      jnp.float32(float.fromhex("0x1.ec038p-8")),  # 7.5075328350067138671875e-3
      jnp.float32(
          float.fromhex("-0x1.263bcep-4")
      ),  # -7.1834377944469451904296875e-2
      jnp.float32(float.fromhex("0x1.33129p-1")),  # 0.599750995635986328125
      jnp.float32(float.fromhex("-0x1.d9d8a2p1")),  # -3.7019236087799072265625
      jnp.float32(float.fromhex("0x1.15d9ccp4")),  # 17.365673065185546875
      jnp.float32(float.fromhex("-0x1.f82626p5")),  # -63.018627166748046875
      jnp.float32(float.fromhex("0x1.6616d2p7")),  # 179.0445709228515625
      jnp.float32(float.fromhex("-0x1.912204p8")),  # -401.13287353515625
      jnp.float32(float.fromhex("0x1.635858p9")),  # 710.690185546875
      jnp.float32(float.fromhex("-0x1.f117ccp9")),  # -994.1859130859375
      jnp.float32(float.fromhex("0x1.10c464p10")),  # 1091.068603515625
      jnp.float32(float.fromhex("-0x1.cfd00ep9")),  # -927.62542724609375
      jnp.float32(float.fromhex("0x1.2b191ap9")),  # 598.19610595703125
      jnp.float32(float.fromhex("-0x1.1aa14p8")),  # -282.6298828125
      jnp.float32(float.fromhex("0x1.70e556p6")),  # 92.22396087646484375
      jnp.float32(float.fromhex("-0x1.2908ep4")),  # -18.564666748046875
      jnp.float32(float.fromhex("0x1.bc9474p0")),  # 1.7366402149200439453125
  ]


def exp32_poly_sollya_21(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 21-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_21_coefficients())


def exp32_poly_sollya_22_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 22-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbbap-3")
      ),  # 0.24022622406482696533203125
      jnp.float32(float.fromhex("0x1.c6d26p-5")),  # 5.5520236492156982421875e-2
      jnp.float32(
          float.fromhex("0x1.2af5dap-7")
      ),  # 9.123546071350574493408203125e-3
      jnp.float32(
          float.fromhex("0x1.61a456p-7")
      ),  # 1.0792295448482036590576171875e-2
      jnp.float32(
          float.fromhex("-0x1.f601cap-4")
      ),  # -0.122560299932956695556640625
      jnp.float32(float.fromhex("0x1.23c5cep0")),  # 1.13973701000213623046875
      jnp.float32(float.fromhex("-0x1.f7380ep2")),  # -7.862796306610107421875
      jnp.float32(float.fromhex("0x1.4ae3bp5")),  # 41.361175537109375
      jnp.float32(float.fromhex("-0x1.521392p7")),  # -169.0382232666015625
      jnp.float32(float.fromhex("0x1.0fec9ep9")),  # 543.84857177734375
      jnp.float32(float.fromhex("-0x1.5b5d34p10")),  # -1389.456298828125
      jnp.float32(float.fromhex("0x1.620b54p11")),  # 2832.35400390625
      jnp.float32(float.fromhex("-0x1.2032a8p12")),  # -4611.166015625
      jnp.float32(float.fromhex("0x1.759be8p12")),  # 5977.744140625
      jnp.float32(float.fromhex("-0x1.7ebed4p12")),  # -6123.9267578125
      jnp.float32(float.fromhex("0x1.31b4a4p12")),  # 4891.2900390625
      jnp.float32(float.fromhex("-0x1.747c4p11")),  # -2979.8828125
      jnp.float32(float.fromhex("0x1.4e2cap10")),  # 1336.697265625
      jnp.float32(float.fromhex("-0x1.9ff28ep8")),  # -415.947479248046875
      jnp.float32(float.fromhex("0x1.40a87cp6")),  # 80.1645355224609375
      jnp.float32(float.fromhex("-0x1.cd2434p2")),  # -7.20533466339111328125
  ]


def exp32_poly_sollya_22(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 22-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_22_coefficients())


def exp32_poly_sollya_23_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 23-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float32(float.fromhex("0x1p0")),  # 1
      jnp.float32(float.fromhex("0x1.62e43p-1")),  # 0.693147182464599609375
      jnp.float32(
          float.fromhex("0x1.ebfbb6p-3")
      ),  # 0.24022619426250457763671875
      jnp.float32(
          float.fromhex("0x1.c6d9e6p-5")
      ),  # 5.55238239467144012451171875e-2
      jnp.float32(
          float.fromhex("0x1.253b78p-7")
      ),  # 8.9487396180629730224609375e-3
      jnp.float32(
          float.fromhex("0x1.fbcedp-7")
      ),  # 1.5497066080570220947265625e-2
      jnp.float32(
          float.fromhex("-0x1.9f5bf4p-3")
      ),  # -0.2028121054172515869140625
      jnp.float32(float.fromhex("0x1.0a231ep1")),  # 2.0791966915130615234375
      jnp.float32(float.fromhex("-0x1.f9eed2p3")),  # -15.81040287017822265625
      jnp.float32(float.fromhex("0x1.6ed34ap6")),  # 91.70633697509765625
      jnp.float32(float.fromhex("-0x1.9deda2p8")),  # -413.928253173828125
      jnp.float32(float.fromhex("0x1.70c6p10")),  # 1475.09375
      jnp.float32(float.fromhex("-0x1.060b7p12")),  # -4192.71484375
      jnp.float32(float.fromhex("0x1.2af422p13")),  # 9566.5166015625
      jnp.float32(float.fromhex("-0x1.129f92p14")),  # -17575.892578125
      jnp.float32(float.fromhex("0x1.961eb2p14")),  # 25991.673828125
      jnp.float32(float.fromhex("-0x1.e17844p14")),  # -30814.06640625
      jnp.float32(float.fromhex("0x1.c5bf74p14")),  # 29039.86328125
      jnp.float32(float.fromhex("-0x1.4f232cp14")),  # -21448.79296875
      jnp.float32(float.fromhex("0x1.7b5ce4p13")),  # 12139.611328125
      jnp.float32(float.fromhex("-0x1.3d807ep12")),  # -5080.03076171875
      jnp.float32(float.fromhex("0x1.720d8cp10")),  # 1480.211669921875
      jnp.float32(float.fromhex("-0x1.0c0936p8")),  # -268.035980224609375
      jnp.float32(float.fromhex("0x1.6b4a6ep4")),  # 22.7056713104248046875
  ]


def exp32_poly_sollya_23(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 23-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2023-04-03.exp32_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp32_poly_sollya_23_coefficients())


# Taylor


def exp32_poly_taylor_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 6-th order Taylor polynomial about x=0."""
  return _exp32_poly_taylor(order=6, x=x)


def exp32_poly_taylor_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 7-th order Taylor polynomial about x=0."""
  return _exp32_poly_taylor(order=7, x=x)


def exp32_poly_taylor_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 8-th order Taylor polynomial about x=0."""
  return _exp32_poly_taylor(order=8, x=x)


def exp32_poly_taylor_9(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 9-th order Taylor polynomial about x=0."""
  return _exp32_poly_taylor(order=9, x=x)


def exp32_poly_taylor_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 10-th order Taylor polynomial about x=0."""
  return _exp32_poly_taylor(order=10, x=x)


def exp32_poly_taylor_11(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 11-th order Taylor polynomial about x=0."""
  return _exp32_poly_taylor(order=11, x=x)


def exp32_poly_taylor_12(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 12-th order Taylor polynomial about x=0."""
  return _exp32_poly_taylor(order=12, x=x)


def _exp32_poly_taylor(order: int, x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the Taylor polynomial about x=0 of the given order."""
  coeffs = _exp2_32_poly_taylor_coeffs()
  coeffs = [jnp.float32(c) for c in coeffs]
  if order >= len(coeffs):
    raise NotImplementedError("Missing coefficients.")
  coeffs = coeffs[: order + 1]
  return _horner_scheme(x[0], coeffs)


def _exp2_32_poly_taylor_coeffs() -> List[jnp.float32]:
  return [
      jnp.float32("1.0000000000000000000000e0"),
      jnp.float32("6.9314718055994530941723e-1"),
      jnp.float32("2.4022650695910071233355e-1"),
      jnp.float32("5.5504108664821579953142e-2"),
      jnp.float32("9.6181291076284771619791e-3"),
      jnp.float32("1.3333558146428443423412e-3"),
      jnp.float32("1.5403530393381609954437e-4"),
      jnp.float32("1.5252733804059840280025e-5"),
      jnp.float32("1.3215486790144309488404e-6"),
      jnp.float32("1.0178086009239699727490e-7"),
      jnp.float32("7.0549116208011233298754e-9"),
      jnp.float32("4.4455382718708114975964e-10"),
      jnp.float32("2.5678435993488205141995e-11"),
      jnp.float32("1.3691488853904128880892e-12"),
      jnp.float32("6.7787263548225456334491e-14"),
      jnp.float32("3.1324367070884286216349e-15"),
      jnp.float32("1.3570247948755147193113e-16"),
      jnp.float32("5.5330465324582420434855e-18"),
      jnp.float32("2.1306753354891179960204e-19"),
      jnp.float32("7.7730084288573564190890e-21"),
      jnp.float32("2.6939194384655834169729e-22"),
      jnp.float32("8.8918222068002391716486e-24"),
      jnp.float32("2.8015188603108621554653e-25"),
      jnp.float32("8.4428908665651537891497e-27"),
      jnp.float32("2.4384024999728957391126e-28"),
      jnp.float32("6.7606872717061392013239e-30"),
      jnp.float32("1.8023658927040843470438e-31"),
      jnp.float32("4.6270549513527591374399e-33"),
      jnp.float32("1.1454393192236071063434e-34"),
      jnp.float32("2.7377863262839532041765e-36"),
      jnp.float32("6.3256295767976421810542e-38"),
      jnp.float32("1.4143846149754470061566e-39"),
      jnp.float32("3.0636772128049840388260e-41"),
  ]


def exp32_rational_minimax_2_2(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (2, 2)th order rational approximation with minimax coeffs.

  Computed in 2023-07-03.rational_function_minimax_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000004340345412482397"),
          jnp.float32("3.6648083737656894587904e-1"),
          jnp.float32("4.7732290961851645674927e-2"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("-3.2664473852531631611764e-1"),
          jnp.float32("3.3751826620697830871997e-2"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp32_rational_minimax_3_3(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (3, 3)th order rational approximation with minimax coeffs.

  Computed in 2023-07-03.rational_function_minimax_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("9.99999999906904737175e-1"),
          jnp.float32("3.5857295387049270758870e-1"),
          jnp.float32("5.2335357780515512510714e-2"),
          jnp.float32("3.3052508151820915209510e-3"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("-3.3457423579158266928350e-1"),
          jnp.float32("4.4018182177239523068827e-2"),
          jnp.float32("-2.3371652649376210213394e-3"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp32_rational_minimax_4_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (4, 4)th order rational approximation with minimax coeffs.

  Computed in 2023-07-03.rational_function_minimax_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000110932315"),
          jnp.float32("3.5514943227000784164247e-1"),
          jnp.float32("5.4504397443498413634232e-2"),
          jnp.float32("4.3961581146562074591197e-3"),
          jnp.float32("1.6357454492149283435106e-4"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("-3.3799774828814290230468e-1"),
          jnp.float32("4.8560076698570416362649e-2"),
          jnp.float32("-3.5712118938156366801419e-3"),
          jnp.float32("1.1566466994349112222271e-4"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp32_rational_minimax_5_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (5, 5)th order rational approximation with minimax coeffs.

  Computed in 2023-07-03.rational_function_minimax_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("9.9999999999999999915881e-1"),
          jnp.float32("3.5324495940459823413109e-1"),
          jnp.float32("5.5725984269045596406224e-2"),
          jnp.float32("4.9793691590860342706272e-3"),
          jnp.float32("2.5695208314218783764187e-4"),
          jnp.float32("6.2974572229969972894340e-6"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("-3.3990222115534727866136e-1"),
          jnp.float32("5.1101743569844926084659e-2"),
          jnp.float32("-4.2922456872064277233628e-3"),
          jnp.float32("2.0395743396291748707697e-4"),
          jnp.float32("-4.4529747066133809809684e-6"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp32_rational_minimax_6_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (6, 6)th order rational approximation with minimax coeffs.

  Computed in 2023-07-03.rational_function_minimax_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000"),
          jnp.float32("3.5203245589268587800835e-1"),
          jnp.float32("5.6507868835118329308353e-2"),
          jnp.float32("5.3439007055788682531794e-3"),
          jnp.float32("3.1770881056990130630117e-4"),
          jnp.float32("1.1429747188738066107409e-5"),
          jnp.float32("1.9838195333385935474055e-7"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("-3.4111472466725943139397e-1"),
          jnp.float32("5.2724071526610519393287e-2"),
          jnp.float32("-4.7609507064225453880904e-3"),
          jnp.float32("2.6716847148572337991205e-4"),
          jnp.float32("-8.9237150912089954334674e-6"),
          jnp.float32("1.4027722446740516891757e-7"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp32_rational_pade_2_2(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (2, 2)th order Pade approximant about x=0.

  Computed in 2023-07-03.pade_approximants_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          jnp.float32("3.4657359027997265470862e-1"),
          jnp.float32("4.0037751159850118722259e-2"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          jnp.float32("-3.4657359027997265470862e-1"),
          jnp.float32("4.0037751159850118722259e-2"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp32_rational_pade_3_3(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (3, 3)th order Pade approximant about x=0.

  Computed in 2023-07-03.pade_approximants_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          jnp.float32("3.4657359027997265470862e-1"),
          jnp.float32("4.8045301391820142466710e-2"),
          jnp.float32("2.7752054332410789976571e-3"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          jnp.float32("-3.4657359027997265470862e-1"),
          jnp.float32("4.8045301391820142466710e-2"),
          jnp.float32("-2.7752054332410789976571e-3"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp32_rational_pade_4_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (4, 4)th order Pade approximant about x=0.

  Computed in 2023-07-03.pade_approximants_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          jnp.float32("3.4657359027997265470862e-1"),
          jnp.float32("5.1477108634093009785761e-2"),
          jnp.float32("3.9645791903443985680816e-3"),
          jnp.float32("1.3740184439469253088542e-4"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          jnp.float32("-3.4657359027997265470862e-1"),
          jnp.float32("5.1477108634093009785761e-2"),
          jnp.float32("-3.9645791903443985680816e-3"),
          jnp.float32("1.3740184439469253088542e-4"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp32_rational_pade_6_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (6, 6)th order Pade approximant about x=0.

  Computed in 2023-07-03.pade_approximants_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000"),
          jnp.float32("3.4657359027997265470862e-1"),
          jnp.float32("5.4596933399795616439443e-2"),
          jnp.float32("5.0458280604383254502857e-3"),
          jnp.float32("2.9145845780692355036300e-4"),
          jnp.float32("1.0101180413960941987434e-5"),
          jnp.float32("1.6670487438724686097876e-7"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000"),
          jnp.float32("-3.4657359027997265470862e-1"),
          jnp.float32("5.4596933399795616439443e-2"),
          jnp.float32("-5.0458280604383254502857e-3"),
          jnp.float32("2.9145845780692355036300e-4"),
          jnp.float32("-1.0101180413960941987434e-5"),
          jnp.float32("1.6670487438724686097876e-7"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp32_rational_pade_8_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (8, 8)th order Pade approximant about x=0.

  Computed in 2023-07-03.pade_approximants_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          jnp.float32("3.4657359027997265470862e-1"),
          jnp.float32("5.6052851623790166211162e-2"),
          jnp.float32("5.5504108664821579953142e-3"),
          jnp.float32("3.6992804260109527546073e-4"),
          jnp.float32("1.7094305315933901824887e-5"),
          jnp.float32("5.3858497878956678162367e-7"),
          jnp.float32("1.0666247415426461734284e-8"),
          jnp.float32("1.0268443504385632858123e-10"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          jnp.float32("-3.4657359027997265470862e-1"),
          jnp.float32("5.6052851623790166211162e-2"),
          jnp.float32("-5.5504108664821579953142e-3"),
          jnp.float32("3.6992804260109527546073e-4"),
          jnp.float32("-1.7094305315933901824887e-5"),
          jnp.float32("5.3858497878956678162367e-7"),
          jnp.float32("-1.0666247415426461734284e-8"),
          jnp.float32("1.0268443504385632858123e-10"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp32_rational_pade_10_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (10, 10)th order Pade approximant about x=0.

  Computed in 2023-07-03.pade_approximants_exp2.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          jnp.float32("3.4657359027997265470862e-1"),
          jnp.float32("5.6895751648208063447420e-2"),
          jnp.float32("5.8425377541917452582255e-3"),
          jnp.float32("4.1688485296222501630869e-4"),
          jnp.float32("2.1672192033668522592233e-5"),
          jnp.float32("8.3455660026061354242306e-7"),
          jnp.float32("2.3611043040340310030999e-8"),
          jnp.float32("4.7209407442763668093845e-10"),
          jnp.float32("6.0598273453439507784532e-12"),
          jnp.float32("3.8185020355501977364066e-14"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          jnp.float32("-3.4657359027997265470862e-1"),
          jnp.float32("5.6895751648208063447420e-2"),
          jnp.float32("-5.8425377541917452582255e-3"),
          jnp.float32("4.1688485296222501630869e-4"),
          jnp.float32("-2.1672192033668522592233e-5"),
          jnp.float32("8.3455660026061354242306e-7"),
          jnp.float32("-2.3611043040340310030999e-8"),
          jnp.float32("4.7209407442763668093845e-10"),
          jnp.float32("-6.0598273453439507784532e-12"),
          jnp.float32("3.8185020355501977364066e-14"),
      ],
  )
  return jnp.divide(numerator, denominator)


################################################################################
# 32-bit baselines for f=log2(x) on [1, 2]. ####################################
################################################################################


def table_log2_32bits_baselines() -> List[Tuple[str, FinalizedFn]]:
  """Baselines for 32-bit log2(x) on (1, 2].

  Returns:
    A list of (name, finalized_fn).
  """
  return [  # pytype: disable=bad-return-type  # jax-ndarray
      ("JAX Numpy", log32_jnp),
      ("Polynomial, Sollya, 5th order", log32_poly_sollya_5),
      ("Polynomial, Sollya, 6th order", log32_poly_sollya_6),
      ("Polynomial, Sollya, 7th order", log32_poly_sollya_7),
      ("Polynomial, Sollya, 8th order", log32_poly_sollya_8),
      ("Polynomial, Sollya, 9th order", log32_poly_sollya_9),
      ("Polynomial, Sollya, 10th order", log32_poly_sollya_10),
      ("Polynomial, Sollya, 11th order", log32_poly_sollya_11),
      ("Polynomial, Sollya, 12th order", log32_poly_sollya_12),
      ("Taylor expansion, 4th order", log32_poly_taylor_4),
      ("Taylor expansion, 6th order", log32_poly_taylor_6),
      ("Taylor expansion, 8th order", log32_poly_taylor_8),
      ("Taylor expansion, 10th order", log32_poly_taylor_10),
      ("Rational, Minimax, (3,3)-order", log32_rational_minimax_3_3),
      ("Rational, Minimax, (4,4)-order", log32_rational_minimax_4_4),
      ("Rational, Minimax, (5,5)-order", log32_rational_minimax_5_5),
      ("Rational, Minimax, (6,6)-order", log32_rational_minimax_6_6),
      ("Rational, Minimax, (7,7)-order", log32_rational_minimax_7_7),
      ("Rational, Minimax, (8,8)-order", log32_rational_minimax_8_8),
      ("Rational, Minimax, (9,9)-order", log32_rational_minimax_9_9),
      ("Rational, Minimax, (10,10)-order", log32_rational_minimax_10_10),
      ("Pade approximant, (3,3)-order", log32_rational_pade_3_3),
      ("Pade approximant, (4,4)-order", log32_rational_pade_4_4),
      ("Pade approximant, (5,5)-order", log32_rational_pade_5_5),
      ("Pade approximant, (6,6)-order", log32_rational_pade_6_6),
  ]


def log32_jnp(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns JAX NumPy approximation."""
  return jnp.log2(x[0])


def log32_poly_sollya_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  # pylint: disable=line-too-long
  """Returns the 5-th order Sollya-optimized polynomial.

  Expression:
  -0x1.729efp1 + x * (0x1.5bddc2p2 + x * (-0x1.028e88p2 + x * (0x1.f94dfcp0 + x
  * (-0x1.11cd9ep-1 + x * 0x1.f3396p-5))))

  Args:
    x: the arguments of the exp2 function.
  """
  # pylint: enable=line-too-long
  return _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("-0x1.729efp1")),
          jnp.float32(float.fromhex("0x1.5bddc2p2")),
          jnp.float32(float.fromhex("-0x1.028e88p2")),
          jnp.float32(float.fromhex("0x1.f94dfcp0")),
          jnp.float32(float.fromhex("-0x1.11cd9ep-1")),
          jnp.float32(float.fromhex("0x1.f3396p-5")),
      ],
  )


def log32_poly_sollya_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  # pylint: disable=line-too-long
  """Returns the 6-th order Sollya-optimized polynomial.

  Expression:
  -0x1.8f77c2p1 + x * (0x1.9d9242p2 + x * (-0x1.7d9c52p2 + x * (0x1.ef14f6p1 + x
  * (-0x1.91c3f6p0 + x * (0x1.6eac3p-2 + x * (-0x1.1f27cp-5))))))

  Args:
    x: the arguments of the exp2 function.
  """
  # pylint: enable=line-too-long
  return _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("-0x1.8f77c2p1")),
          jnp.float32(float.fromhex("0x1.9d9242p2")),
          jnp.float32(float.fromhex("-0x1.7d9c52p2")),
          jnp.float32(float.fromhex("0x1.ef14f6p1")),
          jnp.float32(float.fromhex("-0x1.91c3f6p0")),
          jnp.float32(float.fromhex("0x1.6eac3p-2")),
          jnp.float32(float.fromhex("-0x1.1f27cp-5")),
      ],
  )


def log32_poly_sollya_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  # pylint: disable=line-too-long
  """Returns the 7-th order Sollya-optimized polynomial.

  Expression:
  -0x1.a879c4p1 + x * (0x1.df5262p2 + x * (-0x1.08134ap3 + x * (0x1.aaeeeep2 + x
  * (-0x1.cd632p1 + x * (0x1.3c043ap0 + x * (-0x1.f02f3ap-3 + x *
  0x1.54215p-6))))))

  Args:
    x: the arguments of the exp2 function.
  """
  # pylint: enable=line-too-long
  return _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("-0x1.a879c4p1")),
          jnp.float32(float.fromhex("0x1.df5262p2")),
          jnp.float32(float.fromhex("-0x1.08134ap3")),
          jnp.float32(float.fromhex("0x1.aaeeeep2")),
          jnp.float32(float.fromhex("-0x1.cd632p1")),
          jnp.float32(float.fromhex("0x1.3c043ap0")),
          jnp.float32(float.fromhex("-0x1.f02f3ap-3")),
          jnp.float32(float.fromhex("0x1.54215p-6")),
      ],
  )


def log32_poly_sollya_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  # pylint: disable=line-too-long
  """Returns the 8-th order Sollya-optimized polynomial.

  Expression:
  -0x1.be7802p1 + x * (0x1.1073aap3 + x * (-0x1.5cd45ap3 + x * (0x1.517308p3 + x
  * (-0x1.c7364p2 + x * (0x1.9fafc4p1 + x * (-0x1.ea1ee6p-1 + x * (0x1.50d34ap-3
  + x * (-0x1.9a1d2p-7))))))))

  Args:
    x: the arguments of the exp2 function.
  """
  # pylint: enable=line-too-long
  return _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("-0x1.be7802p1")),
          jnp.float32(float.fromhex("0x1.1073aap3")),
          jnp.float32(float.fromhex("-0x1.5cd45ap3")),
          jnp.float32(float.fromhex("0x1.517308p3")),
          jnp.float32(float.fromhex("-0x1.c7364p2")),
          jnp.float32(float.fromhex("0x1.9fafc4p1")),
          jnp.float32(float.fromhex("-0x1.ea1ee6p-1")),
          jnp.float32(float.fromhex("0x1.50d34ap-3")),
          jnp.float32(float.fromhex("-0x1.9a1d2p-7")),
      ],
  )


def log32_poly_sollya_9(x: List[jnp.ndarray]) -> jnp.ndarray:
  # pylint: disable=line-too-long
  """Returns the 9-th order Sollya-optimized polynomial.

  Expression:
  -0x1.d1ac9p1 + x * (0x1.307736p3 + x * (-0x1.baec3ap3 + x * (0x1.f16ebap3 + x
  * (-0x1.9101a2p3 + x * (0x1.c852f6p2 + x * (-0x1.65e92cp1 + x * (0x1.707a72p-1
  + x * (-0x1.c078p-4 + x * 0x1.e947p-8))))))))

  Args:
    x: the arguments of the exp2 function.
  """
  # pylint: enable=line-too-long
  return _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("-0x1.d1ac9p1")),
          jnp.float32(float.fromhex("0x1.307736p3")),
          jnp.float32(float.fromhex("-0x1.baec3ap3")),
          jnp.float32(float.fromhex("0x1.f16ebap3")),
          jnp.float32(float.fromhex("-0x1.9101a2p3")),
          jnp.float32(float.fromhex("0x1.c852f6p2")),
          jnp.float32(float.fromhex("-0x1.65e92cp1")),
          jnp.float32(float.fromhex("0x1.707a72p-1")),
          jnp.float32(float.fromhex("-0x1.c078p-4")),
          jnp.float32(float.fromhex("0x1.e947p-8")),
      ],
  )


def log32_poly_sollya_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  # pylint: disable=line-too-long
  """Returns the 10-th order Sollya-optimized polynomial.

  Expression:
  -0x1.e387bp1 + x * (0x1.51508ep3 + x * (-0x1.13737p4 + x * (0x1.611b0cp4 + x *
  (-0x1.4bf8cap4 + x * (0x1.c58886p3 + x * (-0x1.bd3dcp2 + x * (0x1.3240dep1 + x
  * (-0x1.186344p-1 + x * (0x1.330f9cp-4 + x * (-0x1.3073cp-8))))))))))

  Args:
    x: the arguments of the exp2 function.
  """
  # pylint: enable=line-too-long
  return _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("-0x1.e387bp1")),
          jnp.float32(float.fromhex("0x1.51508ep3")),
          jnp.float32(float.fromhex("-0x1.13737p4")),
          jnp.float32(float.fromhex("0x1.611b0cp4")),
          jnp.float32(float.fromhex("-0x1.4bf8cap4")),
          jnp.float32(float.fromhex("0x1.c58886p3")),
          jnp.float32(float.fromhex("-0x1.bd3dcp2")),
          jnp.float32(float.fromhex("0x1.3240dep1")),
          jnp.float32(float.fromhex("-0x1.186344p-1")),
          jnp.float32(float.fromhex("0x1.330f9cp-4")),
          jnp.float32(float.fromhex("-0x1.3073cp-8")),
      ],
  )


def log32_poly_sollya_11(x: List[jnp.ndarray]) -> jnp.ndarray:
  # pylint: disable=line-too-long
  """Returns the 11-th order Sollya-optimized polynomial.

  Expression:
  -0x1.f566ap1 + x * (0x1.755ff8p3 + x * (-0x1.552b3ap4 + x * (0x1.efe24cp4 + x
  * (-0x1.0cb412p5 + x * (0x1.b04ef2p4 + x * (-0x1.013aa2p4 + x * (0x1.bf24d4p2
  + x * (-0x1.1414b6p1 + x * (0x1.cb12b2p-2 + x * (-0x1.cd096ap-5 + x *
  0x1.a6b5ap-9))))))))))

  Args:
    x: the arguments of the exp2 function.
  """
  # pylint: enable=line-too-long
  return _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("-0x1.f566ap1")),
          jnp.float32(float.fromhex("0x1.755ff8p3")),
          jnp.float32(float.fromhex("-0x1.552b3ap4")),
          jnp.float32(float.fromhex("0x1.efe24cp4")),
          jnp.float32(float.fromhex("-0x1.0cb412p5")),
          jnp.float32(float.fromhex("0x1.b04ef2p4")),
          jnp.float32(float.fromhex("-0x1.013aa2p4")),
          jnp.float32(float.fromhex("0x1.bf24d4p2")),
          jnp.float32(float.fromhex("-0x1.1414b6p1")),
          jnp.float32(float.fromhex("0x1.cb12b2p-2")),
          jnp.float32(float.fromhex("-0x1.cd096ap-5")),
          jnp.float32(float.fromhex("0x1.a6b5ap-9")),
      ],
  )


def log32_poly_sollya_12(x: List[jnp.ndarray]) -> jnp.ndarray:
  # pylint: disable=line-too-long
  """Returns the 12-th order Sollya-optimized polynomial.

  Expression:
  -0x1.f8474p1 + x * (0x1.7bcacp3 + x * (-0x1.62340ep4 + x * (0x1.07e252p5 + x *
  (-0x1.26dab8p5 + x * (0x1.eceaeep4 + x * (-0x1.341b0cp4 + x * (0x1.1de93p3 + x
  * (-0x1.82bd56p1 + x * (0x1.704d98p-1 + x * (-0x1.cfe69ap-4 + x *
  (0x1.560478p-7 + x * (-0x1.b3f5p-12))))))))))))

  Args:
    x: the arguments of the exp2 function.
  """
  # pylint: enable=line-too-long
  return _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("-0x1.f8474p1")),
          jnp.float32(float.fromhex("0x1.7bcacp3")),
          jnp.float32(float.fromhex("-0x1.62340ep4")),
          jnp.float32(float.fromhex("0x1.07e252p5")),
          jnp.float32(float.fromhex("-0x1.26dab8p5")),
          jnp.float32(float.fromhex("0x1.eceaeep4")),
          jnp.float32(float.fromhex("-0x1.341b0cp4")),
          jnp.float32(float.fromhex("0x1.1de93p3")),
          jnp.float32(float.fromhex("-0x1.82bd56p1")),
          jnp.float32(float.fromhex("0x1.704d98p-1")),
          jnp.float32(float.fromhex("-0x1.cfe69ap-4")),
          jnp.float32(float.fromhex("0x1.560478p-7")),
          jnp.float32(float.fromhex("-0x1.b3f5p-12")),
      ],
  )


def log32_poly_taylor_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 4-th order Taylor polynomial about x=1."""
  return _log32_poly_taylor(order=4, x=x)


def log32_poly_taylor_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 6-th order Taylor polynomial about x=1."""
  return _log32_poly_taylor(order=6, x=x)


def log32_poly_taylor_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 8-th order Taylor polynomial about x=1."""
  return _log32_poly_taylor(order=8, x=x)


def log32_poly_taylor_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 10-th order Taylor polynomial about x=1."""
  return _log32_poly_taylor(order=10, x=x)


def _log32_poly_taylor(order: int, x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the Taylor polynomial about x=1 of the given order."""
  coeffs = [
      jnp.float32("-4.2256079749212059165570e0"),
      jnp.float32("1.4426950408889634073599e1"),
      jnp.float32("-3.2460638420001676665598e1"),
      jnp.float32("5.7707801635558536294397e1"),
      jnp.float32("-7.5741489646670578886396e1"),
      jnp.float32("7.2711830060803755730940e1"),
      jnp.float32("-5.0494326431113719257597e1"),
      jnp.float32("2.4731914986667944126170e1"),
      jnp.float32("-8.1151596050004191663996e0"),
      jnp.float32("1.6029944898766260081777e0"),
      jnp.float32("-1.4426950408889634073599e-1"),
  ]
  if order >= len(coeffs):
    raise NotImplementedError("Missing coefficients.")
  coeffs = coeffs[: order + 1]
  return _horner_scheme(x[0], coeffs)


def log32_rational_minimax_3_3(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (3, 3)th order rational approximation with minimax coeffs.

  From: 2023-06-26.rational_function_baselines.nb

  Args:
    x: the arguments of the exp2 function.
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-4.88497255313338197102"),
          jnp.float32("-6.91241495889474723814"),
          jnp.float32("9.441252050324300302701"),
          jnp.float32("2.356135461703828906454"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("6.76142995363989856730"),
          jnp.float32("5.02474203319686348050"),
          jnp.float32("0.41030776284944301556"),
      ],
  )
  return jnp.divide(numerator, denominator)


def log32_rational_minimax_4_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (4, 4)th order rational approximation with minimax coeffs.

  From: 2023-06-26.rational_function_baselines.nb

  Args:
    x: the arguments of the exp2 function.
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-5.59015316533660532118"),
          jnp.float32("-23.47119083184940361612"),
          jnp.float32("8.77932655354832947660"),
          jnp.float32("18.39184261470559482605"),
          jnp.float32("1.89017482893208463465"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("11.90259054192306395414"),
          jnp.float32("19.76335566583154116897"),
          jnp.float32("6.42950460122252006632"),
          jnp.float32("0.291729313407458341652"),
      ],
  )
  return jnp.divide(numerator, denominator)


def log32_rational_minimax_5_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (5, 5)th order rational approximation with minimax coeffs.

  From: 2023-06-26.rational_function_baselines.nb

  Args:
    x: the arguments of the exp2 function.
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-6.15688839622280846046"),
          jnp.float32("-49.6246183998994829507"),
          jnp.float32("-27.8026620056199830690"),
          jnp.float32("56.3024496659224777102"),
          jnp.float32("25.8190155320540141440"),
          jnp.float32("1.4627036037657826260"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("18.4800082544010690007"),
          jnp.float32("54.2958393441878873593"),
          jnp.float32("39.6263743172523882391"),
          jnp.float32("7.1834459866594989626"),
          jnp.float32("0.20700261663030281778"),
      ],
  )
  return jnp.divide(numerator, denominator)


def log32_rational_minimax_6_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (6, 6)th order rational approximation with minimax coeffs.

  From: 2023-06-26.rational_function_baselines.nb

  Args:
    x: the arguments of the exp2 function.
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-6.63060449709011485061"),
          jnp.float32("-86.2829847560372687405"),
          jnp.float32("-149.026941542869493011"),
          jnp.float32("72.4084340710433363297"),
          jnp.float32("137.615495771916543106"),
          jnp.float32("30.8082674050293156862"),
          jnp.float32("1.10833354800768148023"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("26.4933734354124279268"),
          jnp.float32("121.219430562464028815"),
          jnp.float32("156.930362730170732267"),
          jnp.float32("63.9402857881547647779"),
          jnp.float32("7.37083117090184450218"),
          jnp.float32("0.14672543328776208962"),
      ],
  )
  return jnp.divide(numerator, denominator)


def log32_rational_minimax_7_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (7, 7)th order rational approximation with minimax coeffs.

  From: 2023-06-26.rational_function_baselines.nb

  Args:
    x: the arguments of the exp2 function.
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-7.0375306140421984931"),
          jnp.float32("-134.19238329193473554"),
          jnp.float32("-426.01811918331966037"),
          jnp.float32("-123.10198797525556447"),
          jnp.float32("411.21433553166340235"),
          jnp.float32("245.01391059041524429"),
          jnp.float32("33.293304802962622377"),
          jnp.float32("0.82847013951088985745"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("35.942662559671767884"),
          jnp.float32("236.22442702490321962"),
          jnp.float32("477.02396010381209840"),
          jnp.float32("345.21840544987292461"),
          jnp.float32("89.531982602748493884"),
          jnp.float32("7.1341099914000873570"),
          jnp.float32("0.10393617913040449234"),
      ],
  )
  return jnp.divide(numerator, denominator)


def log32_rational_minimax_8_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (8, 8)th order rational approximation with minimax coeffs.

  From: 2023-06-26.rational_function_baselines.nb

  Args:
    x: the arguments of the exp2 function.
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-7.3941605530587591702"),
          jnp.float32("-193.98272323861052300"),
          jnp.float32("-954.06276714087138345"),
          jnp.float32("-1012.0996153211830649"),
          jnp.float32("645.435141757604730957"),
          jnp.float32("1122.26680293499383317"),
          jnp.float32("365.585345691830911354"),
          jnp.float32("33.6385497656785135347"),
          jnp.float32("0.613426103615741477841"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("46.827509231637394702"),
          jnp.float32("418.08753231400108761"),
          jnp.float32("1214.1089750413717998"),
          jnp.float32("1371.8331645700593437"),
          jnp.float32("632.40041580773773380"),
          jnp.float32("113.42945588038030562"),
          jnp.float32("6.6170322988198994283"),
          jnp.float32("0.073592746389879572871"),
      ],
  )
  return jnp.divide(numerator, denominator)


def log32_rational_minimax_9_9(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (9, 9)th order rational approximation with minimax coeffs.

  From: 2023-06-26.rational_function_baselines.nb

  Args:
    x: the arguments of the exp2 function.
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-7.7115674744094592040"),
          jnp.float32("-266.20326191313577576"),
          jnp.float32("-1854.1209892380285888"),
          jnp.float32("-3570.9333750560153207"),
          jnp.float32("-468.65465391451863006"),
          jnp.float32("3353.51336578614553078"),
          jnp.float32("2295.76142077468924984"),
          jnp.float32("485.526961098370378523"),
          jnp.float32("32.3710149219980953691"),
          jnp.float32("0.451085014904519980181"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("59.148173281868381036"),
          jnp.float32("688.68481768788296978"),
          jnp.float32("2719.1284924026694488"),
          jnp.float32("4421.3905058738763021"),
          jnp.float32("3184.1383822058775370"),
          jnp.float32("1015.6061781553276867"),
          jnp.float32("133.40372168616532277"),
          jnp.float32("5.9418653219007575323"),
          jnp.float32("0.052094356199481672840"),
      ],
  )
  return jnp.divide(numerator, denominator)


def log32_rational_minimax_10_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (10, 10)th order rational approximation with minimax coeffs.

  From: 2023-06-26.rational_function_baselines.nb

  Args:
    x: the arguments of the exp2 function.
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-7.9975151826374413162"),
          jnp.float32("-351.33531427094746533"),
          jnp.float32("-3274.0298513095105656"),
          jnp.float32("-9532.9856177832830362"),
          jnp.float32("-7170.4314546141950196"),
          jnp.float32("6036.72522941784398492"),
          jnp.float32("9724.13076209727870630"),
          jnp.float32("3952.54830828751874247"),
          jnp.float32("593.027062834891147675"),
          jnp.float32("30.0184272366793665963"),
          jnp.float32("0.329963286361580176491"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("72.904243682161307084"),
          jnp.float32("1072.9703459888810738"),
          jnp.float32("5528.2438588969472375"),
          jnp.float32("12228.498539608935873"),
          jnp.float32("12679.137221663741319"),
          jnp.float32("6320.0801894678350428"),
          jnp.float32("1476.6614920852198203"),
          jnp.float32("148.12061369529118404"),
          jnp.float32("5.2011483499082922423"),
          jnp.float32("0.0368675820451393079"),
      ],
  )
  return jnp.divide(numerator, denominator)


def log32_rational_pade_3_3(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (3, 3)th order Pade approximant about x=0."""
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-2.6449409082964329134932e-1"),
          jnp.float32("-6.4921276840003353331197e-1"),
          jnp.float32("6.4921276840003353331197e-1"),
          jnp.float32("2.6449409082964329134932e-1"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("5.0000000000000000000000e-2"),
          jnp.float32("4.5000000000000000000000e-1"),
          jnp.float32("4.5000000000000000000000e-1"),
          jnp.float32("5.0000000000000000000000e-2"),
      ],
  )
  return jnp.divide(numerator, denominator)


def log32_rational_pade_4_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (4, 4)th order Pade approximant about x=0."""
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-8.5874704814819250438091e-2"),
          jnp.float32("-5.4959811081484320280378e-1"),
          None,
          jnp.float32("5.4959811081484320280378e-1"),
          jnp.float32("8.5874704814819250438091e-2"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.4285714285714285714286e-2"),
          jnp.float32("2.2857142857142857142857e-1"),
          jnp.float32("5.1428571428571428571429e-1"),
          jnp.float32("2.2857142857142857142857e-1"),
          jnp.float32("1.4285714285714285714286e-2"),
      ],
  )
  return jnp.divide(numerator, denominator)


def log32_rational_pade_5_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (5, 5)th order Pade approximant about x=0."""
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-2.6144076799178305133374e-2"),
          jnp.float32("-3.1010310072018062658199e-1"),
          jnp.float32("-3.8166535473253000194707e-1"),
          jnp.float32("3.8166535473253000194707e-1"),
          jnp.float32("3.1010310072018062658199e-1"),
          jnp.float32("2.6144076799178305133374e-2"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("3.9682539682539682539683e-3"),
          jnp.float32("9.9206349206349206349206e-2"),
          jnp.float32("3.9682539682539682539683e-1"),
          jnp.float32("3.9682539682539682539683e-1"),
          jnp.float32("9.9206349206349206349206e-2"),
          jnp.float32("3.9682539682539682539683e-3"),
      ],
  )
  return jnp.divide(numerator, denominator)


def log32_rational_pade_6_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (6, 6)th order Pade approximant about x=0."""
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-7.6506555198657150390299e-3"),
          jnp.float32("-1.4426950408889634073599e-1"),
          jnp.float32("-4.0985654570709187709089e-1"),
          None,
          jnp.float32("4.0985654570709187709089e-1"),
          jnp.float32("1.4426950408889634073599e-1"),
          jnp.float32("7.6506555198657150390299e-3"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0822510822510822510823e-3"),
          jnp.float32("3.8961038961038961038961e-2"),
          jnp.float32("2.4350649350649350649351e-1"),
          jnp.float32("4.3290043290043290043290e-1"),
          jnp.float32("2.4350649350649350649351e-1"),
          jnp.float32("3.8961038961038961038961e-2"),
          jnp.float32("1.0822510822510822510823e-3"),
      ],
  )
  return jnp.divide(numerator, denominator)


################################################################################
# 32-bit baselines for f=erf01(x) on [0, 1]. ###################################
################################################################################


def table_erf_32bits_0to1_baselines() -> List[Tuple[str, FinalizedFn]]:
  """Baselines for 32-bit log2(x) on (1, 2].

  Returns:
    A list of (name, finalized_fn).
  """
  return [  # pytype: disable=bad-return-type  # jax-ndarray
      ("JAX Numpy", erf32_0_1_jscipy),
      ("Polynomial, Sollya, 4th order", erf32_0_1_poly_sollya_4),
      ("Polynomial, Sollya, 6th order", erf32_0_1_poly_sollya_6),
      ("Polynomial, Sollya, 7th order", erf32_0_1_poly_sollya_7),
      ("Polynomial, Sollya, 8th order", erf32_0_1_poly_sollya_8),
      ("Polynomial, Sollya, 10th order", erf32_0_1_poly_sollya_10),
      ("Polynomial, Sollya, 11th order", erf32_0_1_poly_sollya_11),
      ("Polynomial, Sollya, 12th order", erf32_0_1_poly_sollya_12),
      ("Polynomial, Sollya, 13th order", erf32_0_1_poly_sollya_13),
      ("Polynomial, Sollya, 14th order", erf32_0_1_poly_sollya_14),
      ("Polynomial, Sollya, 16th order", erf32_0_1_poly_sollya_16),
      ("Taylor expansion, 10th order", erf32_0_1_poly_taylor_10),
      ("Taylor expansion, 18th order", erf32_0_1_poly_taylor_18),
      ("Rational, Minimax, (2,2)-order", erf32_0_1_rational_minimax_2_2),
      ("Rational, Minimax, (3,3)-order", erf32_0_1_rational_minimax_3_3),
      ("Rational, Minimax, (4,4)-order", erf32_0_1_rational_minimax_4_4),
      ("Rational, Minimax, (5,5)-order", erf32_0_1_rational_minimax_5_5),
      ("Rational, Minimax, (6,6)-order", erf32_0_1_rational_minimax_6_6),
      ("Rational, Minimax, (7,7)-order", erf32_0_1_rational_minimax_7_7),
      ("Pade approximant, (4,4)-order", erf32_0_1_rational_pade_4_4),
      ("Pade approximant, (6,6)-order", erf32_0_1_rational_pade_6_6),
      ("Pade approximant, (7,7)-order", erf32_0_1_rational_pade_7_7),
      ("Pade approximant, (8,8)-order", erf32_0_1_rational_pade_8_8),
      ("Pade approximant, (9,9)-order", erf32_0_1_rational_pade_9_9),
      ("Pade approximant, (10,10)-order", erf32_0_1_rational_pade_10_10),
      ("Pade approximant, (16,16)-order", erf32_0_1_rational_pade_16_16),
  ]


def erf32_0_1_jscipy(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns JAX NumPy approximation."""
  return jax.scipy.special.erf(x[0])


def erf32_0_1_poly_sollya_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 4-th order Sollya-optimized polynomial."""
  return _horner_scheme(
      x[0],
      [
          jnp.float32("7.142707545426674187183380126953125e-7"),
          jnp.float32("1.12792193889617919921875"),
          jnp.float32("1.2856344692409038543701171875e-2"),
          jnp.float32("-0.4503224790096282958984375"),
          jnp.float32("0.15204274654388427734375"),
      ],
  )


def erf32_0_1_poly_sollya_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 6-th order Sollya-optimized polynomial."""
  return _horner_scheme(
      x[0],
      [
          jnp.float32("-1.119280224060048567480407655239105224609375e-9"),
          jnp.float32("1.12838137149810791015625"),
          jnp.float32("-1.5063575119711458683013916015625e-4"),
          jnp.float32("-0.373941957950592041015625"),
          jnp.float32("-1.239164732396602630615234375e-2"),
          jnp.float32("0.14649288356304168701171875"),
          jnp.float32("-4.569016396999359130859375e-2"),
      ],
  )


def erf32_0_1_poly_sollya_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 7-th order Sollya-optimized polynomial."""
  return _horner_scheme(
      x[0],
      [
          jnp.float32("-3.89418364221683077630586922168731689453125e-9"),
          jnp.float32("1.1283819675445556640625"),
          jnp.float32("-1.6426542424596846103668212890625e-4"),
          jnp.float32("-0.3738418519496917724609375"),
          jnp.float32("-1.2717790901660919189453125e-2"),
          jnp.float32("0.1470175683498382568359375"),
          jnp.float32("-4.60989214479923248291015625e-2"),
          jnp.float32("1.23170568258501589298248291015625e-4"),
      ],
  )


def erf32_0_1_poly_sollya_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 8-th order Sollya-optimized polynomial."""
  return _horner_scheme(
      x[0],
      [
          jnp.float32("-7.8630831013004609530980815179646015167236328125e-11"),
          jnp.float32("1.12837922573089599609375"),
          jnp.float32("-4.228859324939548969268798828125e-6"),
          jnp.float32("-0.376065909862518310546875"),
          jnp.float32("-2.792896120809018611907958984375e-4"),
          jnp.float32("0.1130127608776092529296875"),
          jnp.float32("2.24622595123946666717529296875e-3"),
          jnp.float32("-3.42190153896808624267578125e-2"),
          jnp.float32("9.631057269871234893798828125e-3"),
      ],
  )


def erf32_0_1_poly_sollya_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 10-th order Sollya-optimized polynomial."""
  return _horner_scheme(
      x[0],
      [
          jnp.float32("8.2564212411373461009134189225733280181884765625e-11"),
          jnp.float32("1.1283791065216064453125"),
          jnp.float32("3.505581616991548798978328704833984375e-6"),
          jnp.float32("-0.3761885464191436767578125"),
          jnp.float32("5.04935975186526775360107421875e-4"),
          jnp.float32("0.110612094402313232421875"),
          jnp.float32("5.7438970543444156646728515625e-3"),
          jnp.float32("-3.56801785528659820556640625e-2"),
          jnp.float32("7.56282173097133636474609375e-3"),
          jnp.float32("2.62916763313114643096923828125e-3"),
          jnp.float32("-8.66009271703660488128662109375e-4"),
      ],
  )


def erf32_0_1_poly_sollya_11(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 11-th order Sollya-optimized polynomial."""
  return _horner_scheme(
      x[0],
      [
          jnp.float32("-7.7404409271064622544145095162093639373779296875e-11"),
          jnp.float32("1.12837922573089599609375"),
          jnp.float32("-3.96638324673403985798358917236328125e-6"),
          jnp.float32("-0.3760426044464111328125"),
          jnp.float32("-8.225184865295886993408203125e-4"),
          jnp.float32("0.1173066198825836181640625"),
          jnp.float32("-1.4680976979434490203857421875e-2"),
          jnp.float32("3.58506408520042896270751953125e-3"),
          jnp.float32("-4.031713306903839111328125e-2"),
          jnp.float32("3.8568072021007537841796875e-2"),
          jnp.float32("-1.601711474359035491943359375e-2"),
          jnp.float32("2.74612731300294399261474609375e-3"),
      ],
  )


def erf32_0_1_poly_sollya_12(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 12-th order Sollya-optimized polynomial."""
  return _horner_scheme(
      x[0],
      [
          jnp.float32("-7.5198534899012514642890891991555690765380859375e-11"),
          jnp.float32("1.12837922573089599609375"),
          jnp.float32("-4.65263883597799576818943023681640625e-6"),
          jnp.float32("-0.3760094344615936279296875"),
          jnp.float32("-1.3768677599728107452392578125e-3"),
          jnp.float32("0.121891252696514129638671875"),
          jnp.float32("-3.64644229412078857421875e-2"),
          jnp.float32("6.760726869106292724609375e-2"),
          jnp.float32("-0.16062264144420623779296875"),
          jnp.float32("0.18340218067169189453125"),
          jnp.float32("-0.124103136360645294189453125"),
          jnp.float32("4.82790805399417877197265625e-2"),
          jnp.float32("-8.277061395347118377685546875e-3"),
      ],
  )


def erf32_0_1_poly_sollya_13(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 13-th order Sollya-optimized polynomial."""
  return _horner_scheme(
      x[0],
      [
          jnp.float32("-7.2774598847136218182640732266008853912353515625e-11"),
          jnp.float32("1.12837922573089599609375"),
          jnp.float32("-5.385345502872951328754425048828125e-6"),
          jnp.float32("-0.3759670257568359375"),
          jnp.float32("-2.22826306708157062530517578125e-3"),
          jnp.float32("0.13040407001972198486328125"),
          jnp.float32("-8.58866870403289794921875e-2"),
          jnp.float32("0.24798591434955596923828125"),
          jnp.float32("-0.59190165996551513671875"),
          jnp.float32("0.869235813617706298828125"),
          jnp.float32("-0.843082249164581298828125"),
          jnp.float32("0.525652468204498291015625"),
          jnp.float32("-0.1901988983154296875"),
          jnp.float32("3.031347133219242095947265625e-2"),
      ],
  )


def erf32_0_1_poly_sollya_14(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 14-th order Sollya-optimized polynomial."""
  return _horner_scheme(
      x[0],
      [
          jnp.float32("-7.0698502607768887173733673989772796630859375e-11"),
          jnp.float32("1.12837922573089599609375"),
          jnp.float32("-6.13982092545484192669391632080078125e-6"),
          jnp.float32("-0.375915944576263427734375"),
          jnp.float32("-3.43443662859499454498291015625e-3"),
          jnp.float32("0.14468371868133544921875"),
          jnp.float32("-0.18494059145450592041015625"),
          jnp.float32("0.68541729450225830078125"),
          jnp.float32("-1.880005359649658203125"),
          jnp.float32("3.4571816921234130859375"),
          jnp.float32("-4.40484523773193359375"),
          jnp.float32("3.8299314975738525390625"),
          jnp.float32("-2.1655051708221435546875"),
          jnp.float32("0.717438280582427978515625"),
          jnp.float32("-0.10567803680896759033203125"),
      ],
  )


def erf32_0_1_poly_sollya_16(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 16-th order Sollya-optimized polynomial."""
  return _horner_scheme(
      x[0],
      [
          jnp.float32("-6.6593196235142926298067322932183742523193359375e-11"),
          jnp.float32("1.12837922573089599609375"),
          jnp.float32("-7.7337217589956708252429962158203125e-6"),
          jnp.float32("-0.3757815659046173095703125"),
          jnp.float32("-7.4239545501768589019775390625e-3"),
          jnp.float32("0.20476011931896209716796875"),
          jnp.float32("-0.72297871112823486328125"),
          jnp.float32("3.81307315826416015625"),
          jnp.float32("-14.31510162353515625"),
          jnp.float32("38.350589752197265625"),
          jnp.float32("-74.67037200927734375"),
          jnp.float32("105.80786895751953125"),
          jnp.float32("-107.9329376220703125"),
          jnp.float32("77.18360137939453125"),
          jnp.float32("-36.710056304931640625"),
          jnp.float32("10.4271717071533203125"),
          jnp.float32("-1.3380839824676513671875"),
      ],
  )


def erf32_0_1_poly_taylor_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 10-th order Taylor polynomial about x=1."""
  return _erf32_0_1_poly_taylor(order=10, x=x)


def erf32_0_1_poly_taylor_18(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 20-th order Taylor polynomial about x=1."""
  return _erf32_0_1_poly_taylor(order=18, x=x)


def _erf32_0_1_poly_taylor(order: int, x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the Taylor polynomial about x=1 of the given order."""
  coeffs = [
      jnp.float32("1.1283791670955125738962e0"),
      None,
      jnp.float32("-3.7612638903183752463205e-1"),
      None,
      jnp.float32("1.1283791670955125738962e-1"),
      None,
      jnp.float32("-2.6866170645131251759432e-2"),
      None,
      jnp.float32("5.2239776254421878421118e-3"),
      None,
      jnp.float32("-8.5483270234508528325467e-4"),
      None,
      jnp.float32("1.2055332981789664251027e-4"),
      None,
      jnp.float32("-1.4925650358406250977462e-5"),
      None,
      jnp.float32("1.6462114365889247401613e-6"),
      None,
      jnp.float32("-1.6365844691234924317393e-7"),
  ]
  if order >= len(coeffs):
    raise NotImplementedError("Missing coefficients.")
  coeffs = coeffs[: order + 1]
  return _horner_scheme(x[0], coeffs)


def erf32_0_1_rational_minimax_2_2(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (2, 2)th order rational approximation with minimax coeffs."""
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.9878702785541921229731e-7"),
          jnp.float32("1.1282591066914507989086"),
          jnp.float32("-1.0743945110140126072755e-2"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("-1.1290154270688239178320e-2"),
          jnp.float32("3.3749669965298691069295e-1"),
      ],
  )
  return jnp.divide(numerator, denominator)


def erf32_0_1_rational_minimax_3_3(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (3, 3)th order rational approximation with minimax coeffs."""
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-1.8703520380358571378470e-8"),
          jnp.float32("1.1283917411148391189123"),
          jnp.float32("-8.3340789173245068441856e-1"),
          jnp.float32("2.8677792816508094267386e-2"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("-7.3814118624545541396612e-1"),
          jnp.float32("3.5473527757742365496799e-1"),
          jnp.float32("-2.3251978743918479859880e-1"),
      ],
  )
  return jnp.divide(numerator, denominator)


def erf32_0_1_rational_minimax_4_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (4, 4)th order rational approximation with minimax coeffs."""
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("5.3713829575927563517965e-11"),
          jnp.float32("1.1283791265296846588310"),
          jnp.float32("-4.7431246085752544237314e-1"),
          jnp.float32("1.6578447050432443877665e-1"),
          jnp.float32("2.7262790385575842681883e-2"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("-4.2035116106150474561421e-1"),
          jnp.float32("4.8030627293907132602769e-1"),
          jnp.float32("-1.1634550365733770616319e-1"),
          jnp.float32("6.1627298891778475456634e-2"),
      ],
  )
  return jnp.divide(numerator, denominator)


def erf32_0_1_rational_minimax_5_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (5, 5)th order rational approximation with minimax coeffs."""
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-6.4747676149014660029990e-13"),
          jnp.float32("1.1283791676308590986675e0"),
          jnp.float32("4.9644687599641965864754e-1"),
          jnp.float32("-1.2990715444289608451695e-1"),
          jnp.float32("1.1201481704625980067190e-1"),
          jnp.float32("2.7538360709035119110526e-2"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1e0"),
          jnp.float32("4.3996463980270401282523e-1"),
          jnp.float32("2.1820462282649054195984e-1"),
          jnp.float32("2.4594390477008005051273e-1"),
          jnp.float32("-2.9793020909441815786330e-3"),
          jnp.float32("3.8430070515459661844594e-2"),
      ],
  )
  return jnp.divide(numerator, denominator)


def erf32_0_1_rational_minimax_6_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (6, 6)th order rational approximation with minimax coeffs."""
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("2.7302358994713730342668e-15"),
          jnp.float32("1.1283791670931095771873"),
          jnp.float32("9.3254345781152920435991e-2"),
          jnp.float32("1.4840758222896727868669e-1"),
          jnp.float32("-8.1144466321223732609172e-3"),
          jnp.float32("4.1958592435420510589785e-2"),
          jnp.float32("2.1285251485343059613199e-3"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("8.2644511860475960369380e-2"),
          jnp.float32("4.6485613968917291310109e-1"),
          jnp.float32("2.0356743445811125570233e-2"),
          jnp.float32("9.2138557209015542540455e-2"),
          jnp.float32("3.9850781566898841953938e-4"),
          jnp.float32("8.0669652199746676075109e-3"),
      ],
  )
  return jnp.divide(numerator, denominator)


def erf32_0_1_rational_minimax_7_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (7, 7)th order rational approximation with minimax coeffs."""
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-1.2691004261975409448108e-17"),
          jnp.float32("1.1283791670955244863639"),
          jnp.float32("-3.4155143650617117772759e-1"),
          jnp.float32("1.4151818682101682676148e-1"),
          jnp.float32("-2.9045762087928854740341e-2"),
          jnp.float32("3.3360421097146878089049e-2"),
          jnp.float32("-8.4291807972367366350864e-3"),
          jnp.float32("5.1590311324983491721485e-4"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("-3.0269207945710337922259e-1"),
          jnp.float32("4.5875056084864395199009e-1"),
          jnp.float32("-1.2663849430545715337949e-1"),
          jnp.float32("8.2481733167563395688983e-2"),
          jnp.float32("-1.9413611830956959088589e-2"),
          jnp.float32("5.8847159770353360007381e-3"),
          jnp.float32("-1.0114519264705703767303e-3"),
      ],
  )
  return jnp.divide(numerator, denominator)


def erf32_0_1_rational_pade_4_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (4, 4)th order Pade approximant about x=0."""
  numerator = _horner_scheme(
      x[0],
      [
          None,
          jnp.float32("1.1283791670955125738962e0"),
          None,
          jnp.float32("5.9105575419288753870751e-1"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          None,
          jnp.float32("8.5714285714285714285714e-1"),
          None,
          jnp.float32("1.8571428571428571428571e-1"),
      ],
  )
  return jnp.divide(numerator, denominator)


def erf32_0_1_rational_pade_6_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (6, 6)th order Pade approximant about x=0."""
  numerator = _horner_scheme(
      x[0],
      [
          None,
          jnp.float32("1.1283791670955125738962e0"),
          None,
          jnp.float32("1.1317568551751440340140e-1"),
          None,
          jnp.float32("3.3311870314118763176141e-2"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          None,
          jnp.float32("4.3363267314552835527125e-1"),
          None,
          jnp.float32("7.4066100791405256899168e-2"),
          None,
          jnp.float32("5.1349567587727262964069e-3"),
      ],
  )
  return jnp.divide(numerator, denominator)


def erf32_0_1_rational_pade_7_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (7, 7)th order Pade approximant about x=0."""
  numerator = _horner_scheme(
      x[0],
      [
          None,
          jnp.float32("1.1283791670955125738962e0"),
          None,
          jnp.float32("9.8063182994563590769408e-2"),
          None,
          jnp.float32("3.2213953464160798155867e-2"),
          None,
          jnp.float32("-2.2727186333863757702572e-4"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          None,
          jnp.float32("4.2023956649871660804259e-1"),
          None,
          jnp.float32("6.8628728434793650118301e-2"),
          None,
          jnp.float32("4.4603955265615119213370e-3"),
      ],
  )
  return jnp.divide(numerator, denominator)


def erf32_0_1_rational_pade_8_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (8, 8)th order Pade approximant about x=0."""
  numerator = _horner_scheme(
      x[0],
      [
          None,
          jnp.float32("1.1283791670955125738962e0"),
          None,
          jnp.float32("4.5634420770887662952300e-1"),
          None,
          jnp.float32("6.8149303710243124509716e-2"),
          None,
          jnp.float32("1.0349856268168709462407e-2"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          None,
          jnp.float32("7.3775785747934586094240e-1"),
          None,
          jnp.float32("2.0631503372532361893019e-1"),
          None,
          jnp.float32("2.7977737269447540873249e-2"),
          None,
          jnp.float32("1.6304426943049473623013e-3"),
      ],
  )
  return jnp.divide(numerator, denominator)


def erf32_0_1_rational_pade_9_9(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (9, 9)th order Pade approximant about x=0."""
  numerator = _horner_scheme(
      x[0],
      [
          None,
          jnp.float32("1.1283791670955125738962e0"),
          None,
          jnp.float32("1.2816805383758214973694e-1"),
          None,
          jnp.float32("3.9628750362585707429091e-2"),
          None,
          jnp.float32("9.8079695092744124672501e-4"),
          None,
          jnp.float32("6.6099417792001126781560e-5"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          None,
          jnp.float32("4.4691931362707732479836e-1"),
          None,
          jnp.float32("8.4093170135728309107513e-2"),
          None,
          jnp.float32("8.0178578250393792466426e-3"),
          None,
          jnp.float32("3.3318775435462466917652e-4"),
      ],
  )
  return jnp.divide(numerator, denominator)


def erf32_0_1_rational_pade_10_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (10, 10)th order Pade approximant about x=0."""
  numerator = _horner_scheme(
      x[0],
      [
          None,
          jnp.float32("1.1283791670955125738962e0"),
          None,
          jnp.float32("1.4199224411779842923427e-1"),
          None,
          jnp.float32("4.5219591938366107846813e-2"),
          None,
          jnp.float32("1.8157192619974617875246e-3"),
          None,
          jnp.float32("1.9289933274642390922058e-4"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          None,
          jnp.float32("4.5917068327598729944380e-1"),
          None,
          jnp.float32("9.3131714359095617719936e-2"),
          None,
          jnp.float32("1.0545499567335646850244e-2"),
          None,
          jnp.float32("6.7595335558271170071120e-4"),
          None,
          jnp.float32("1.9975156139294636083392e-5"),
      ],
  )
  return jnp.divide(numerator, denominator)


def erf32_0_1_rational_pade_16_16(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (16, 16)th order Pade approximant about x=0."""
  numerator = _horner_scheme(
      x[0],
      [
          None,
          jnp.float32("1.1283791670955125738962e0"),
          None,
          jnp.float32("3.3997004981802198526596e-1"),
          None,
          jnp.float32("7.6255673689647398547647e-2"),
          None,
          jnp.float32("1.1187997872420654930688e-2"),
          None,
          jnp.float32("8.3423597877077414090881e-4"),
          None,
          jnp.float32("6.8436631572243194183460e-5"),
          None,
          jnp.float32("1.7901917365815379278755e-6"),
          None,
          jnp.float32("7.4863870936087738734559e-8"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0000000000000000000000e0"),
          None,
          jnp.float32("6.3462394532957993137377e-1"),
          None,
          jnp.float32("1.7912114635216496282292e-1"),
          None,
          jnp.float32("2.9969283017068159013209e-2"),
          None,
          jnp.float32("3.2974330638299256918584e-3"),
          None,
          jnp.float32("2.4715757395632222310009e-4"),
          None,
          jnp.float32("1.2456968146329319597806e-5"),
          None,
          jnp.float32("3.9018991921310246456080e-7"),
          None,
          jnp.float32("5.8842261951384946903630e-9"),
      ],
  )
  return jnp.divide(numerator, denominator)


################################################################################
# 32-bit baselines for f=leading_asymptotic * erf15(x) on [1, 5]. ##############
################################################################################


def table_erf_32bits_1to5_baselines() -> List[Tuple[str, FinalizedFn]]:
  """Baselines for 32-bit log2(x) on (1, 2].

  Returns:
    A list of (name, finalized_fn).
  """
  return [  # pytype: disable=bad-return-type  # jax-ndarray
      ("JAX Numpy", erf32_1_5_jscipy),
      ("Polynomial, Sollya, 4th order", erf32_1_5_poly_sollya_4),
      ("Polynomial, Sollya, 8th order", erf32_1_5_poly_sollya_8),
      ("Polynomial, Sollya, 12th order", erf32_1_5_poly_sollya_12),
      ("Polynomial, Sollya, 13th order", erf32_1_5_poly_sollya_13),
      ("Polynomial, Sollya, 14th order", erf32_1_5_poly_sollya_14),
      ("Polynomial, Sollya, 15th order", erf32_1_5_poly_sollya_15),
      ("Polynomial, Sollya, 16th order", erf32_1_5_poly_sollya_16),
      ("Taylor expansion, 4th order", erf32_1_5_poly_taylor_4),
      ("Taylor expansion, 6th order", erf32_1_5_poly_taylor_6),
      ("Taylor expansion, 8th order", erf32_1_5_poly_taylor_8),
      ("Taylor expansion, 10th order", erf32_1_5_poly_taylor_10),
      ("Taylor expansion, 12th order", erf32_1_5_poly_taylor_12),
      ("Taylor expansion, 14th order", erf32_1_5_poly_taylor_14),
      ("Taylor expansion, 16th order", erf32_1_5_poly_taylor_16),
      ("Taylor expansion, 18th order", erf32_1_5_poly_taylor_18),
      ("Taylor expansion, 20th order", erf32_1_5_poly_taylor_20),
      ("Rational, Minimax, (2,2)-order", erf32_1_5_rational_minimax_2_2),
      ("Rational, Minimax, (3,3)-order", erf32_1_5_rational_minimax_3_3),
      ("Rational, Minimax, (4,4)-order", erf32_1_5_rational_minimax_4_4),
      ("Rational, Minimax, (5,5)-order", erf32_1_5_rational_minimax_5_5),
      ("Rational, Minimax, (6,6)-order", erf32_1_5_rational_minimax_6_6),
      ("Rational, Minimax, (7,7)-order", erf32_1_5_rational_minimax_7_7),
      ("Pade approximant, (1,1)-order", erf32_1_5_rational_pade_1_1),
      ("Pade approximant, (2,2)-order", erf32_1_5_rational_pade_2_2),
      ("Pade approximant, (3,3)-order", erf32_1_5_rational_pade_3_3),
      ("Pade approximant, (4,4)-order", erf32_1_5_rational_pade_4_4),
      ("Pade approximant, (5,5)-order", erf32_1_5_rational_pade_5_5),
  ]


def erf32_1_5_jscipy(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns JAX NumPy approximation."""
  return jax.scipy.special.erf(x[0])


def erf32_1_5_following(x: jnp.ndarray, poly: jnp.ndarray) -> jnp.ndarray:
  return (
      jnp.float32("1")
      - poly * jnp.exp(-jnp.multiply(x, x)) / jnp.sqrt(jnp.float32(jnp.pi)) / x
  )


def erf32_1_5_poly_sollya_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 4-th order Sollya-optimized polynomial."""
  poly = _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("0x1.6b219ep-2")),
          jnp.float32(float.fromhex("0x1.345732p-1")),
          jnp.float32(float.fromhex("-0x1.e70076p-3")),
          jnp.float32(float.fromhex("0x1.6335p-5")),
          jnp.float32(float.fromhex("-0x1.86767ap-9")),
      ],
  )
  return erf32_1_5_following(x[0], poly)


def erf32_1_5_poly_sollya_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 6-th order Sollya-optimized polynomial."""
  poly = _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("0x1.7283bap-3")),
          jnp.float32(float.fromhex("0x1.0a054ep0")),
          jnp.float32(float.fromhex("-0x1.502ca2p-1")),
          jnp.float32(float.fromhex("0x1.eba702p-3")),
          jnp.float32(float.fromhex("-0x1.a2b8bep-5")),
          jnp.float32(float.fromhex("0x1.806bacp-8")),
          jnp.float32(float.fromhex("-0x1.2583a4p-12")),
      ],
  )
  return erf32_1_5_following(x[0], poly)


def erf32_1_5_poly_sollya_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 8-th order Sollya-optimized polynomial."""
  poly = _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("0x1.4ff108p-4")),
          jnp.float32(float.fromhex("0x1.5ffe8ep0")),
          jnp.float32(float.fromhex("-0x1.20597ap0")),
          jnp.float32(float.fromhex("0x1.30224ep-1")),
          jnp.float32(float.fromhex("-0x1.ab481ap-3")),
          jnp.float32(float.fromhex("0x1.8c96c6p-5")),
          jnp.float32(float.fromhex("-0x1.d2413p-8")),
          jnp.float32(float.fromhex("0x1.39f5e8p-11")),
          jnp.float32(float.fromhex("-0x1.70754p-16")),
      ],
  )
  return erf32_1_5_following(x[0], poly)


def erf32_1_5_poly_sollya_9(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 9-th order Sollya-optimized polynomial."""
  poly = _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("0x1.b302cap-5")),
          jnp.float32(float.fromhex("0x1.7e57acp0")),
          jnp.float32(float.fromhex("-0x1.554e9ep0")),
          jnp.float32(float.fromhex("0x1.97702ap-1")),
          jnp.float32(float.fromhex("-0x1.51f0eap-2")),
          jnp.float32(float.fromhex("0x1.8606b4p-4")),
          jnp.float32(float.fromhex("-0x1.32958ep-6")),
          jnp.float32(float.fromhex("0x1.38746cp-9")),
          jnp.float32(float.fromhex("-0x1.73a676p-13")),
          jnp.float32(float.fromhex("0x1.872c0ap-18")),
      ],
  )
  return erf32_1_5_following(x[0], poly)


def erf32_1_5_poly_sollya_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 10-th order Sollya-optimized polynomial."""
  poly = _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("0x1.12fd24p-5")),
          jnp.float32(float.fromhex("0x1.95130cp0")),
          jnp.float32(float.fromhex("-0x1.820e2ap0")),
          jnp.float32(float.fromhex("0x1.fbf12ep-1")),
          jnp.float32(float.fromhex("-0x1.e0afc2p-2")),
          jnp.float32(float.fromhex("0x1.492bc8p-3")),
          jnp.float32(float.fromhex("-0x1.42962ep-5")),
          jnp.float32(float.fromhex("0x1.b805b8p-8")),
          jnp.float32(float.fromhex("-0x1.8bf194p-11")),
          jnp.float32(float.fromhex("0x1.a5e1f4p-15")),
          jnp.float32(float.fromhex("-0x1.92a7aep-20")),
      ],
  )
  return erf32_1_5_following(x[0], poly)


def erf32_1_5_poly_sollya_11(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 11-th order Sollya-optimized polynomial."""
  poly = _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("0x1.53df58p-6")),
          jnp.float32(float.fromhex("0x1.a575c6p0")),
          jnp.float32(float.fromhex("-0x1.a5f808p0")),
          jnp.float32(float.fromhex("0x1.2b9c08p0")),
          jnp.float32(float.fromhex("-0x1.3b1f0ap-1")),
          jnp.float32(float.fromhex("0x1.ef38a8p-3")),
          jnp.float32(float.fromhex("-0x1.20f1dcp-4")),
          jnp.float32(float.fromhex("0x1.ec3564p-7")),
          jnp.float32(float.fromhex("-0x1.2886aap-9")),
          jnp.float32(float.fromhex("0x1.ddd3c8p-13")),
          jnp.float32(float.fromhex("-0x1.cd2fccp-17")),
          jnp.float32(float.fromhex("0x1.92b47ap-22")),
      ],
  )
  return erf32_1_5_following(x[0], poly)


def erf32_1_5_poly_sollya_12(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 12-th order Sollya-optimized polynomial."""
  poly = _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("0x1.9c40e4p-7")),
          jnp.float32(float.fromhex("0x1.b0d41ap0")),
          jnp.float32(float.fromhex("-0x1.c1708ep0")),
          jnp.float32(float.fromhex("0x1.529b36p0")),
          jnp.float32(float.fromhex("-0x1.839a0ap-1")),
          jnp.float32(float.fromhex("0x1.5497e6p-2")),
          jnp.float32(float.fromhex("-0x1.c9f734p-4")),
          jnp.float32(float.fromhex("0x1.d1b0aap-6")),
          jnp.float32(float.fromhex("-0x1.5ee18ap-8")),
          jnp.float32(float.fromhex("0x1.7acedcp-11")),
          jnp.float32(float.fromhex("-0x1.14819ap-14")),
          jnp.float32(float.fromhex("0x1.e837a2p-19")),
          jnp.float32(float.fromhex("-0x1.8937eep-24")),
      ],
  )
  return erf32_1_5_following(x[0], poly)


def erf32_1_5_poly_sollya_13(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 13-th order Sollya-optimized polynomial."""
  poly = _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("0x1.e43c6ap-8")),
          jnp.float32(float.fromhex("0x1.b8a536p0")),
          jnp.float32(float.fromhex("-0x1.d60e42p0")),
          jnp.float32(float.fromhex("0x1.72e5f2p0")),
          jnp.float32(float.fromhex("-0x1.c6a94p-1")),
          jnp.float32(float.fromhex("0x1.b62412p-2")),
          jnp.float32(float.fromhex("-0x1.4b535ap-3")),
          jnp.float32(float.fromhex("0x1.85d1a4p-5")),
          jnp.float32(float.fromhex("-0x1.5f9f06p-7")),
          jnp.float32(float.fromhex("0x1.db89f6p-10")),
          jnp.float32(float.fromhex("-0x1.d170e8p-13")),
          jnp.float32(float.fromhex("0x1.36c188p-16")),
          jnp.float32(float.fromhex("-0x1.f9cd0ap-21")),
          jnp.float32(float.fromhex("0x1.7a2008p-26")),
      ],
  )
  return erf32_1_5_following(x[0], poly)


def erf32_1_5_poly_sollya_14(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 14-th order Sollya-optimized polynomial."""
  poly = _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("0x1.19e474p-8")),
          jnp.float32(float.fromhex("0x1.bda8a6p0")),
          jnp.float32(float.fromhex("-0x1.e46f98p0")),
          jnp.float32(float.fromhex("0x1.8b9fcp0")),
          jnp.float32(float.fromhex("-0x1.ff9f38p-1")),
          jnp.float32(float.fromhex("0x1.099672p-1")),
          jnp.float32(float.fromhex("-0x1.ba7eep-3")),
          jnp.float32(float.fromhex("0x1.25b65ap-4")),
          jnp.float32(float.fromhex("-0x1.332e3cp-6")),
          jnp.float32(float.fromhex("0x1.f20cb2p-9")),
          jnp.float32(float.fromhex("-0x1.31a46ep-11")),
          jnp.float32(float.fromhex("0x1.11d94p-14")),
          jnp.float32(float.fromhex("-0x1.51567cp-18")),
          jnp.float32(float.fromhex("0x1.fe047p-23")),
          jnp.float32(float.fromhex("-0x1.64585p-28")),
      ],
  )
  return erf32_1_5_following(x[0], poly)


def erf32_1_5_poly_sollya_15(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 15-th order Sollya-optimized polynomial."""
  poly = _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("0x1.56ccccp-9")),
          jnp.float32(float.fromhex("0x1.c096fap0")),
          jnp.float32(float.fromhex("-0x1.ed80eep0")),
          jnp.float32(float.fromhex("0x1.9c9456p0")),
          jnp.float32(float.fromhex("-0x1.153ee6p0")),
          jnp.float32(float.fromhex("0x1.3069c8p-1")),
          jnp.float32(float.fromhex("-0x1.11521cp-2")),
          jnp.float32(float.fromhex("0x1.8f1774p-4")),
          jnp.float32(float.fromhex("-0x1.d57f76p-6")),
          jnp.float32(float.fromhex("0x1.b7766ap-8")),
          jnp.float32(float.fromhex("-0x1.41be8cp-10")),
          jnp.float32(float.fromhex("0x1.67996ep-13")),
          jnp.float32(float.fromhex("-0x1.27a0cp-16")),
          jnp.float32(float.fromhex("0x1.50761cp-20")),
          jnp.float32(float.fromhex("-0x1.d8f6eep-25")),
          jnp.float32(float.fromhex("0x1.3503aep-30")),
      ],
  )
  return erf32_1_5_following(x[0], poly)


def erf32_1_5_poly_sollya_16(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 16-th order Sollya-optimized polynomial."""
  poly = _horner_scheme(
      x[0],
      [
          jnp.float32(float.fromhex("0x1.8c2ad2p-9")),
          jnp.float32(float.fromhex("0x1.bfce8ap0")),
          jnp.float32(float.fromhex("-0x1.ead124p0")),
          jnp.float32(float.fromhex("0x1.96fdap0")),
          jnp.float32(float.fromhex("-0x1.0d5be4p0")),
          jnp.float32(float.fromhex("0x1.20668ap-1")),
          jnp.float32(float.fromhex("-0x1.f23f5ap-3")),
          jnp.float32(float.fromhex("0x1.578d6p-4")),
          jnp.float32(float.fromhex("-0x1.739afap-6")),
          jnp.float32(float.fromhex("0x1.3271fap-8")),
          jnp.float32(float.fromhex("-0x1.6d8bap-11")),
          jnp.float32(float.fromhex("0x1.1506a6p-14")),
          jnp.float32(float.fromhex("-0x1.0748p-19")),
          jnp.float32(float.fromhex("-0x1.cafca8p-22")),
          jnp.float32(float.fromhex("0x1.24103p-24")),
          jnp.float32(float.fromhex("-0x1.2c5328p-28")),
          jnp.float32(float.fromhex("0x1.f00d3cp-34")),
      ],
  )
  return erf32_1_5_following(x[0], poly)


def erf32_1_5_poly_taylor_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 4-th order Taylor polynomial about x=1."""
  return _erf32_1_5_poly_taylor(order=4, x=x)


def erf32_1_5_poly_taylor_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 6-th order Taylor polynomial about x=1."""
  return _erf32_1_5_poly_taylor(order=6, x=x)


def erf32_1_5_poly_taylor_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 8-th order Taylor polynomial about x=1."""
  return _erf32_1_5_poly_taylor(order=8, x=x)


def erf32_1_5_poly_taylor_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 10-th order Taylor polynomial about x=1."""
  return _erf32_1_5_poly_taylor(order=10, x=x)


def erf32_1_5_poly_taylor_12(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 12-th order Taylor polynomial about x=1."""
  return _erf32_1_5_poly_taylor(order=12, x=x)


def erf32_1_5_poly_taylor_14(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 14-th order Taylor polynomial about x=1."""
  return _erf32_1_5_poly_taylor(order=14, x=x)


def erf32_1_5_poly_taylor_16(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 16-th order Taylor polynomial about x=1."""
  return _erf32_1_5_poly_taylor(order=16, x=x)


def erf32_1_5_poly_taylor_18(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 18-th order Taylor polynomial about x=1."""
  return _erf32_1_5_poly_taylor(order=18, x=x)


def erf32_1_5_poly_taylor_20(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 20-th order Taylor polynomial about x=1."""
  return _erf32_1_5_poly_taylor(order=20, x=x)


def _erf32_1_5_poly_taylor(order: int, x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the Taylor polynomial about x=1 of the given order."""
  coeffs = [
      # N[Normal[Series[Sqrt[Pi] x Exp[x^2] Erfc[x], {x, 1, 10}]], 100]
      # To use with expansion in (x-1).
      jnp.float32("7.5787215614131210604335123991421791634789653256165e-1"),
      jnp.float32("2.7361646842393631813005371974265374904368959768496e-1"),
      jnp.float32("-2.1063921929343946978324380042891041826051733719173e-1"),
      jnp.float32("1.3319032222831000494122451945671347020334470622381e-1"),
      jnp.float32("-7.3830985081471310718216940557583543738672538349252e-2"),
      jnp.float32("3.7062767081566478183325483505323317606203337772202e-2"),
      jnp.float32("-1.7178138338733031559511615054592311626734569415634e-2"),
      jnp.float32("7.4462161685031979964861282957004454039531403767433e-3"),
      jnp.float32("-3.0464854832264952321675007988155491137930204531381e-3"),
      jnp.float32("1.1845572680754672253065318964105448812565027712663e-3"),
      jnp.float32("-4.4008532043523882875369379823245749348048176866632e-4"),
      jnp.float32("1.5689594080868638528790750687620761380030386904119e-4"),
      jnp.float32("-5.3866189338292662225414287926533426817612707038526e-5"),
      jnp.float32("1.7862217416069372590228540182978177661443561706067e-5"),
      jnp.float32("-5.7353607068922574448662528416785897345913362237417e-6"),
      jnp.float32("1.7870305842337998535742910758206896060702354138916e-6"),
      jnp.float32("-5.4133593788974267761871399441289243052006540309578e-7"),
      jnp.float32("1.5969224210101996080046709101724591481524232048989e-7"),
      jnp.float32("-4.5942998472536136362934061125090739852954626192130e-8"),
      jnp.float32("1.2907477353296659653076383795766310316587256244734e-8"),
      jnp.float32("-3.5453573670425589150012101599066047423364981852246e-9"),
  ]
  if order >= len(coeffs):
    raise NotImplementedError("Missing coefficients.")
  coeffs = coeffs[: order + 1]
  poly = _horner_scheme(x[0] - jnp.float32("1.0"), coeffs)
  return (
      jnp.float32("1")
      - poly
      * jnp.exp(-jnp.multiply(x[0], x[0]))
      / jnp.sqrt(jnp.float32(jnp.pi))
      / x[0]
  )


def erf32_1_5_rational_minimax_2_2(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (2, 2)th order rational approximation with minimax coeffs.

  From: 2023-07-03.rational_minimax_erf15.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("0.10162881711712468201490060289655432243604182982759"),
          jnp.float32("1.2940247620656379193316338827974618575363203056251"),
          jnp.float32("1.2701256720277410392054859031101556265828705624103"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("1.2435574742734936596624537150230314880032014044391"),
          jnp.float32("1.2738465980126524365094365050198679103834866523659"),
      ],
  )
  rational = jnp.divide(numerator, denominator)
  return (
      jnp.float32("1")
      - rational
      * jnp.exp(-jnp.multiply(x[0], x[0]))
      / jnp.sqrt(jnp.float32(jnp.pi))
      / x[0]
  )


def erf32_1_5_rational_minimax_3_3(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (3, 3)th order rational approximation with minimax coeffs.

  From: 2023-07-03.rational_minimax_erf15.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("-0.012686678662885126877382149220362859692363127631561"),
          jnp.float32("1.8483761492508161205775441072413036147041236462015"),
          jnp.float32("1.6219797585053871613467939197319663722303492306115"),
          jnp.float32("0.71735130639374467980232844329982882066400937224480"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0"),
          jnp.float32("2.1649114582583774487355550498826596652738188517312"),
          jnp.float32("1.6268490140355457759453636323024984789633924164500"),
          jnp.float32("0.71711171435851824488469767006425367620193265038850"),
      ],
  )
  rational = jnp.divide(numerator, denominator)
  return (
      jnp.float32("1")
      - rational
      * jnp.exp(-jnp.multiply(x[0], x[0]))
      / jnp.sqrt(jnp.float32(jnp.pi))
      / x[0]
  )


def erf32_1_5_rational_minimax_4_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (4, 4)th order rational approximation with minimax coeffs.

  From: 2023-07-03.rational_minimax_erf15.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32(
              "0.00097590363524920692111520653288726787069915436070373"
          ),
          jnp.float32("1.7654092801438545517517016598716262653713015064715"),
          jnp.float32("2.0230153741574367825637043566039395940646746108841"),
          jnp.float32("1.0323556956263036670184810971338518810282711009240"),
          jnp.float32("0.26313592784611886891811186850912542802466641633758"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("2.2561039600442497864093742394575523758052482476962"),
          jnp.float32("2.1581145424265600692992842554831210366541401555264"),
          jnp.float32("1.0320677171819659579880590665864510923448251433746"),
          jnp.float32("0.26314641811977328995824985281653095119102805236423"),
      ],
  )
  rational = jnp.divide(numerator, denominator)
  return (
      jnp.float32("1")
      - rational
      * jnp.exp(-jnp.multiply(x[0], x[0]))
      / jnp.sqrt(jnp.float32(jnp.pi))
      / x[0]
  )


def erf32_1_5_rational_minimax_5_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (5, 5)th order rational approximation with minimax coeffs.

  From: 2023-07-03.rational_minimax_erf15.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32(
              "-0.000065476119261431308225834865684668165290916075470810"
          ),
          jnp.float32("1.7730052953658519701753258432070595925167769600998"),
          jnp.float32("2.3654830448265485884535442112434551198423551535778"),
          jnp.float32("1.5014811735418808273648673858225044540995193123464"),
          jnp.float32("0.51317982630480227705118654814756884112881310306101"),
          jnp.float32("0.087444755275383735928671440372817187235269713257130"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("2.4642105495619058912234895548215432906338496461264"),
          jnp.float32("2.6244682052097635898215235832261200869122399614785"),
          jnp.float32("1.5449579291097902596600295891235016501234373436065"),
          jnp.float32("0.51319514860876798794082826745797713890628732338014"),
          jnp.float32("0.087444316376086641664557366813770569151190936185593"),
      ],
  )
  rational = jnp.divide(numerator, denominator)
  return (
      jnp.float32("1")
      - rational
      * jnp.exp(-jnp.multiply(x[0], x[0]))
      / jnp.sqrt(jnp.float32(jnp.pi))
      / x[0]
  )


def erf32_1_5_rational_minimax_6_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (6, 6)th order rational approximation with minimax coeffs.

  From: 2023-07-03.rational_minimax_erf15.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32("3.8047225456881580945927467096635415815647406433590e-6"),
          jnp.float32("1.7724173181673858824215262076749109670153728059711"),
          jnp.float32("2.6855127116633402169344837468498055591598747424361"),
          jnp.float32("1.9719058833536308280059387645484672459085400129724"),
          jnp.float32("0.84306555742246984693996008827342714768426554031605"),
          jnp.float32("0.21031661160229684560382788483443315540357391341589"),
          jnp.float32("0.026016535016402820928513471306087855574752093761146"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1"),
          jnp.float32("2.6434226816591717922902822227766634036489854144296"),
          jnp.float32("3.0955911872142912607530668909831024780145429492269"),
          jnp.float32("2.0768833999946606980039413909589750302869741291030"),
          jnp.float32("0.85608842552468686280333380974974793818588258166044"),
          jnp.float32("0.21031587951434818631800370372782147058344497710605"),
          jnp.float32("0.026016552175129467690764729911622401572117114671037"),
      ],
  )
  rational = jnp.divide(numerator, denominator)
  return (
      jnp.float32("1")
      - rational
      * jnp.exp(-jnp.multiply(x[0], x[0]))
      / jnp.sqrt(jnp.float32(jnp.pi))
      / x[0]
  )


def erf32_1_5_rational_minimax_7_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (3, 3)th order rational approximation with minimax coeffs.

  From: 2023-07-03.rational_minimax_erf15.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  numerator = _horner_scheme(
      x[0],
      [
          jnp.float32(
              "-1.9614916475701777702999229266439266641336642720879e-7"
          ),
          jnp.float32("1.7724559620295653168304635335349762459829170170210"),
          jnp.float32("2.9833841461462608928042744875415515196718881501517"),
          jnp.float32("2.4614424230120754419331458991265536913135160399034"),
          jnp.float32("1.2283043183733677402273128373571143047740882790126"),
          jnp.float32("0.38870508843658483595565785375114755084909610021483"),
          jnp.float32("0.074598355617751205519339552423885767983983592100798"),
          jnp.float32("0.0070735143131332247580231806869741781730535067144148"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float32("1.0"),
          jnp.float32("2.8115795848124490880455485371425941791255635962543"),
          jnp.float32("3.5612275553798650713094914749433296277510467274720"),
          jnp.float32("2.6521338825989750649414470002237575192189661641153"),
          jnp.float32("1.2656151678083154322809260106735895878727229407347"),
          jnp.float32("0.39224107268056633543490163904184388564395459276573"),
          jnp.float32("0.074598387835152232661328277985162779227299725006330"),
          jnp.float32("0.0070735136775738188598738864389650534382839079333380"),
      ],
  )
  rational = jnp.divide(numerator, denominator)
  return (
      jnp.float32("1")
      - rational
      * jnp.exp(-jnp.multiply(x[0], x[0]))
      / jnp.sqrt(jnp.float32(jnp.pi))
      / x[0]
  )


def erf32_1_5_rational_pade_1_1(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (1, 1)th order Pade approximant about x=0.

  From: 2023-07-03.pade_approximants_erf15.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  y = x[0] - jnp.float32("1.0")
  numerator = _horner_scheme(
      y,
      [
          jnp.float32("0.757872156141312106043351239914217916347897"),
          jnp.float32("0.857052108147573428830261417180370858277673"),
      ],
  )
  denominator = _horner_scheme(
      y,
      [
          jnp.float32("1.0"),
          jnp.float32("0.769833849938735931158689645942120454720712"),
      ],
  )
  rational = jnp.divide(numerator, denominator)
  return (
      jnp.float32("1")
      - rational
      * jnp.exp(-jnp.multiply(x[0], x[0]))
      / jnp.sqrt(jnp.float32(jnp.pi))
      / x[0]
  )


def erf32_1_5_rational_pade_2_2(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (2, 2)th order Pade approximant about x=0.

  From: 2023-07-03.pade_approximants_erf15.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  y = x[0] - jnp.float32("1.0")
  numerator = _horner_scheme(
      y,
      [
          jnp.float32("0.7578721561413121060433512399142179163479"),
          jnp.float32("1.024595994706117234384306536684884297028"),
          jnp.float32("0.2697031827897539751997962307739449001369"),
      ],
  )
  denominator = _horner_scheme(
      y,
      [
          jnp.float32("1.0"),
          jnp.float32("0.9909052868570540323149059164765146258992"),
          jnp.float32("0.2760549985316373697515725891246932426284"),
      ],
  )
  rational = jnp.divide(numerator, denominator)
  return (
      jnp.float32("1")
      - rational
      * jnp.exp(-jnp.multiply(x[0], x[0]))
      / jnp.sqrt(jnp.float32(jnp.pi))
      / x[0]
  )


def erf32_1_5_rational_pade_3_3(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (3, 3)th order Pade approximant about x=0.

  From: 2023-07-03.pade_approximants_erf15.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  y = x[0] - jnp.float32("1.0")
  numerator = _horner_scheme(
      y,
      [
          jnp.float32("0.7578721561413121060433512399142179163479"),
          jnp.float32("1.215281071796266433877881634242730417521"),
          jnp.float32("0.5450560066818093684605981980620844244132"),
          jnp.float32("0.08771517931402782125006589137209315254441"),
      ],
  )
  denominator = _horner_scheme(
      y,
      [
          jnp.float32("1.0"),
          jnp.float32("1.242511148802184320554781632397756776717"),
          jnp.float32("0.5485406873624628747124262824524340440191"),
          jnp.float32("0.08729265118629502005701518643226429887103"),
      ],
  )
  rational = jnp.divide(numerator, denominator)
  return (
      jnp.float32("1")
      - rational
      * jnp.exp(-jnp.multiply(x[0], x[0]))
      / jnp.sqrt(jnp.float32(jnp.pi))
      / x[0]
  )


def erf32_1_5_rational_pade_4_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (4, 4)th order Pade approximant about x=0.

  From: 2023-07-03.pade_approximants_erf15.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  y = x[0] - jnp.float32("1.0")
  numerator = _horner_scheme(
      y,
      [
          jnp.float32("0.7578721561413121060433512399142179163479"),
          jnp.float32("1.379694531447414270551224592419285228276"),
          jnp.float32("0.8225665774219249295779348709896822828057"),
          jnp.float32("0.2247066431743771893530957316343217078682"),
          jnp.float32("0.02396363550084764395670581221579926882342"),
      ],
  )
  denominator = _horner_scheme(
      y,
      [
          jnp.float32("1.0"),
          jnp.float32("1.459452038263350199165933228932249056450"),
          jnp.float32("0.8363886692961432615298489517597662794003"),
          jnp.float32("0.2244236624623508933244581950348062837288"),
          jnp.float32("0.02398818792146530030932498103556056957386"),
      ],
  )
  rational = jnp.divide(numerator, denominator)
  return (
      jnp.float32("1")
      - rational
      * jnp.exp(-jnp.multiply(x[0], x[0]))
      / jnp.sqrt(jnp.float32(jnp.pi))
      / x[0]
  )


def erf32_1_5_rational_pade_5_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (5, 5)th order Pade approximant about x=0.

  From: 2023-07-03.pade_approximants_erf15.nb.

  Args:
    x: ...

  Returns:
    ...
  """
  y = x[0] - jnp.float32("1.0")
  numerator = _horner_scheme(
      y,
      [
          jnp.float32("0.7578721561413121060433512399142179163479"),
          jnp.float32("1.527252531583988005562371901793055532917"),
          jnp.float32("1.102090777491842685291547355207100754578"),
          jnp.float32("0.4023347185808178663142216191751178181939"),
          jnp.float32("0.07555125036669256945869600193463546041309"),
          jnp.float32("0.005926950965712886500948146627301120767421"),
      ],
  )
  denominator = _horner_scheme(
      y,
      [
          jnp.float32("1.0"),
          jnp.float32("1.654152422676285688568324360779164555905"),
          jnp.float32("1.134923147245589122527233613379611624273"),
          jnp.float32("0.4051344349933138506227513522392758620704"),
          jnp.float32("0.07557081118119012894463996846570573618559"),
          jnp.float32("0.005925655966366904240916954270753929041903"),
      ],
  )
  rational = jnp.divide(numerator, denominator)
  return (
      jnp.float32("1")
      - rational
      * jnp.exp(-jnp.multiply(x[0], x[0]))
      / jnp.sqrt(jnp.float32(jnp.pi))
      / x[0]
  )


################################################################################
# Baselines in graph form. #####################################################
################################################################################

# These are useful for benchmarking.


def exp32_poly_sollya_8_graph():
  """Returns the 8-th order Sollya-optimized polynomial as a graph."""
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("c0", "RandomInitVariableOp"),
          ("c1", "RandomInitVariableOp"),
          ("c2", "RandomInitVariableOp"),
          ("c3", "RandomInitVariableOp"),
          ("c4", "RandomInitVariableOp"),
          ("c5", "RandomInitVariableOp"),
          ("c6", "RandomInitVariableOp"),
          ("c7", "RandomInitVariableOp"),
          ("c8", "RandomInitVariableOp"),
          ("x7", "MultOp"),
          ("x6", "MultOp"),
          ("x5", "MultOp"),
          ("x4", "MultOp"),
          ("x3", "MultOp"),
          ("x2", "MultOp"),
          ("x1", "MultOp"),
          ("x0", "MultOp"),
          ("y7", "AddOp"),
          ("y6", "AddOp"),
          ("y5", "AddOp"),
          ("y4", "AddOp"),
          ("y3", "AddOp"),
          ("y2", "AddOp"),
          ("y1", "AddOp"),
          ("y0", "AddOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[
          (["c8", "x"], "x7"),
          (["x7", "c7"], "y7"),
          (["y7", "x"], "x6"),
          (["x6", "c6"], "y6"),
          (["y6", "x"], "x5"),
          (["x5", "c5"], "y5"),
          (["y5", "x"], "x4"),
          (["x4", "c4"], "y4"),
          (["y4", "x"], "x3"),
          (["x3", "c3"], "y3"),
          (["y3", "x"], "x2"),
          (["x2", "c2"], "y2"),
          (["y2", "x"], "x1"),
          (["x1", "c1"], "y1"),
          (["y1", "x"], "x0"),
          (["x0", "c0"], "y0"),
          (["y0"], "f"),
      ],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
  )
  learnable_params = {
      "c0": JnpPreciseFloat("1.0"),
      "c1": JnpPreciseFloat("0.693147182464599609375"),
      "c2": JnpPreciseFloat("0.24022646248340606689453125"),
      "c3": JnpPreciseFloat("5.5504463613033294677734375e-2"),
      "c4": JnpPreciseFloat("9.616785682737827301025390625e-3"),
      "c5": JnpPreciseFloat("1.336037297733128070831298828125e-3"),
      "c6": JnpPreciseFloat("1.5121899195946753025054931640625e-4"),
      "c7": JnpPreciseFloat("1.657833854551427066326141357421875e-5"),
      "c8": JnpPreciseFloat("1.27110934045049361884593963623046875e-6"),
  }
  return graph, learnable_params


def exp32_rational_minimax_4_4_graph():
  """Returns the (4, 4)th order rational minimax approximation as a graph."""
  graph = graph_lib.Graph()
  graph.parse(
      desired_vertices=[
          ("x", "ProduceXOp"),
          ("cn0", "RandomInitVariableOp"),
          ("cn1", "RandomInitVariableOp"),
          ("cn2", "RandomInitVariableOp"),
          ("cn3", "RandomInitVariableOp"),
          ("cn4", "RandomInitVariableOp"),
          ("cd0", "RandomInitVariableOp"),
          ("cd1", "RandomInitVariableOp"),
          ("cd2", "RandomInitVariableOp"),
          ("cd3", "RandomInitVariableOp"),
          ("cd4", "RandomInitVariableOp"),
          ("xn3", "MultOp"),
          ("xn2", "MultOp"),
          ("xn1", "MultOp"),
          ("xn0", "MultOp"),
          ("yn3", "AddOp"),
          ("yn2", "AddOp"),
          ("yn1", "AddOp"),
          ("yn0", "AddOp"),
          ("xd3", "MultOp"),
          ("xd2", "MultOp"),
          ("xd1", "MultOp"),
          ("xd0", "MultOp"),
          ("yd3", "AddOp"),
          ("yd2", "AddOp"),
          ("yd1", "AddOp"),
          ("yd0", "AddOp"),
          ("r", "DivOp"),
          ("f", "ConsumeFOp"),
      ],
      desired_edges=[
          # Numerator.
          (["cn4", "x"], "xn3"),
          (["xn3", "cn3"], "yn3"),
          (["yn3", "x"], "xn2"),
          (["xn2", "cn2"], "yn2"),
          (["yn2", "x"], "xn1"),
          (["xn1", "cn1"], "yn1"),
          (["yn1", "x"], "xn0"),
          (["xn0", "cn0"], "yn0"),
          # Denominator.
          (["cd4", "x"], "xd3"),
          (["xd3", "cd3"], "yd3"),
          (["yd3", "x"], "xd2"),
          (["xd2", "cd2"], "yd2"),
          (["yd2", "x"], "xd1"),
          (["xd1", "cd1"], "yd1"),
          (["yd1", "x"], "xd0"),
          (["xd0", "cd0"], "yd0"),
          # Ratio.
          (["yn0", "yd0"], "r"),
          (["r"], "f"),
      ],
      required_input_vertex_ids=[
          "x",
      ],
      required_output_vertex_ids=["f"],
      op_init_params=None,
  )
  learnable_params = {
      "cn0": JnpPreciseFloat("1.0000000000000110932315"),
      "cn1": JnpPreciseFloat("3.5514943227000784164247e-1"),
      "cn2": JnpPreciseFloat("5.4504397443498413634232e-2"),
      "cn3": JnpPreciseFloat("4.3961581146562074591197e-3"),
      "cn4": JnpPreciseFloat("1.6357454492149283435106e-4"),
      "cd0": JnpPreciseFloat("1"),
      "cd1": JnpPreciseFloat("-3.3799774828814290230468e-1"),
      "cd2": JnpPreciseFloat("4.8560076698570416362649e-2"),
      "cd3": JnpPreciseFloat("-3.5712118938156366801419e-3"),
      "cd4": JnpPreciseFloat("1.1566466994349112222271e-4"),
  }
  return graph, learnable_params


################################################################################
# 64-bit baselines for f=2^x on [0, 1]. ########################################
################################################################################


def table_exp2_64bits_baselines() -> List[Tuple[str, FinalizedFn]]:
  """Baselines for 64-bit 2^x on [0, 1].

  Returns:
    A list of (name, finalized_fn).
  """
  return [  # pytype: disable=bad-return-type  # jax-ndarray
      ("JAX Numpy", exp64_jnp),
      ("Polynomial, Sollya, 5th order", exp64_poly_sollya_5),
      ("Polynomial, Sollya, 6th order", exp64_poly_sollya_6),
      ("Polynomial, Sollya, 7th order", exp64_poly_sollya_7),
      ("Polynomial, Sollya, 8th order", exp64_poly_sollya_8),
      ("Polynomial, Sollya, 9th order", exp64_poly_sollya_9),
      ("Polynomial, Sollya, 10th order", exp64_poly_sollya_10),
      ("Polynomial, Sollya, 11th order", exp64_poly_sollya_11),
      ("Polynomial, Sollya, 12th order", exp64_poly_sollya_12),
      ("Polynomial, Sollya, 13th order", exp64_poly_sollya_13),
      ("Polynomial, Sollya, 14th order", exp64_poly_sollya_14),
      ("Polynomial, Sollya, 15th order", exp64_poly_sollya_15),
      ("Polynomial, Sollya, 16th order", exp64_poly_sollya_16),
      ("Polynomial, Sollya, 17th order", exp64_poly_sollya_17),
      ("Polynomial, Sollya, 18th order", exp64_poly_sollya_18),
      ("Polynomial, Sollya, 19th order", exp64_poly_sollya_19),
      ("Polynomial, Sollya, 20th order", exp64_poly_sollya_20),
      ("Polynomial, Sollya, 21th order", exp64_poly_sollya_21),
      ("Polynomial, Sollya, 22th order", exp64_poly_sollya_22),
      ("Polynomial, Sollya, 23th order", exp64_poly_sollya_23),
      ("Rational, Minimax, (1,1)-order", exp64_rational_minimax_1_1),
      ("Rational, Minimax, (2,2)-order", exp64_rational_minimax_2_2),
      ("Rational, Minimax, (3,3)-order", exp64_rational_minimax_3_3),
      ("Rational, Minimax, (4,4)-order", exp64_rational_minimax_4_4),
      ("Rational, Minimax, (5,5)-order", exp64_rational_minimax_5_5),
      ("Rational, Minimax, (6,6)-order", exp64_rational_minimax_6_6),
      ("Rational, Minimax, (7,7)-order", exp64_rational_minimax_7_7),
      ("Rational, Minimax, (8,8)-order", exp64_rational_minimax_8_8),
      ("Rational, Minimax, (9,9)-order", exp64_rational_minimax_9_9),
      ("Rational, Minimax, (10,10)-order", exp64_rational_minimax_10_10),
      ("Rational, Minimax, (11,11)-order", exp64_rational_minimax_11_11),
      ("Taylor expansion, 10th order", exp64_poly_taylor_10),
      ("Taylor expansion, 11th order", exp64_poly_taylor_11),
      ("Taylor expansion, 12th order", exp64_poly_taylor_12),
      ("Taylor expansion, 13th order", exp64_poly_taylor_13),
      ("Taylor expansion, 14th order", exp64_poly_taylor_14),
      ("Taylor expansion, 15th order", exp64_poly_taylor_15),
      ("Taylor expansion, 16th order", exp64_poly_taylor_16),
      ("Taylor expansion, 17th order", exp64_poly_taylor_17),
      ("Taylor expansion, 18th order", exp64_poly_taylor_18),
      ("Taylor expansion, 19th order", exp64_poly_taylor_19),
      ("Taylor expansion, 20th order", exp64_poly_taylor_20),
      ("Taylor expansion, 21th order", exp64_poly_taylor_21),
      ("Taylor expansion, 22th order", exp64_poly_taylor_22),
      ("Taylor expansion, 23th order", exp64_poly_taylor_23),
      ("Pade Approximant, (1,1)-order", exp64_rational_pade_1_1),
      ("Pade Approximant, (2,2)-order", exp64_rational_pade_2_2),
      ("Pade Approximant, (3,3)-order", exp64_rational_pade_3_3),
      ("Pade Approximant, (4,4)-order", exp64_rational_pade_4_4),
      ("Pade Approximant, (5,5)-order", exp64_rational_pade_5_5),
      ("Pade Approximant, (6,6)-order", exp64_rational_pade_6_6),
      ("Pade Approximant, (7,7)-order", exp64_rational_pade_7_7),
      ("Pade Approximant, (8,8)-order", exp64_rational_pade_8_8),
      ("Pade Approximant, (9,9)-order", exp64_rational_pade_9_9),
      ("Pade Approximant, (10,10)-order", exp64_rational_pade_10_10),
      ("Pade Approximant, (11,11)-order", exp64_rational_pade_11_11),
      ("Pade Approximant, (12,12)-order", exp64_rational_pade_12_12),
      ("Pade Approximant, (13,13)-order", exp64_rational_pade_13_13),
      ("Pade Approximant, (14,14)-order", exp64_rational_pade_14_14),
      ("Pade Approximant, (15,15)-order", exp64_rational_pade_15_15),
      ("Pade Approximant, (16,16)-order", exp64_rational_pade_16_16),
      ("Pade Approximant, (17,17)-order", exp64_rational_pade_17_17),
  ]


def exp64_jnp(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns JAX NumPy approximation."""
  return jnp.exp2(x[0])


def exp64_poly_sollya_5_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 5-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(
          float.fromhex("0x1.fffffd7c4cf55p-1")
      ),  # 0.99999992506352930465851613917038775980472564697266
      jnp.float64(
          float.fromhex("0x1.62e4f5a908809p-1")
      ),  # 0.69315307320016572578680325023015029728412628173828
      jnp.float64(
          float.fromhex("0x1.ebd5a8d9be26bp-3")
      ),  # 0.240153617043478234682751804029976483434438705444336
      jnp.float64(
          float.fromhex("0x1.c9544649f36edp-5")
      ),  # 5.5826318057013192552862079764963709749281406402588e-2
      jnp.float64(
          float.fromhex("0x1.26900cd70420ap-7")
      ),  # 8.9893400854928091903150999542049248702824115753174e-3
      jnp.float64(
          float.fromhex("0x1.ec3209b3473c1p-10")
      ),  # 1.87757667737929755981396429120877655805088579654694e-3
  ]


def exp64_poly_sollya_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 5-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_5_coefficients())


def exp64_poly_sollya_6_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 6-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(
          float.fromhex("0x1.00000007f8794p0")
      ),  # 1.00000000185579995815032816608436405658721923828125
      jnp.float64(
          float.fromhex("0x1.62e42955d541bp-1")
      ),  # 0.69314698384062312097597668980597518384456634521484
      jnp.float64(
          float.fromhex("0x1.ebfd9ed290159p-3")
      ),  # 0.240229836274020752684421609046694356948137283325195
      jnp.float64(
          float.fromhex("0x1.c68500696e562p-5")
      ),  # 5.5483341984152514947403744827170157805085182189941e-2
      jnp.float64(
          float.fromhex("0x1.3d2800c5d0061p-7")
      ),  # 9.6788409970260409070919038754254870582371950149536e-3
      jnp.float64(
          float.fromhex("0x1.461954f13941p-10")
      ),  # 1.24396878191502038846216748879669466987252235412598e-3
      jnp.float64(
          float.fromhex("0x1.c72118d1f46fap-13")
      ),  # 2.17022554862604359366545780929413922422099858522415e-4
  ]


def exp64_poly_sollya_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 6-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_6_coefficients())


def exp64_poly_sollya_7_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 7-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(
          float.fromhex("0x1.ffffffffa7933p-1")
      ),  # 0.9999999999597889432934039177780505269765853881836
      jnp.float64(
          float.fromhex("0x1.62e4301f16e74p-1")
      ),  # 0.69314718608388981024859276658389717340469360351562
      jnp.float64(
          float.fromhex("0x1.ebfbcf8c908e8p-3")
      ),  # 0.24022638461798995201945672306464985013008117675781
      jnp.float64(
          float.fromhex("0x1.c6b2b013ea99cp-5")
      ),  # 5.5505126859617365253640741684648673981428146362305e-2
      jnp.float64(
          float.fromhex("0x1.3b083852fe42bp-7")
      ),  # 9.6140170116872560729381902433487994130700826644897e-3
      jnp.float64(
          float.fromhex("0x1.5fddc70b233a4p-10")
      ),  # 1.342263482715135881895296421362218097783625125885e-3
      jnp.float64(
          float.fromhex("0x1.2cfd662d4818bp-13")
      ),  # 1.435231401183508154802831446872346532472874969244e-4
      jnp.float64(
          float.fromhex("0x1.68b07cbfcc3a8p-16")
      ),  # 2.1498763771169278552833348250317158090183511376381e-5
  ]


def exp64_poly_sollya_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 7-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_7_coefficients())


def exp64_poly_sollya_8_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 8-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(
          float.fromhex("0x1.0000000000dap0")
      ),  # 1.00000000000077449158197850920259952545166015625
      jnp.float64(
          float.fromhex("0x1.62e42fee7d6cfp-1")
      ),  # 0.69314718042616074722417351949843578040599822998047
      jnp.float64(
          float.fromhex("0x1.ebfbe0790f81ep-3")
      ),  # 0.240226510710216112354231654535396955907344818115234
      jnp.float64(
          float.fromhex("0x1.c6b077f098021p-5")
      ),  # 5.550406862016666470482917361550789792090654373169e-2
      jnp.float64(
          float.fromhex("0x1.3b2c7e817594cp-7")
      ),  # 9.618341226922531850274111775433993898332118988037e-3
      jnp.float64(
          float.fromhex("0x1.5d5e0531e26fep-10")
      ),  # 1.3327303572360778109728851603676957893185317516327e-3
      jnp.float64(
          float.fromhex("0x1.4548af5f42e0cp-13")
      ),  # 1.55107462866830073631996267380372955813072621822357e-4
      jnp.float64(
          float.fromhex("0x1.dc6691aa1a0c9p-17")
      ),  # 1.419784605989660908516043075211499058241315651685e-5
      jnp.float64(
          float.fromhex("0x1.f4304d85b58ap-20")
      ),  # 1.8633480477945634141458131072255355320521630346775e-6
  ]


def exp64_poly_sollya_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 8-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_8_coefficients())


def exp64_poly_sollya_9_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 9-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(
          float.fromhex("0x1.fffffffffff87p-1")
      ),  # 0.99999999999998656630140203560586087405681610107422
      jnp.float64(
          float.fromhex("0x1.62e42fefa9e24p-1")
      ),  # 0.69314718056279334135183489706832915544509887695312
      jnp.float64(
          float.fromhex("0x1.ebfbdff4c9b74p-3")
      ),  # 0.240226506860576116686445402592653408646583557128906
      jnp.float64(
          float.fromhex("0x1.c6b08e2452432p-5")
      ),  # 5.5504109974727447451847694992466131225228309631348e-2
      jnp.float64(
          float.fromhex("0x1.3b2aa423a922bp-7")
      ),  # 9.6181203328536604008069943461123330052942037582397e-3
      jnp.float64(
          float.fromhex("0x1.5d8a3e52c36e8p-10")
      ),  # 1.33338933364107216361649221880725235678255558013916e-3
      jnp.float64(
          float.fromhex("0x1.42df89694c4f5p-13")
      ),  # 1.53957934608629478040811577876922910945722833275795e-4
      jnp.float64(
          float.fromhex("0x1.01bc6178ad7b1p-16")
      ),  # 1.53622545180015600675780401696002286371367517858744e-5
      jnp.float64(
          float.fromhex("0x1.49f5e28b2e332p-20")
      ),  # 1.2291986049418540906310193944195319204482075292617e-6
      jnp.float64(
          float.fromhex("0x1.3444299a67df6p-23")
      ),  # 1.4354766337836700146343249344382808629916326026432e-7
  ]


def exp64_poly_sollya_9(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 9-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_9_coefficients())


def exp64_poly_sollya_10_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 10-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(
          float.fromhex("0x1.0000000000001p0")
      ),  # 1.00000000000000022204460492503130808472633361816406
      jnp.float64(
          float.fromhex("0x1.62e42fefa3801p-1")
      ),  # 0.69314718055989044120934750026208348572254180908203
      jnp.float64(
          float.fromhex("0x1.ebfbdff84064ep-3")
      ),  # 0.24022650696137975989685742206347640603780746459961
      jnp.float64(
          float.fromhex("0x1.c6b08d6b3c355p-5")
      ),  # 5.5504108628048477724892251217170269228518009185791e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab7a2083ap-7")
      ),  # 9.618129410286446745459443263825960457324981689453e-3
      jnp.float64(
          float.fromhex("0x1.5d87e5a422438p-10")
      ),  # 1.33335436933835864425912376418636995367705821990967e-3
      jnp.float64(
          float.fromhex("0x1.430b5f0effbaep-13")
      ),  # 1.5403958283851992621879545275476175447693094611168e-4
      jnp.float64(
          float.fromhex("0x1.ff8689ef50997p-17")
      ),  # 1.52446491187327404589644738730491724254534346982837e-5
      jnp.float64(
          float.fromhex("0x1.655ce735731f7p-20")
      ),  # 1.3312805554498870926185888005632840247471904149279e-6
      jnp.float64(
          float.fromhex("0x1.9653ee727aa61p-24")
      ),  # 9.4605576518290385344116136343700640409792868013028e-8
      jnp.float64(
          float.fromhex("0x1.55fb387340127p-27")
      ),  # 9.9529666509374388019884200072377733459205728649977e-9
  ]


def exp64_poly_sollya_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 10-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_10_coefficients())


def exp64_poly_sollya_11_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 11-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39f5p-1")
      ),  # 0.69314718055994595236057875808910466730594635009766
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c02ap-3")
      ),  # 0.24022650695906239137755733281665015965700149536133
      jnp.float64(
          float.fromhex("0x1.c6b08d7065f8fp-5")
      ),  # 5.5504108665615274620375174663422512821853160858154e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6f7267e9p-7")
      ),  # 9.618129099454324551499162510026508243754506111145e-3
      jnp.float64(
          float.fromhex("0x1.5d87ff4efe8f2p-10")
      ),  # 1.33335586337906793555352358282561908708885312080383e-3
      jnp.float64(
          float.fromhex("0x1.4308f9f4c254fp-13")
      ),  # 1.54035121930721417794415972757349209132371470332146e-4
      jnp.float64(
          float.fromhex("0x1.ffcfcb3fe1797p-17")
      ),  # 1.52531771389177337269358811222552674280450446531177e-5
      jnp.float64(
          float.fromhex("0x1.628f0b19e31d5p-20")
      ),  # 1.32083432055888705908372905001302299865528766531497e-6
      jnp.float64(
          float.fromhex("0x1.b8612c837df9fp-24")
      ),  # 1.02533862337995870705910481015393775905408801918384e-7
      jnp.float64(
          float.fromhex("0x1.c2b7f006714cbp-28")
      ),  # 6.5588174947098374579243424450491345423230882261123e-9
      jnp.float64(
          float.fromhex("0x1.58683ad3a9de7p-31")
      ),  # 6.264729520369589538179779228995857492945376066018e-10
  ]


def exp64_poly_sollya_11(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 11-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_11_coefficients())


def exp64_poly_sollya_12_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 12-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39efp-1")
      ),  # 0.69314718055994528622676398299518041312694549560547
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c5bep-3")
      ),  # 0.240226506959102026339536450905143283307552337646484
      jnp.float64(
          float.fromhex("0x1.c6b08d704911cp-5")
      ),  # 5.5504108664793799787773309617477934807538986206055e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6fbcfdf8p-7")
      ),  # 9.618129107933789945228397755272453650832176208496e-3
      jnp.float64(
          float.fromhex("0x1.5d87fe6fd142p-10")
      ),  # 1.33335581263450270173009926111262757331132888793945e-3
      jnp.float64(
          float.fromhex("0x1.430914243d38p-13")
      ),  # 1.54035312455082923355220714256574865430593490600586e-4
      jnp.float64(
          float.fromhex("0x1.ffcbc6e696635p-17")
      ),  # 1.5252709500024246993079603262888355175164178945124e-5
      jnp.float64(
          float.fromhex("0x1.62c36603f47cap-20")
      ),  # 1.3215961880319652414132822887404827838508936110884e-6
      jnp.float64(
          float.fromhex("0x1.b4df31345242ep-24")
      ),  # 1.017171527606571386703548194249679959000332019059e-7
      jnp.float64(
          float.fromhex("0x1.e8c1fc0f10bccp-28")
      ),  # 7.1123613795907350003003510830953959320765989104984e-9
      jnp.float64(
          float.fromhex("0x1.c46ad61661487p-32")
      ),  # 4.1147116391113014653044845589933194845988495558231e-10
      jnp.float64(
          float.fromhex("0x1.40b9775e3c329p-35")
      ),  # 3.6462151645395223491938090935309967240141038047341e-11
  ]


def exp64_poly_sollya_12(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 12-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_12_coefficients())


def exp64_poly_sollya_13_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 13-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39efp-1")
      ),  # 0.69314718055994528622676398299518041312694549560547
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c5bbp-3")
      ),  # 0.24022650695910194307280960401840275153517723083496
      jnp.float64(
          float.fromhex("0x1.c6b08d70492dp-5")
      ),  # 5.5504108664796825145515413169050589203834533691406e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6fbc9b01p-7")
      ),  # 9.6181291078898407259645253475355275440961122512817e-3
      jnp.float64(
          float.fromhex("0x1.5d87fe7156983p-10")
      ),  # 1.33335581298030312995261947861536100390367209911346e-3
      jnp.float64(
          float.fromhex("0x1.430913e98c7dap-13")
      ),  # 1.54035310787003023886426733568555391684640198946e-4
      jnp.float64(
          float.fromhex("0x1.ffcbd27315cdfp-17")
      ),  # 1.52527147518191938624322265272148513304273365065455e-5
      jnp.float64(
          float.fromhex("0x1.62c2a23a84c44p-20")
      ),  # 1.32158505883741995420377693393199081128841498866677e-6
      jnp.float64(
          float.fromhex("0x1.b4f0ca8bd8e9p-24")
      ),  # 1.01733158950447007099642839078601497249110252596438e-7
      jnp.float64(
          float.fromhex("0x1.e7b2d582363c6p-28")
      ),  # 7.0969482532135137997205274660271479003625927362009e-9
      jnp.float64(
          float.fromhex("0x1.cee133ff0d1f6p-32")
      ),  # 4.2098663439035766622362444268150814818962857088991e-10
      jnp.float64(
          float.fromhex("0x1.22c69e521943cp-35")
      ),  # 3.3057387245450528432618637749644107531876713323982e-11
      jnp.float64(
          float.fromhex("0x1.2e41db0c2f4d5p-41")
      ),  # 5.3691673089963703867075784627791456042004286652869e-13
  ]


def exp64_poly_sollya_13(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 13-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_13_coefficients())


def exp64_poly_sollya_14_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 14-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39efp-1")
      ),  # 0.69314718055994528622676398299518041312694549560547
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c5c2p-3")
      ),  # 0.240226506959102137361838913420797325670719146728516
      jnp.float64(
          float.fromhex("0x1.c6b08d7048dffp-5")
      ),  # 5.5504108664788269489331895556460949592292308807373e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6fbdec2dp-7")
      ),  # 9.6181291080395751175169394286967872176319360733032e-3
      jnp.float64(
          float.fromhex("0x1.5d87fe6b18a41p-10")
      ),  # 1.3333558115610352149543738065062825626228004693985e-3
      jnp.float64(
          float.fromhex("0x1.4309150d20b82p-13")
      ),  # 1.54035319074177139229281374710467389377299696207047e-4
      jnp.float64(
          float.fromhex("0x1.ffcb8c5208082p-17")
      ),  # 1.5252682860789288770428720476424189200770342722535e-5
      jnp.float64(
          float.fromhex("0x1.62c8665e80792p-20")
      ),  # 1.3216689677135403037822481558949405666680831927806e-6
      jnp.float64(
          float.fromhex("0x1.b44809ceac922p-24")
      ),  # 1.01579679094217288506271060395669270093321756576188e-7
      jnp.float64(
          float.fromhex("0x1.f51b1bb61fbccp-28")
      ),  # 7.2920504548927340321726420181366878914275275747059e-9
      jnp.float64(
          float.fromhex("0x1.14f8a93dc292ap-32")
      ),  # 2.5190395937582924791816029811406842109366976956153e-10
      jnp.float64(
          float.fromhex("0x1.1a3a91d696498p-33")
      ),  # 1.2834279360267805206706475681195309124227321717626e-10
      jnp.float64(
          float.fromhex("-0x1.1028f601d2aaap-35")
      ),  # -3.094101018300121239419618779617243871371545793636e-11
      jnp.float64(
          float.fromhex("0x1.4581a61e91b0cp-38")
      ),  # 4.6257247423546475997123396163240405080940842452719e-12
  ]


def exp64_poly_sollya_14(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 14-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_14_coefficients())


def exp64_poly_sollya_15_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 15-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39efp-1")
      ),  # 0.69314718055994528622676398299518041312694549560547
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c5cap-3")
      ),  # 0.24022650695910235940644383845210541039705276489258
      jnp.float64(
          float.fromhex("0x1.c6b08d70487bcp-5")
      ),  # 5.5504108664777146442403932269371580332517623901367e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6fbfe235p-7")
      ),  # 9.618129108262521778649478676470607751980423927307e-3
      jnp.float64(
          float.fromhex("0x1.5d87fe605fbcap-10")
      ),  # 1.333355809123070671168287404384500405285507440567e-3
      jnp.float64(
          float.fromhex("0x1.43091754059c9p-13")
      ),  # 1.5403533564102418165937813387955657162819989025593e-4
      jnp.float64(
          float.fromhex("0x1.ffcae75e2f564p-17")
      ),  # 1.5252607849065409021272615719411192003462929278612e-5
      jnp.float64(
          float.fromhex("0x1.62d89631071c5p-20")
      ),  # 1.3219045167440270392911942989866425079981127055362e-6
      jnp.float64(
          float.fromhex("0x1.b207bdd9ecd4p-24")
      ),  # 1.01055540295895555414924099624229825167276430875063e-7
      jnp.float64(
          float.fromhex("0x1.172857a21bb1p-27")
      ),  # 8.1245550879116722805818292044155448650144535349682e-9
      jnp.float64(
          float.fromhex("-0x1.790a7e3c2e52dp-31")
      ),  # -6.858335631377232651384585070434510983838904962795e-10
      jnp.float64(
          float.fromhex("0x1.d8cdd13875525p-31")
      ),  # 8.6002541811847589635729933568220918760349746889915e-10
      jnp.float64(
          float.fromhex("-0x1.bf7901b376424p-32")
      ),  # -4.0697403365407330717977942886745892914213129643031e-10
      jnp.float64(
          float.fromhex("0x1.05ed331ad0f2dp-33")
      ),  # 1.19110409762748787317120955754157305556129031742785e-10
      jnp.float64(
          float.fromhex("-0x1.13280cc9c40bbp-36")
      ),  # -1.56408330628691934253425326553727718430220061662794e-11
  ]


def exp64_poly_sollya_15(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 15-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_15_coefficients())


def exp64_poly_sollya_16_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 16-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39efp-1")
      ),  # 0.69314718055994528622676398299518041312694549560547
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c5d1p-3")
      ),  # 0.240226506959102553695473147854499984532594680786133
      jnp.float64(
          float.fromhex("0x1.c6b08d7048143p-5")
      ),  # 5.5504108664765648695205157991949818097054958343506e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6fc24229p-7")
      ),  # 9.6181291085325072015566050254165020305663347244263e-3
      jnp.float64(
          float.fromhex("0x1.5d87fe5132e14p-10")
      ),  # 1.3333558056726245791279161778675188543274998664856e-3
      jnp.float64(
          float.fromhex("0x1.43091b19b6fdap-13")
      ),  # 1.54035363087666823684457328980101920024026185274124e-4
      jnp.float64(
          float.fromhex("0x1.ffc9a5e2c22eep-17")
      ),  # 1.52524616559164765230201948886801233129517640918493e-5
      jnp.float64(
          float.fromhex("0x1.62fe025fce83p-20")
      ),  # 1.3224490870837183112678631102809845287993084639311e-6
      jnp.float64(
          float.fromhex("0x1.abc78b77df761p-24")
      ),  # 9.9600169776627101591910554232583985267979187483434e-8
      jnp.float64(
          float.fromhex("0x1.7842f71496138p-27")
      ),  # 1.09506533087677573989662783114640287607244317769073e-8
      jnp.float64(
          float.fromhex("-0x1.418a86ed43fd6p-28")
      ),  # -4.6790391398807010134980243431200574999451191615663e-9
      jnp.float64(
          float.fromhex("0x1.524622a08dc42p-28")
      ),  # 4.9225340752830202759080687716260193997896976725315e-9
      jnp.float64(
          float.fromhex("-0x1.c64188c4de2f4p-29")
      ),  # -3.3051473523632321484356385806781253311470436528907e-9
      jnp.float64(
          float.fromhex("0x1.9ad988ea72ff8p-30")
      ),  # 1.49466266673543740294644656700293050555217178043677e-9
      jnp.float64(
          float.fromhex("-0x1.be063e2f93807p-32")
      ),  # -4.0565681627475268877445812078783634463396978730998e-10
      jnp.float64(
          float.fromhex("0x1.b793e14ccc22p-35")
      ),  # 4.9974193706818873528725041962498393255387441058701e-11
  ]


def exp64_poly_sollya_16(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 16-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_16_coefficients())


def exp64_poly_sollya_17_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 17-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39efp-1")
      ),  # 0.69314718055994528622676398299518041312694549560547
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c5dap-3")
      ),  # 0.240226506959102803495653688514721579849720001220703
      jnp.float64(
          float.fromhex("0x1.c6b08d7047835p-5")
      ),  # 5.5504108664749564339135901036570430733263492584229e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6fc5eccdp-7")
      ),  # 9.6181291089493473750460239557469321880489587783813e-3
      jnp.float64(
          float.fromhex("0x1.5d87fe3710a92p-10")
      ),  # 1.33335579973051622862389997692389442818239331245422e-3
      jnp.float64(
          float.fromhex("0x1.4309226a1bf8fp-13")
      ),  # 1.54035416304318097070399917747351992147741839289665e-4
      jnp.float64(
          float.fromhex("0x1.ffc6e1740d2c1p-17")
      ),  # 1.5252139498136780542823960626908075255414587445557e-5
      jnp.float64(
          float.fromhex("0x1.635cb807000fdp-20")
      ),  # 1.3238272928984254405375799515387491567253164248541e-6
      jnp.float64(
          float.fromhex("0x1.99628e09bb63p-24")
      ),  # 9.5317368355607159509618013515375167798993061296642e-8
      jnp.float64(
          float.fromhex("0x1.64d92120db1cbp-26")
      ),  # 2.0771296796645253046995177842286295355478387136827e-8
      jnp.float64(
          float.fromhex("-0x1.6f9f5871f995bp-26")
      ),  # -2.13984425247689551835138000194722063262986466725124e-8
      jnp.float64(
          float.fromhex("0x1.be83c525dd4e8p-26")
      ),  # 2.59905778214070094266817606901143200559545221040025e-8
      jnp.float64(
          float.fromhex("-0x1.85b7581229719p-26")
      ),  # -2.26844677006575771862937831871778748826784521952504e-8
      jnp.float64(
          float.fromhex("0x1.e5bb59d4067d6p-27")
      ),  # 1.4136657101914818242744382548008175159992561020772e-8
      jnp.float64(
          float.fromhex("-0x1.987d427453d6cp-28")
      ),  # -5.9443015963732203287623912261669612000503093440784e-9
      jnp.float64(
          float.fromhex("0x1.9f5fa6f412bb6p-30")
      ),  # 1.5111205039170420363037077759104861995886892600538e-9
      jnp.float64(
          float.fromhex("-0x1.81ab1466d2becp-33")
      ),  # -1.7538162867573325382705641374363712825523720084675e-10
  ]


def exp64_poly_sollya_17(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 17-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_17_coefficients())


def exp64_poly_sollya_18_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 18-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39efp-1")
      ),  # 0.69314718055994528622676398299518041312694549560547
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c5e2p-3")
      ),  # 0.240226506959103025540258613546029664576053619384766
      jnp.float64(
          float.fromhex("0x1.c6b08d7046ee2p-5")
      ),  # 5.5504108664733001199387274482432985678315162658691e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6fca43d3p-7")
      ),  # 9.6181291094427408955302993831537605728954076766968e-3
      jnp.float64(
          float.fromhex("0x1.5d87fe139d62dp-10")
      ),  # 1.33335579167005312435645425495067684096284210681915e-3
      jnp.float64(
          float.fromhex("0x1.43092dc952602p-13")
      ),  # 1.54035499045954203754110745094862977566663175821304e-4
      jnp.float64(
          float.fromhex("0x1.ffc1efb9efc45p-17")
      ),  # 1.52515639121314969646686990345507695110427448526025e-5
      jnp.float64(
          float.fromhex("0x1.6420187c23cf5p-20")
      ),  # 1.32667039934639287850515632777304020351039071101695e-6
      jnp.float64(
          float.fromhex("0x1.6d4211f5ebcedp-24")
      ),  # 8.5043275392953150114597031267865689940776974253822e-8
      jnp.float64(
          float.fromhex("0x1.a01c1d859b447p-25")
      ),  # 4.8441559247259465155302954634738976658070441771997e-8
      jnp.float64(
          float.fromhex("-0x1.4ccc128fa406bp-24")
      ),  # -7.7485376554508143031483539519416181562405654403847e-8
      jnp.float64(
          float.fromhex("0x1.dfde193b03471p-24")
      ),  # 1.11727875770806271384063461469399181069661608489696e-7
      jnp.float64(
          float.fromhex("-0x1.03afb53b6593dp-23")
      ),  # -1.2092588428926872273101169544956556833881222701166e-7
      jnp.float64(
          float.fromhex("0x1.a15811c343ea2p-24")
      ),  # 9.7170477043402470246198057886310994746281721745618e-8
      jnp.float64(
          float.fromhex("-0x1.e254d023713f5p-25")
      ),  # -5.6150753626210500050204007605261780078365063673118e-8
      jnp.float64(
          float.fromhex("0x1.7ad070cb954a2p-26")
      ),  # 2.20498897320868605517021128915172711870695820834953e-8
      jnp.float64(
          float.fromhex("-0x1.69ec33a80ab19p-28")
      ),  # -5.266667914319438690365455868003588535408709958574e-9
      jnp.float64(
          float.fromhex("0x1.3d674dc75238cp-31")
      ),  # 5.7735365873163727775441165840277640830535688110103e-10
  ]


def exp64_poly_sollya_18(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 18-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_18_coefficients())


def exp64_poly_sollya_19_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 19-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39efp-1")
      ),  # 0.69314718055994528622676398299518041312694549560547
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c5ecp-3")
      ),  # 0.240226506959103303096014769835164770483970642089844
      jnp.float64(
          float.fromhex("0x1.c6b08d704624fp-5")
      ),  # 5.5504108664710664899910597114285337738692760467529e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6fd0a79ep-7")
      ),  # 9.6181291101691789025007750524309813044965267181396e-3
      jnp.float64(
          float.fromhex("0x1.5d87fdda23a66p-10")
      ),  # 1.33335577860163026139850295237465616082772612571716e-3
      jnp.float64(
          float.fromhex("0x1.4309423c9827p-13")
      ),  # 1.540356478413498590718244685149329598061740398407e-4
      jnp.float64(
          float.fromhex("0x1.ffb8016960669p-17")
      ),  # 1.52504078012621934937577752866566527245595352724195e-5
      jnp.float64(
          float.fromhex("0x1.65d9a4974958p-20")
      ),  # 1.3330957580685030058192941382344542944338172674179e-6
      jnp.float64(
          float.fromhex("0x1.f83bac4cf91fp-25")
      ),  # 5.8700458361962202183815650452514134371995169203728e-8
      jnp.float64(
          float.fromhex("0x1.16950cb8f4b57p-23")
      ),  # 1.29724957691365425553346106372909485315858546528034e-7
      jnp.float64(
          float.fromhex("-0x1.2055a4fff078cp-22")
      ),  # -2.6853247447552144870793183056845165879167325329036e-7
      jnp.float64(
          float.fromhex("0x1.e92afeb7a13adp-22")
      ),  # 4.555731538474663087929649330637404958110892039258e-7
      jnp.float64(
          float.fromhex("-0x1.3f0302dc5e164p-21")
      ),  # -5.9420571178664694045952650303110686991203692741692e-7
      jnp.float64(
          float.fromhex("0x1.3d5c1a073fe51p-21")
      ),  # 5.911286401760295035467539410956039347411206108518e-7
      jnp.float64(
          float.fromhex("-0x1.d88e4ea2032dep-22")
      ),  # -4.4010196564927654677414027226123316438588517485186e-7
      jnp.float64(
          float.fromhex("0x1.fd9834a51eecp-23")
      ),  # 2.372984556942233806469885748491321919573238119483e-7
      jnp.float64(
          float.fromhex("-0x1.77b8032e3b9efp-24")
      ),  # -8.7478849695078657930183711668309376463525950384792e-8
      jnp.float64(
          float.fromhex("0x1.52c49f77db2f5p-26")
      ),  # 1.9718896265342284878701871225943620391518606993486e-8
      jnp.float64(
          float.fromhex("-0x1.19ae0f5f989b8p-29")
      ),  # -2.04949117381917949695797323463736239190779997443315e-9
  ]


def exp64_poly_sollya_19(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 19-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_19_coefficients())


def exp64_poly_sollya_20_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 20-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39efp-1")
      ),  # 0.69314718055994528622676398299518041312694549560547
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c5f5p-3")
      ),  # 0.240226506959103552896195310495386365801095962524414
      jnp.float64(
          float.fromhex("0x1.c6b08d7045564p-5")
      ),  # 5.5504108664687717977770375910040456801652908325195e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6fd8173ep-7")
      ),  # 9.6181291110145582246016005001365556381642818450928e-3
      jnp.float64(
          float.fromhex("0x1.5d87fd8ea02fdp-10")
      ),  # 1.33335576143184188387114108564901471254415810108185e-3
      jnp.float64(
          float.fromhex("0x1.4309608c9a57dp-13")
      ),  # 1.54035868394058252659264973694064337905729189515114e-4
      jnp.float64(
          float.fromhex("0x1.ffa760e61717dp-17")
      ),  # 1.5248472163325400963022372702315720971455448307097e-5
      jnp.float64(
          float.fromhex("0x1.691f087bd15d4p-20")
      ),  # 1.34528138935726218774037790670838887763238744810224e-6
      jnp.float64(
          float.fromhex("0x1.ff9041269783dp-30")
      ),  # 1.86105715009595905549560669194108022961309245602024e-9
      jnp.float64(
          float.fromhex("0x1.62dbe5777ff7cp-22")
      ),  # 3.3048816969193784504001034502129741099452076014131e-7
      jnp.float64(
          float.fromhex("-0x1.b489aeab15784p-21")
      ),  # -8.1311505562940601603391828355871240319174830801785e-7
      jnp.float64(
          float.fromhex("0x1.ad393828d6b0fp-20")
      ),  # 1.59898218950763334511350113181782361948535253759474e-6
      jnp.float64(
          float.fromhex("-0x1.49b7a1b033ff3p-19")
      ),  # -2.4565853991922959497146757706920183750298747327179e-6
      jnp.float64(
          float.fromhex("0x1.8a11b33dae43bp-19")
      ),  # 2.9360438976414472774954713546469164953123254235834e-6
      jnp.float64(
          float.fromhex("-0x1.6a42c54cb082ap-19")
      ),  # -2.69905345926057779385289284168614187819912331178784e-6
      jnp.float64(
          float.fromhex("0x1.f6144140bb11cp-20")
      ),  # 1.8703904773277856484771895784202655477201915346086e-6
      jnp.float64(
          float.fromhex("-0x1.fb1d64d4c8954p-21")
      ),  # -9.4457495922556670978260419391236979436143883503973e-7
      jnp.float64(
          float.fromhex("0x1.6019c6a2d513ap-22")
      ),  # 3.2791931852306947535376788552263782605677988613024e-7
      jnp.float64(
          float.fromhex("-0x1.2c65bff02b753p-24")
      ),  # -6.9941733962377529284271972503245207164468411065172e-8
      jnp.float64(
          float.fromhex("0x1.daabb2bc4e001p-28")
      ),  # 6.9073677301390444310070334448057455256630987605604e-9
  ]


def exp64_poly_sollya_20(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 20-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_20_coefficients())


def exp64_poly_sollya_21_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 21-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39efp-1")
      ),  # 0.69314718055994528622676398299518041312694549560547
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c6p-3")
      ),  # 0.2402265069591038582075270824134349822998046875
      jnp.float64(
          float.fromhex("0x1.c6b08d7044479p-5")
      ),  # 5.5504108664657665628272553703936864621937274932861e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6fe2a20bp-7")
      ),  # 9.618129112213066531089644684016093378886580467224e-3
      jnp.float64(
          float.fromhex("0x1.5d87fd19e765p-10")
      ),  # 1.33335573489236705602145605098485248163342475891113e-3
      jnp.float64(
          float.fromhex("0x1.430993eea8974p-13")
      ),  # 1.5403624225480592405818081047641499026212841272354e-4
      jnp.float64(
          float.fromhex("0x1.ff884d7970b8bp-17")
      ),  # 1.52448544551481257293889734483194331460254034027457e-5
      jnp.float64(
          float.fromhex("0x1.6fe626105a2cdp-20")
      ),  # 1.37053064371889161343284497390770937386150762904435e-6
      jnp.float64(
          float.fromhex("-0x1.16124eeffc502p-23")
      ),  # -1.29487140565087633105979337058699663032257376471534e-7
      jnp.float64(
          float.fromhex("0x1.c918782f5e816p-21")
      ),  # 8.514068720454528011826146258189673687866161344573e-7
      jnp.float64(
          float.fromhex("-0x1.43c8265410e37p-19")
      ),  # -2.41236265633036702597520421764709652734381961636245e-6
      jnp.float64(
          float.fromhex("0x1.6cce616cae358p-18")
      ),  # 5.4360356044829173118547449572446339516318403184414e-6
      jnp.float64(
          float.fromhex("-0x1.44f282cf03acp-17")
      ),  # -9.6841844463823592223383762700450461125001311302185e-6
      jnp.float64(
          float.fromhex("0x1.c91109c1b5308p-17")
      ),  # 1.36216448284680980785124604359737077174941077828407e-5
      jnp.float64(
          float.fromhex("-0x1.f85fb15fda276p-17")
      ),  # -1.50315105995207490871112127051034690339292865246534e-5
      jnp.float64(
          float.fromhex("0x1.af353d969fc34p-17")
      ),  # 1.28509989683040136095905359159274894409463740885258e-5
      jnp.float64(
          float.fromhex("-0x1.17a5342721f78p-17")
      ),  # -8.3340801906442912643999121335980362346163019537926e-6
      jnp.float64(
          float.fromhex("0x1.09c5129ea9e3dp-18")
      ),  # 3.9602788592173542462433928623699586069051292724907e-6
      jnp.float64(
          float.fromhex("-0x1.5ced143c62e36p-20")
      ),  # -1.29985097805069437607874104079197508099241531454027e-6
      jnp.float64(
          float.fromhex("0x1.1aa33c7117e19p-22")
      ),  # 2.6322681551634228073232647272805539984119604923762e-7
      jnp.float64(
          float.fromhex("-0x1.a99da7a6117bp-26")
      ),  # -2.4774102457229423977276170851236081205115624470636e-8
  ]


def exp64_poly_sollya_21(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 21-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_21_coefficients())


def exp64_poly_sollya_22_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 22-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39efp-1")
      ),  # 0.69314718055994528622676398299518041312694549560547
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c60ap-3")
      ),  # 0.24022650695910413576328323870257008820772171020508
      jnp.float64(
          float.fromhex("0x1.c6b08d7043324p-5")
      ),  # 5.550410866462687775602091733162524178624153137207e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6feeb828p-7")
      ),  # 9.6181291135871288533465417458501178771257400512695e-3
      jnp.float64(
          float.fromhex("0x1.5d87fc84a3d55p-10")
      ),  # 1.3333557009536829447010619631441841193009167909622e-3
      jnp.float64(
          float.fromhex("0x1.4309dd25423b1p-13")
      ),  # 1.5403677495154152331420516208737581109744496643543e-4
      jnp.float64(
          float.fromhex("0x1.ff56f1c192d4fp-17")
      ),  # 1.52391083957567049161170233406714658030978171154857e-5
      jnp.float64(
          float.fromhex("0x1.7beb737f0c03cp-20")
      ),  # 1.4153112883989865046128163450500636599826975725591e-6
      jnp.float64(
          float.fromhex("-0x1.a3216caec23a3p-22")
      ),  # -3.9034575653788541165159910011073218782939875381999e-7
      jnp.float64(
          float.fromhex("0x1.0e6315bb0d4bap-19")
      ),  # 2.0145405108759884864268359383476791890643653459847e-6
      jnp.float64(
          float.fromhex("-0x1.b0be6cb333c7p-18")
      ),  # -6.4483858068362700502115791856283522065496072173119e-6
      jnp.float64(
          float.fromhex("0x1.1415a9ad61977p-16")
      ),  # 1.64559257221029646802486906542739575343148317188025e-5
      jnp.float64(
          float.fromhex("-0x1.192125c2bbc33p-15")
      ),  # -3.3513245872522980128068587069023465119244065135717e-5
      jnp.float64(
          float.fromhex("0x1.c95c3eb395b02p-15")
      ),  # 5.4521600216508536392891193234078173190937377512455e-5
      jnp.float64(
          float.fromhex("-0x1.284aa89e865c8p-14")
      ),  # -7.0641430717797233825028246911870155599899590015411e-5
      jnp.float64(
          float.fromhex("0x1.2f69748958f02p-14")
      ),  # 7.233904229547383148320183199331268042442388832569e-5
      jnp.float64(
          float.fromhex("-0x1.e4866b20ce063p-15")
      ),  # -5.7759889620035536337335629886169385827088262885809e-5
      jnp.float64(
          float.fromhex("0x1.271b112481e36p-15")
      ),  # 3.5179344454458144164470112524867317915777675807476e-5
      jnp.float64(
          float.fromhex("-0x1.08b129ef30132p-16")
      ),  # -1.57768753836783724307964160704642608834546990692616e-5
      jnp.float64(
          float.fromhex("0x1.4963c025bef1fp-18")
      ),  # 4.908288280477072594435491298403562154817336704582e-6
      jnp.float64(
          float.fromhex("-0x1.fbbe37e3f091dp-21")
      ),  # -9.4574511110720182151869470446592380596939619863406e-7
      jnp.float64(
          float.fromhex("0x1.6d00d35a5b25fp-24")
      ),  # 8.498393581019069762269661219220018288922346982872e-8
  ]


def exp64_poly_sollya_22(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 22-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_22_coefficients())


def exp64_poly_sollya_23_coefficients() -> List[Optional[JnpFloat]]:
  """Returns the 23-th order Sollya-optimized polynomial coefficients."""
  return [
      jnp.float64(float.fromhex("0x1p0")),  # 1
      jnp.float64(
          float.fromhex("0x1.62e42fefa39efp-1")
      ),  # 0.69314718055994528622676398299518041312694549560547
      jnp.float64(
          float.fromhex("0x1.ebfbdff82c615p-3")
      ),  # 0.240226506959104441074615010620618704706430435180664
      jnp.float64(
          float.fromhex("0x1.c6b08d7041e92p-5")
      ),  # 5.5504108664590337540722941866988549008965492248535e-2
      jnp.float64(
          float.fromhex("0x1.3b2ab6ffe4667p-7")
      ),  # 9.618129115355601374548477622283826349303126335144e-3
      jnp.float64(
          float.fromhex("0x1.5d87fbb38559bp-10")
      ),  # 1.3333556534055105140951225095591325953137129545212e-3
      jnp.float64(
          float.fromhex("0x1.430a4d2c6c50ep-13")
      ),  # 1.54037590062418674238114979502256574050989001989365e-4
      jnp.float64(
          float.fromhex("0x1.ff042eb225a6dp-17")
      ),  # 1.5229473636230531153668878163731648101020255126059e-5
      jnp.float64(
          float.fromhex("0x1.9215b0567e17ep-20")
      ),  # 1.4978823138484091534092506464714844582886144053191e-6
      jnp.float64(
          float.fromhex("-0x1.eea947f2f02b5p-21")
      ),  # -9.2137838546981330981614161448889355199298734078184e-7
      jnp.float64(
          float.fromhex("0x1.376bb92de9fbp-18")
      ),  # 4.6405314559141308148812465894650358677608892321587e-6
      jnp.float64(
          float.fromhex("-0x1.16a28914b9d33p-16")
      ),  # -1.66079344862377763388221890705764849371917080134153e-5
      jnp.float64(
          float.fromhex("0x1.8f22ff0a0166dp-15")
      ),  # 4.7580802927921262183378475763007031673623714596033e-5
      jnp.float64(
          float.fromhex("-0x1.cbbed12a9c8bp-14")
      ),  # -1.09611840039910589036931121142970368964597582817078e-4
      jnp.float64(
          float.fromhex("0x1.aafce281b1f3ep-13")
      ),  # 2.036036640247154846060373412086619282490573823452e-4
      jnp.float64(
          float.fromhex("-0x1.3f8da54fc8d94p-12")
      ),  # -3.0474977846691700573977801269620613311417400836945e-4
      jnp.float64(
          float.fromhex("0x1.7fc5aa49fb2e4p-12")
      ),  # 3.6599362340332245759066953638694030814804136753082e-4
      jnp.float64(
          float.fromhex("-0x1.6ea02f5941206p-12")
      ),  # -3.4964153526598239826034264510212778986897319555283e-4
      jnp.float64(
          float.fromhex("0x1.12a1e128fdec5p-12")
      ),  # 2.6190981094441025860478089448690752760739997029305e-4
      jnp.float64(
          float.fromhex("-0x1.3b6617971a8d4p-13")
      ),  # -1.50393866280857020113759237034400939592160284519196e-4
      jnp.float64(
          float.fromhex("0x1.0be0255cf3a45p-14")
      ),  # 6.3866512802970390222552510284259597028722055256367e-5
      jnp.float64(
          float.fromhex("-0x1.3ce1316abdb24p-16")
      ),  # -1.88874995883061598403798719836288455553585663437843e-5
      jnp.float64(
          float.fromhex("0x1.d1eee0ec6b25dp-19")
      ),  # 3.4714722600608575055864627595392235548388271126896e-6
      jnp.float64(
          float.fromhex("-0x1.407e947c36e4fp-22")
      ),  # -2.9848371930843058447263538811300431774498065351509e-7
  ]


def exp64_poly_sollya_23(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 23-th order Sollya-optimized polynomial.

  http://cs/experimental/users/evol-brain-gpu/projects/graphs/joy/sollya/2022-12-06.exp64_stuff.sollya

  Args:
    x: the arguments of the exp2 function.
  """
  return _horner_scheme(x[0], exp64_poly_sollya_23_coefficients())


################################################################################
# 64-bit baselines for f=log2(x) on [1, 2]. ####################################
################################################################################


def table_log2_64bits_baselines() -> List[Tuple[str, FinalizedFn]]:
  """Baselines for 64-bit log2(x) on (1, 2].

  Returns:
    A list of (name, finalized_fn).
  """
  return [("JAX Numpy", log64_jnp)]  # pytype: disable=bad-return-type  # jax-ndarray


def log64_jnp(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns JAX NumPy approximation."""
  return jnp.log2(x[0])


def exp64_rational_minimax_1_1(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (1 1)th order rational approximation with minimax coeffs.

  HornerForm[

          MiniMaxApproximation[2^x, {x, {0, 1}, 1, 1},
             WorkingPrecision -> 50][[2, 1]]]

  Args:
    x: the arguments of the exp2 function.
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("-3.4041284781325858605203315026390577568818065650"),
          jnp.float64("-1.414213030554616229941944571196552728415404429539"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("-3.407083236107348289659620511677855700270152527089"),
          jnp.float64("1.000000000000000000000000000000000000000000000000"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_minimax_2_2(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (2 2)th order rational approximation with minimax coeffs.

  HornerForm[

          MiniMaxApproximation[2^x, {x, {0, 1}, 2, 2},
             WorkingPrecision -> 50][[2, 1]]]

  Args:
    x: the arguments of the exp2 function.
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("29.6280389583805549892872298454995601807130876102"),
          jnp.float64("10.85810381450080540340057113890973592026406769929"),
          jnp.float64("1.414213562372961839761481428816763964033322265077"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("29.62802609879383910843057015255410087460109394249"),
          jnp.float64("-9.67783883806176121564316128933597469042603975905"),
          jnp.float64("1.000000000000000000000000000000000000000000000000"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_minimax_3_3(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (3 3)th order rational approximation with minimax coeffs.

  HornerForm[

          MiniMaxApproximation[2^x, {x, {0, 1}, 3, 3},
             WorkingPrecision -> 50][[2, 1]]]

  Args:
    x: the arguments of the exp2 function.
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("-427.868758324026672251643413591460325345258366956"),
          jnp.float64("-153.4221645554290853785938780294732551909048787267"),
          jnp.float64("-22.392664552077511752238825168768022579423374940"),
          jnp.float64("-1.414213562373095048791828120744743046873792831957"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("-427.8687583638592267661080189707553500600563206147"),
          jnp.float64("143.1538628486815462916441274563324404303064260689"),
          jnp.float64("-18.83400495360963233463144047259592596459149193859"),
          jnp.float64("1.000000000000000000000000000000000000000000000000"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_minimax_4_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (4 4)th order rational approximation with minimax coeffs.

  HornerForm[

          MiniMaxApproximation[2^x, {x, {0, 1}, 4, 4},
             WorkingPrecision -> 50][[2, 1]]]

  Args:
    x: the arguments of the exp2 function.
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("8645.68238934645214197504472025719719634145426289"),
          jnp.float64("3070.50919216316330904212384877142458145919949954"),
          jnp.float64("471.227709119188795110217324278013928235065477709"),
          jnp.float64("38.0077867926652523849476644329851846278924260882"),
          jnp.float64("1.414213562373095048800901713489923403307279482954"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("8645.68238934635623341888395547730562376432032606"),
          jnp.float64("-2922.221180013519614972074316014323094577151272385"),
          jnp.float64("419.8349999381385954828009410142861718889184443602"),
          jnp.float64("-30.87556377898609955860826239266911154118678986975"),
          jnp.float64("1.000000000000000000000000000000000000000000000000"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_minimax_5_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (5 5)th order rational approximation with minimax coeffs.

  HornerForm[

          MiniMaxApproximation[2^x, {x, {0, 1}, 5, 5},
             WorkingPrecision -> 50][[2, 1]]]

  Args:
    x: the arguments of the exp2 function.
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("-224568.982733012105331946796590909042716821862904"),
          jnp.float64("-79327.8611890547830156092551015826007704194973383"),
          jnp.float64("-12514.32759909540475780357000064511465781270093335"),
          jnp.float64("-1118.211866708084638039121624106021549644570144091"),
          jnp.float64("-57.703467922369471434040878603392496747570687837"),
          jnp.float64("-1.414213562373095048801467739278322884222287429197"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("-224568.9827330121055208518761816507890684202391375"),
          jnp.float64("76331.4960335476450257473889292830895089761018935"),
          jnp.float64("-11475.86656936331759861571169928922293762816729708"),
          jnp.float64("963.905247616105946284130679231812997173355560144"),
          jnp.float64("-45.80251346588787392866178016964726023488695484491"),
          jnp.float64("1.000000000000000000000000000000000000000000000000"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_minimax_6_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (6 6)th order rational approximation with minimax coeffs.

  HornerForm[
          MiniMaxApproximation[2^x, {x, {0, 1}, 6, 6},
             WorkingPrecision -> 50][[2, 1]]]

  Args:
    x: the arguments of the exp2 function.
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("1.0000000000000000000000441599721678342879445406787"),
          jnp.float64("0.35203245589268587800834721529829750482601260661364"),
          jnp.float64("0.056507868835118329308353468509810112347424392310025"),
          jnp.float64("0.0053439007055788682531793669868282447124238885655518"),
          jnp.float64(
              "0.00031770881056990130630117428505603535708559831774140"
          ),
          jnp.float64(
              "0.000011429747188738066107408801566885768885594797377839"
          ),
          jnp.float64("1.9838195333385935474055030850130918642773613567649e-7"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("1.0000000000000000000000000000000000000000000000000"),
          jnp.float64("-0.34111472466725943139396894833701788077845821406477"),
          jnp.float64("0.052724071526610519393286798673206867198647950024721"),
          jnp.float64(
              "-0.0047609507064225453880903568085425761994317947890137"
          ),
          jnp.float64(
              "0.00026716847148572337991204927871765626355580452847803"
          ),
          jnp.float64(
              "-8.9237150912089954334673589763357096151576653352675e-6"
          ),
          jnp.float64("1.4027722446740516891757040671335835834876884135408e-7"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_minimax_7_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (7 7)th order rational approximation with minimax coeffs.

  HornerForm[
          MiniMaxApproximation[2^x, {x, {0, 1}, 4, 4},
             WorkingPrecision -> 50][[2, 1]]]

  Args:
    x: the arguments of the exp2 function.
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("0.99999999999999999999999999829993652804215712627332"),
          jnp.float64("0.35119284586271568325369021201163149881768181319405"),
          jnp.float64("0.057050891096499309549786858780302459524112914642737"),
          jnp.float64("0.0055937002233860288082468521871083090957624329238467"),
          jnp.float64(
              "0.00036031625884456039313174049026883530170571832678060"
          ),
          jnp.float64(
              "0.000015394798266580508419024201132291208651137336739007"
          ),
          jnp.float64("4.0884517296657103901913825880117954961254995789348e-7"),
          jnp.float64("5.2882099197173746218349475701120558908775872526831e-9"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("1.0000000000000000000000000000000000000000000000000"),
          jnp.float64("-0.34195433469722962616354267408595925242555318950190"),
          jnp.float64("0.053849067113035192152610301101634838717771011195823"),
          jnp.float64(
              "-0.0050892421227804814077252168942691195483451500008603"
          ),
          jnp.float64(
              "0.00031367823567945660266201166827407151422614847889039"
          ),
          jnp.float64(
              "-0.000012698875325844253225154234481505776151548432727643"
          ),
          jnp.float64("3.1527249792204018856236661275031888354692094699148e-7"),
          jnp.float64(
              "-3.7393290945701236676521450456289291697603637690054e-9"
          ),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_minimax_8_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (8 8)th order rational approximation with minimax coeffs.

  HornerForm[
          MiniMaxApproximation[2^x, {x, {0, 1}, 4, 4},
             WorkingPrecision -> 50][[2, 1]]]

  Args:
    x: the arguments of the exp2 function.
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("1.0000000000000000000000000000000500490580858090300"),
          jnp.float64("0.35057705717277538501975983073141603869990064565756"),
          jnp.float64("0.057449902259976446656218863487195952461169594519892"),
          jnp.float64("0.0057756807927924803154215408696754815713476833893776"),
          jnp.float64(
              "0.00039182247872600479732829430202958706417095092057487"
          ),
          jnp.float64(
              "0.000018489890022711667699657232513732007311974846366279"
          ),
          jnp.float64("5.9745195132585512066277168489697365186763391302523e-7"),
          jnp.float64("1.2204676507073559105199458077680441828156139694793e-8"),
          jnp.float64(
              "1.2217418741658062276014636621833823061206434028459e-10"
          ),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("1.0000000000000000000000000000000000000000000000000"),
          jnp.float64("-0.34257012338716992439747229069784362378619450935126"),
          jnp.float64("0.054674910470765149336716964262311370589742500426284"),
          jnp.float64(
              "-0.0053317637823594294725407612464455184521318242902591"
          ),
          jnp.float64(
              "0.00034907699736093946359911382098551923979938566471979"
          ),
          jnp.float64(
              "-0.000015795168626672788526205865033503064113470258399224"
          ),
          jnp.float64("4.8529131835745647002276780956185437018702982434704e-7"),
          jnp.float64(
              "-9.3211310916056228767598144401304043536853585019834e-9"
          ),
          jnp.float64(
              "8.6390196408220322395151114644505851820262509718636e-11"
          ),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_minimax_9_9(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (9 9)th order rational approximation with minimax coeffs.

  HornerForm[
          MiniMaxApproximation[2^x, {x, {0, 1}, 4, 4},
             WorkingPrecision -> 50][[2, 1]]]

  Args:
    x: the arguments of the exp2 function.
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("0.99999999999999999999999999999999999883677336666516"),
          jnp.float64("0.35010612505041590646860510232554656308222855230655"),
          jnp.float64("0.057755446420513876536092746927968528638398673923919"),
          jnp.float64("0.0059142016377023039067979852281747447359676499166697"),
          jnp.float64(
              "0.00041605533346446597378251756863428831292415549454955"
          ),
          jnp.float64(
              "0.000020954466151369728569912039253337125782917036585586"
          ),
          jnp.float64("7.5988794766798358413509031621519691911696408057272e-7"),
          jnp.float64("1.9262179315751670788975177671305709522298718261760e-8"),
          jnp.float64(
              "3.1222956280161212023440290750999131096650325840082e-10"
          ),
          jnp.float64(
              "2.4905796509733531786810834307177429354899884329317e-12"
          ),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("1.0000000000000000000000000000000000000000000000000"),
          jnp.float64("-0.34304105550952940294862701913263084457693515538273"),
          jnp.float64("0.055306879904151162953563328264473210325424524337680"),
          jnp.float64(
              "-0.0055181603896319340487010738010528415157720925423936"
          ),
          jnp.float64(
              "0.00037683299310222800027926310932796301289175685318124"
          ),
          jnp.float64(
              "-0.000018339590965341709469695758913536115474183992029241"
          ),
          jnp.float64("6.3899460687604865201756266891288163893745707612441e-7"),
          jnp.float64(
              "-1.5450054551121091291404052301358976794115798027146e-8"
          ),
          jnp.float64(
              "2.3662959298652728469542261003277754360897147430326e-10"
          ),
          jnp.float64(
              "-1.7611057602884827511655044203934724577132931206649e-12"
          ),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_minimax_10_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (10 10)th order rational approximation with minimax coeffs.

  HornerForm[
          MiniMaxApproximation[2^x, {x, {0, 1}, 10, 10},
             WorkingPrecision -> 50][[2, 1]]]

  Args:
    x: the arguments of the exp2 function.
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("1.0000000000000000000000000000000000000000218857983"),
          jnp.float64("0.34973431845727055712489981325646068010587534180370"),
          jnp.float64("0.057996906165875506100952273539096983628071592088481"),
          jnp.float64("0.0060231882243165029119827621319533574743593403605035"),
          jnp.float64(
              "0.00043526815445206770620421333219511642018524350537790"
          ),
          jnp.float64(
              "0.000022956223462736023103364157384069287623484447725715"
          ),
          jnp.float64("8.9869608745734935904229746832413298003143669997539e-7"),
          jnp.float64("2.5914769376625271477167842991897881917516748490295e-8"),
          jnp.float64(
              "5.2983233515848041120989333195478210528144386490915e-10"
          ),
          jnp.float64(
              "6.9830823961388695095788419296081824472390813679121e-12"
          ),
          jnp.float64(
              "4.5427842798250504726622728478160199299442123325856e-14"
          ),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("1.0000000000000000000000000000000000000000000000000"),
          jnp.float64("-0.34341286210267475229233230820171588795032651983124"),
          jnp.float64("0.055806056341265090103969166318626439741538504365581"),
          jnp.float64(
              "-0.0056658587438695416997553587249401760685715375690179"
          ),
          jnp.float64(
              "0.00039914389360704018330566469479078484253376839597767"
          ),
          jnp.float64(
              "-0.000020451771105081488846638327571067002940353961796762"
          ),
          jnp.float64("7.7465732644233133100225370328175353475043017533449e-7"),
          jnp.float64(
              "-2.1503308393012798493863247876231655546608342316961e-8"
          ),
          jnp.float64(
              "4.2053360643189227842535602760101897990003997942753e-10"
          ),
          jnp.float64(
              "-5.2590082728673934190535364305286303431274128413855e-12"
          ),
          jnp.float64(
              "3.2122335697319397352169525345703260849628464880088e-14"
          ),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_minimax_11_11(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (11 11)th order rational approximation with minimax coeffs.

  HornerForm[
          MiniMaxApproximation[2^x, {x, {0, 1}, 11, 11},
             WorkingPrecision -> 50][[2, 1]]]

  Args:
    x: the arguments of the exp2 function.
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("0.99999999999999999999999999999999999999999999965984"),
          jnp.float64("0.34943332204263369866218344220643828692946122808510"),
          jnp.float64("0.058192521898718047302254131602074528512343114538108"),
          jnp.float64("0.0061111847195379820920805295377762520336414273483853"),
          jnp.float64(
              "0.00045087218559863483811925604582721543968769039853549"
          ),
          jnp.float64(
              "0.000024611147386329746595657148980983934447631725584782"
          ),
          jnp.float64("1.0176203364731161625308819965601017606972606586247e-6"),
          jnp.float64("3.1993297217662878918754167891857413540464216564556e-8"),
          jnp.float64(
              "7.5276020975259132178099873419602514483408264160863e-10"
          ),
          jnp.float64(
              "1.2687045405117078492291040872021326548909825386200e-11"
          ),
          jnp.float64(
              "1.3866079027602754077339465654865458558119016517862e-13"
          ),
          jnp.float64(
              "7.4969039011527073388163836659823624866042436701338e-16"
          ),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("1.0000000000000000000000000000000000000000000000000"),
          jnp.float64("-0.34371385851731161075504867925173828114603926608920"),
          jnp.float64("0.056210306890271822003289325322085873003915851437007"),
          jnp.float64(
              "-0.0057857600596364630924301229064339706258780478430905"
          ),
          jnp.float64(
              "0.00041745202410441651585995438903048838288775810338436"
          ),
          jnp.float64(
              "-0.000022226145973552895725364372980351371513759714939242"
          ),
          jnp.float64("8.9360328130793157748369214533454029680799698932329e-7"),
          jnp.float64(
              "-2.7215832351617338654767634548604103493215039865686e-8"
          ),
          jnp.float64(
              "6.1752133914625910078333665038449193879269765996655e-10"
          ),
          jnp.float64(
              "-9.9807318037938119797968828638970551209471684042666e-12"
          ),
          jnp.float64(
              "1.0387920783391461919732233344511599265134361172794e-13"
          ),
          jnp.float64(
              "-5.3011115864089618224894774732373741939505204522290e-16"
          ),
      ],
  )
  return jnp.divide(numerator, denominator)


def _exp64_poly_taylor(order: int, x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the Taylor polynomial about x=0 of the given order."""
  coeffs = _exp2_64_poly_taylor_coeffs()
  coeffs = [jnp.float64(c) for c in coeffs]
  if order >= len(coeffs):
    raise NotImplementedError("Missing coefficients.")
  coeffs = coeffs[: order + 1]
  return _horner_scheme(x[0], coeffs)


def _exp2_64_poly_taylor_coeffs() -> List[jnp.float64]:
  """Generated with N[HornerForm[Normal[Series[2^x, {x, 1/2, 24}]]], 50]."""
  return [
      jnp.float64("0.99999999999999999999999999999999997972699546278672"),
      jnp.float64("0.69314718055994530941723212145817754062473963736397"),
      jnp.float64("0.24022650695910071233355126316331013109260302701240"),
      jnp.float64("0.055504108664821579953142263768949405854279102881558"),
      jnp.float64("0.0096181291076284771619790715702211004240612456234161"),
      jnp.float64("0.0013333558146428443423412222262793797497899747323237"),
      jnp.float64("0.00015403530393381609954437079944520462039070202592000"),
      jnp.float64("0.000015252733804059840280026332371870730693721307902463"),
      jnp.float64("1.3215486790144309488365832706711755529447630944893e-6"),
      jnp.float64("1.0178086009239699728836783060919013983640816855326e-7"),
      jnp.float64("7.0549116208011232895293762897295261412447220689548e-9"),
      jnp.float64("4.4455382718708125229246804069401275420558433372934e-10"),
      jnp.float64("2.5678435993487983401663437128728325798549370246946e-11"),
      jnp.float64("1.3691488853908213657774584787653210249055397006751e-12"),
      jnp.float64("6.7787263547585207802232500206426630067524101553423e-14"),
      jnp.float64("3.1324367079394728201174151211325778173122918681054e-15"),
      jnp.float64("1.3570247853371144418327766990788685827257134989538e-16"),
      jnp.float64("5.5330474259987011456626818064960408160405758295139e-18"),
      jnp.float64("2.1306684273285042564895361861633819699733061068733e-19"),
      jnp.float64("7.7734412635361926908845704881688266327441039336812e-21"),
      jnp.float64("2.6917792643491948442073783483937597209476217008280e-22"),
      jnp.float64("8.9720032059695829600741720250614027785502235767951e-24"),
      jnp.float64("2.5888401290454146451892783545549402223522747704149e-25"),
      jnp.float64("1.1940050769132373625119692798290992874369536586670e-26"),
  ]


def exp64_poly_taylor_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 10-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=10, x=x)


def exp64_poly_taylor_11(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 11-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=11, x=x)


def exp64_poly_taylor_12(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 12-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=12, x=x)


def exp64_poly_taylor_13(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 13-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=13, x=x)


def exp64_poly_taylor_14(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 14-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=14, x=x)


def exp64_poly_taylor_15(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 15-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=15, x=x)


def exp64_poly_taylor_16(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 16-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=16, x=x)


def exp64_poly_taylor_17(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 17-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=17, x=x)


def exp64_poly_taylor_18(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 18-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=18, x=x)


def exp64_poly_taylor_19(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 19-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=19, x=x)


def exp64_poly_taylor_20(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 20-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=20, x=x)


def exp64_poly_taylor_21(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 21-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=21, x=x)


def exp64_poly_taylor_22(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 22-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=22, x=x)


def exp64_poly_taylor_23(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the 23-th order Taylor polynomial about x=1/2."""

  return _exp64_poly_taylor(order=23, x=x)


def exp64_rational_pade_1_1(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (1, 1)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {1, 1}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("-4.6765961060238330034928531732035589329872268024087"),
          jnp.float64("-1.9605162869370943834278034472704667625829213981981"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("-4.6931471805599453094172321214581765680755001343603"),
          jnp.float64("1.3862943611198906188344642429163531361510002687205"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_2_2(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (2, 2)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {2, 2}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("56.798616440652145896000905598798967056557190760664"),
          jnp.float64("20.808342769780533184786971285820561708153180251342"),
          jnp.float64("2.7178526734645994163466700814250394428418765270348"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("56.798219180637545137673887983824783788636554563918"),
          jnp.float64("-18.557344389111493124681981020302897520734215031024"),
          jnp.float64("1.9218120556728056986684101053066598869222118063782"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_3_3(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (3, 3)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {3, 3}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("-1138.2192135070601931434988003256948795878807584720"),
          jnp.float64("-408.12125257843621118754863578096808230832569852856"),
          jnp.float64("-59.57684840978249567792830117368693479494583742912"),
          jnp.float64("-3.7677438355785935429278538536760078888394661464727"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("-1138.2192203204126379318650233644140662038094295701"),
          jnp.float64("380.83228391685466216662638232295497322730990675809"),
          jnp.float64("-50.119785160014490524668085518700603816006601177260"),
          jnp.float64("2.6641972159114358377508286608938443532490118827889"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_4_4(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (4, 4)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {4, 4}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("31897.427044652980637733542741234910242838841555935"),
          jnp.float64("11328.236237373417066882607747290851259197568683902"),
          jnp.float64("1738.6240978099062160053249534582795950518711214315"),
          jnp.float64("140.26334935633007663525585274329142283879465818583"),
          jnp.float64("5.2232020334068325409291507018744463573919938365396"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("31897.427044562077767684755316176382351651122403755"),
          jnp.float64("-10781.375384076092804356479829787646026288331193218"),
          jnp.float64("1549.0965554051002561516049407008532862953728045810"),
          jnp.float64("-113.95461179111610397043307340432378281854659667343"),
          jnp.float64("3.6933615773293352301999634842850043442930606809365"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_5_5(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (5, 5)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {5, 5}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("-1.1488542344591363748284013462406773193840998421870e6"),
          jnp.float64("-405825.14422477210813984645324718169810220075312961"),
          jnp.float64("-64022.202460669614213785868690050610225823137823819"),
          jnp.float64("-5721.1276385779718434434062910699802704759198237989"),
          jnp.float64("-295.2898831896553562196391409932462013308636720828"),
          jnp.float64("-7.2408955259018384944439604476882320450623832438138"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("-1.1488542344591373672236305155593404618235124167032e6"),
          jnp.float64("390499.92926495481631432330576086052201450927902720"),
          jnp.float64("-58710.887690451761314900186436235394085513714241962"),
          jnp.float64("4931.8549278313037407318635015243353625150582060061"),
          jnp.float64("-234.40191046033141949847354216557658839913460206462"),
          jnp.float64("5.1200863282285222745902932433905310966203844833724"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_6_6(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (6, 6)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {6, 6}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("5.0564911531161839965492104309263212502225596375811e7"),
          jnp.float64("1.7800448171293238310071474784175170545046039578380e7"),
          jnp.float64("2.8573423543261907244906107599793552070983210690402e6"),
          jnp.float64("270226.43826541071480151673257028494937904358976280"),
          jnp.float64("16067.013319196381117150826415633475008513792092189"),
          jnp.float64("578.12118626470654276360327652592031007340556638678"),
          jnp.float64("10.038012637015963589896467026630393903944875364526"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("5.0564911531161839956324299004764239763035662054498e7"),
          jnp.float64("-1.7248477691794657977015100922444505494080769483369e7"),
          jnp.float64("2.6660439635178068372981892117780553845448094259193e6"),
          jnp.float64("-240753.21628669248399550205094611416855479165042397"),
          jnp.float64("13511.530329274319473137106192500604904392490147166"),
          jnp.float64("-451.38109198700660866659847579758763393813271077261"),
          jnp.float64("7.0979468052702458670046144509276739406734713897781"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_7_7(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (7, 7)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {7, 7}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("-2.6299273700999142582970688393178280521842002017681e9"),
          jnp.float64("-9.2361040290292065229198609699109354979802203705014e8"),
          jnp.float64("-1.5004052434841982591033741739281181605508866823688e8"),
          jnp.float64("-1.4711404536965004784958108652415534605054253812741e7"),
          jnp.float64("-947670.27396434783565047435090824382347720595994648"),
          jnp.float64("-40493.221006273568975319377592810525966412148343952"),
          jnp.float64("-1075.5526742413789039833924895559968441749376355960"),
          jnp.float64("-13.915640315545433738574804979030649447682401494545"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("-2.6299273700999142582971422423363248218663447791018e9"),
          jnp.float64("8.9931633875926672908609058678373913356197532603395e8"),
          jnp.float64("-1.4162084331566173642615860891225740317113835488547e8"),
          jnp.float64("1.3384835535035823407290270251445950554034653314517e7"),
          jnp.float64("-825021.58624651132095746615414037882504823402473028"),
          jnp.float64("33402.851418763199519883256550418625719750275839827"),
          jnp.float64("-829.40949490113033166959185775789628357136486060554"),
          jnp.float64("9.8398436316750841614500112154276577759817328429700"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_8_8(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (8, 8)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {8, 8}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("1.5781993627013850942103225365866947626289796011022e11"),
          jnp.float64("5.5328000196573616018009867860654546510456331493443e10"),
          jnp.float64("9.0667714203337671018988554116321736448750632896610e9"),
          jnp.float64("9.1153194423284698680688185478954689129132702227451e8"),
          jnp.float64("6.1839886439743343392359678534515926316883985920435e7"),
          jnp.float64("2.9183176673672673252525587697060647757657560505999e6"),
          jnp.float64("94304.722877991707133034643650466883417599443524233"),
          jnp.float64("1926.6875106352894577147960704295791276279628420342"),
          jnp.float64("19.291173700813250159993961637708598209575743295063"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("1.5781993627013850942103225314008972063802702450665e11"),
          jnp.float64(
              "-5.4064443665223143043572615587763534424117647021039e10"
          ),
          jnp.float64("8.6288560967419248584932739535063815987155361266598e9"),
          jnp.float64("-8.4147777051066149978720423977635780550884609068671e8"),
          jnp.float64("5.5094104693804025339087399331312317816425791459980e7"),
          jnp.float64("-2.4930370079110190463162587899475767351553372722390e6"),
          jnp.float64("76602.071425674569202473500482018706778739766279317"),
          jnp.float64("-1471.5011619247826596466194230742763984021588536717"),
          jnp.float64("13.640919740892635099454452013173419665197331071006"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_9_9(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (9, 9)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {9, 9}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64(
              "-1.0733019222900769113604630483249338616476737251661e13"
          ),
          jnp.float64(
              "-3.7576935386433649098885985050711173592066133131430e12"
          ),
          jnp.float64(
              "-6.1989176452318747017052746753596218198121585889429e11"
          ),
          jnp.float64(
              "-6.3477896264993737727154566555982927630426735131651e10"
          ),
          jnp.float64("-4.4656447228791176145798946296374864775228012744912e9"),
          jnp.float64("-2.2491636690876141992371119923716408633303071232533e8"),
          jnp.float64("-8.1566617052319575632408462009273723602025151910011e6"),
          jnp.float64("-206774.78674154152596890449906090033372665197596566"),
          jnp.float64("-3352.0666622026871962612928086850290845949980608570"),
          jnp.float64("-26.743245320821740563915619133893020695252384945396"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64(
              "-1.0733019222900769113604630483252616701451994342633e13"
          ),
          jnp.float64("3.6818684746059983907858221671381901428333890964213e12"),
          jnp.float64(
              "-5.9361279984654851702728237766723776484838056575918e11"
          ),
          jnp.float64("5.9227404805008265295346002114323676762736141953749e10"),
          jnp.float64("-4.0446872855993512544824290585396430453948144524358e9"),
          jnp.float64("1.9685123976365536258696339051459449470543876785718e8"),
          jnp.float64("-6.8590697824267418088281415520867238083539744994793e6"),
          jnp.float64("165854.77831023162652406148442980072716512346825538"),
          jnp.float64("-2540.4620388884723854448520263420984570508904185880"),
          jnp.float64("18.910330117288459454011258660196203847860183512675"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_10_10(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (10, 10)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {10, 10}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("8.1578528600449581915305469306500837476955783983287e14"),
          jnp.float64("2.8530799118519477770981901455930404531714886938369e14"),
          jnp.float64("4.7313100499773078040889687797165091728522503782367e13"),
          jnp.float64("4.9136634515818289140683686074508033876618261169647e12"),
          jnp.float64("3.5509154764070279697948378944252941365280498156174e11"),
          jnp.float64("1.8727994439860946189380791259762220331788946784787e10"),
          jnp.float64("7.3318777979186888790818551234463068815014916542848e8"),
          jnp.float64("2.1143031597274066958503400796219379447353767823412e7"),
          jnp.float64("432299.55687830518778261762497477787841034543614744"),
          jnp.float64("5698.1439196492775287558542515117199133441306645682"),
          jnp.float64("37.074010186301079061116391588948927922278804683795"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("8.1578528600449581915305469306500835602113044513757e14"),
          jnp.float64(
              "-2.8015127975111011104772344670335770954121831214616e14"
          ),
          jnp.float64("4.5525920504024568562086951193395476211721859672090e13"),
          jnp.float64(
              "-4.6221717950040686857314613707493831853951258164485e12"
          ),
          jnp.float64("3.2562290563153244319994286532076525185045067726639e11"),
          jnp.float64(
              "-1.6684934647727184123601332111788289064753452796093e10"
          ),
          jnp.float64("6.3199777033461536373605301079643377733866053450933e8"),
          jnp.float64("-1.7544033500143061523244357867329656200300133290747e7"),
          jnp.float64("343124.40180482035192273941090926439144928144258122"),
          jnp.float64("-4291.3490458460242355404702397630307997786957757114"),
          jnp.float64("26.215284008512631131598666903973190649891080584564"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_11_11(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (11, 11)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {11, 11}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64(
              "-6.8531120735811733475518689785316649022438176187795e16"
          ),
          jnp.float64(
              "-2.3947049809006044817140474105745068070600561915457e16"
          ),
          jnp.float64(
              "-3.9880035358397568348566113019602771088442169086703e15"
          ),
          jnp.float64(
              "-4.1880849335726700690155462145934833116967713585956e14"
          ),
          jnp.float64(
              "-3.0899158414247367118633314706942362786296891970038e13"
          ),
          jnp.float64(
              "-1.6866700511407123226096933948304834272659112465660e12"
          ),
          jnp.float64(
              "-6.9741575274554124416496755371545665330314213471672e10"
          ),
          jnp.float64("-2.1926854143847402057111132938285048928686608464102e9"),
          jnp.float64("-5.1592961809498758945329179461381115773684216648294e7"),
          jnp.float64("-869597.17396958355714931049758912124039824842989049"),
          jnp.float64("-9504.8634872239467295746930549523749736437641376831"),
          jnp.float64("-51.395491265370571374551695369116726879607327061568"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64(
              "-6.8531120735811733475518689785316649022536081275632e16"
          ),
          jnp.float64("2.3555103309635062857450698113526093217783619950597e16"),
          jnp.float64(
              "-3.8521652303550885733290476157998461105943806995158e15"
          ),
          jnp.float64("3.9650755822253312552635504952770544404316972257679e14"),
          jnp.float64(
              "-2.8608903530972711575038288957068883455454542580988e13"
          ),
          jnp.float64("1.5232261575478911429836699148063650760215962588822e12"),
          jnp.float64(
              "-6.1242548041500613402742955527687522457505250681902e10"
          ),
          jnp.float64("1.8652714281906286772676602585094764468776128444845e9"),
          jnp.float64("-4.2324255035264542816556225218623484490060601478588e7"),
          jnp.float64("684106.40839702376722379515133157453249162867757571"),
          jnp.float64("-7120.7165304262008861094535828074042416556264939509"),
          jnp.float64("36.342100396157503157710094574269438197160221750181"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_12_12(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (12, 12)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {12, 12}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("6.3052550541940504617074092362413506849216669517838e18"),
          jnp.float64("2.2016978735153320698140822165192114201364263098795e18"),
          jnp.float64("3.6793853467943498947934187173956569651887936913855e17"),
          jnp.float64("3.8990123623391031790931871871839084858239888463976e16"),
          jnp.float64("2.9243769281879927705632312331799586692249798869580e15"),
          jnp.float64("1.6394336106796419952702340327395655997851066064590e14"),
          jnp.float64("7.0630047191678674647165638952596689048022600391748e12"),
          jnp.float64("2.3631985273547242537324145622401045402375509975046e11"),
          jnp.float64("6.1151763673577830846749989606702914011233745250775e9"),
          jnp.float64("1.1980781221266324290914555429295679081250376362841e8"),
          jnp.float64("1.6943129915355331481960521014385029884703364060635e6"),
          jnp.float64("15607.897596426599379310155272950163155131823550777"),
          jnp.float64("71.249279728169814924995613702375938550943748738749"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("6.3052550541940504617074092362413506849216664814168e18"),
          jnp.float64(
              "-2.1687718900106191727713607550260562801752323714636e18"
          ),
          jnp.float64("3.5652725836270748382123893944001608693070868848495e17"),
          jnp.float64(
              "-3.7106806462550889335479935938567690511057184190738e16"
          ),
          jnp.float64("2.7285507797051667588787356889097996526823008012553e15"),
          jnp.float64(
              "-1.4965355432622003278665681735453593876807790356757e14"
          ),
          jnp.float64("6.2924693992516056661791190942498884077367293917006e12"),
          jnp.float64(
              "-2.0489328996561166073910438564962875406237516943574e11"
          ),
          jnp.float64("5.1422935876704333418332711623770646391850687813182e9"),
          jnp.float64("-9.7315607061795567443699180003841064909567034511952e7"),
          jnp.float64("1.3227862943268535365081098800492758901559581120337e6"),
          jnp.float64("-11641.020416703817522815226000890490122482909761788"),
          jnp.float64("50.380848850446089601612748953070900828153429288607"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_13_13(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (13, 13)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {13, 13}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64(
              "-6.3055843140290975906778364508562822363212786543438e20"
          ),
          jnp.float64(
              "-2.2004958884976954163725614739657823976768996307018e20"
          ),
          jnp.float64(
              "-3.6881451721259916215214506366728651071819154115938e19"
          ),
          jnp.float64(
              "-3.9375711158894935744038893814521604195230551606662e18"
          ),
          jnp.float64(
              "-2.9931186047427739899927972542598098634579854279636e17"
          ),
          jnp.float64(
              "-1.7140635308907491567821517508165082352301521207256e16"
          ),
          jnp.float64(
              "-7.6247489177924272419744403008195877340794218366643e14"
          ),
          jnp.float64(
              "-2.6740471393943396250895279185749286492477901465191e13"
          ),
          jnp.float64(
              "-7.4135869561036471376415247083875549134346535922020e11"
          ),
          jnp.float64(
              "-1.6095976309518606738267638154366753642952384456662e10"
          ),
          jnp.float64("-2.6691663945471025197864552978036256909530989748807e8"),
          jnp.float64("-3.2137502240378167638410582363584796059944811828074e6"),
          jnp.float64("-25292.716735367211573824544687684604753253437547927"),
          jnp.float64("-98.772474721015547519055184612344135275397998919565"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64(
              "-6.3055843140290975906778364508562822363212786543647e20"
          ),
          jnp.float64("2.1702021005545903763260435562254762948265346943410e20"),
          jnp.float64(
              "-3.5831549036197710044451859817729858855377649994807e19"
          ),
          jnp.float64("3.7635427685996933853770850657597132868410369793378e18"),
          jnp.float64(
              "-2.8103397379014049774997320042060472535245177716784e17"
          ),
          jnp.float64("1.5783081212483130938404908836330579163520145183588e16"),
          jnp.float64(
              "-6.8718465419044249381833716778525858566574254913117e14"
          ),
          jnp.float64("2.3535276232078594301502885102042743014565176962807e13"),
          jnp.float64(
              "-6.3553106184701250637247993804871166161875031964264e11"
          ),
          jnp.float64("1.3397929768538235975348182890081754968835555220467e10"),
          jnp.float64("-2.1493603811956592674784359248463997448618482863595e8"),
          jnp.float64("2.4925281242356751018642460307509918626653262230406e6"),
          jnp.float64("-18792.606444916121708706504129368282761014983061860"),
          jnp.float64("69.842686669806937495302078530842286087251507816532"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_14_14(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (14, 14)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {14, 14}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("6.8103339970308564483325285461021878762554845960663e22"),
          jnp.float64("2.3754295892976936217123842761891050109073132494817e22"),
          jnp.float64("3.9912598287349654032031953704963334086137469114530e21"),
          jnp.float64("4.2880055424305062793006194424108362589764698575951e20"),
          jnp.float64("3.2959262978992858788762896773771309377732842436939e19"),
          jnp.float64("1.9205789672396289236151039279596453884254543552862e18"),
          jnp.float64("8.7655716262238014869611463810264231965363012195415e16"),
          jnp.float64("3.1895790112607448317986233324028075110230286300810e15"),
          jnp.float64("9.3189282457424601602194403712926759220323005246186e13"),
          jnp.float64("2.1808331257689774029813393016727031467988102132885e12"),
          jnp.float64("4.0349782666371552940538327629454799807604232295046e10"),
          jnp.float64("5.7408446979838911898512398329275483863061432213861e8"),
          jnp.float64("5.9578033268023685052031403608698401394925996822902e6"),
          jnp.float64("40525.945309649324394143423130422186974439365846661"),
          jnp.float64("136.92772473960079483710777239462140588968481422234"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("6.8103339970308564483325285461021878762554845960663e22"),
          jnp.float64(
              "-2.3451342194157874756362616857272541354396841217560e22"
          ),
          jnp.float64("3.8862640776466457610043660163327919946122172313248e21"),
          jnp.float64(
              "-4.1133359743329282605669156907074719201980174592353e20"
          ),
          jnp.float64("3.1109470801085584214857993129477984742305184858930e19"),
          jnp.float64(
              "-1.7812007422395984444141488021085939851078935399966e18"
          ),
          jnp.float64("7.9750333676286885983093642772988373833701699477758e16"),
          jnp.float64(
              "-2.8416072230930325574482908377323600097783764122779e15"
          ),
          jnp.float64("8.1126535256146328604923356000540181188252064871374e13"),
          jnp.float64(
              "-1.8506720871718140051824019273258612666115617484330e12"
          ),
          jnp.float64("3.3283271771025478054494739915233004127373783346417e10"),
          jnp.float64("-4.5876308388478964952056989937466995889538605072766e8"),
          jnp.float64("4.5941442025748731570964202365037197130226057878393e6"),
          jnp.float64("-30011.686060189630713367123313491106512622737543705"),
          jnp.float64("96.822522695816709334321475791049479425300608680282"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_15_15(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (15, 15)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {15, 15}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64(
              "-7.9002903902546125415264943393831564452110384227142e24"
          ),
          jnp.float64(
              "-2.7543922322782226881699353739817992168294221249040e24"
          ),
          jnp.float64(
              "-4.6379278078047205927257347689418893416529001796712e23"
          ),
          jnp.float64(
              "-5.0094770844252718651452541942465429951266870194690e22"
          ),
          jnp.float64(
              "-3.8867302577566843568376499865601400747733057202552e21"
          ),
          jnp.float64(
              "-2.2978744596841172052177444307529067165287051168748e20"
          ),
          jnp.float64(
              "-1.0710709482066175681432051822001435781865380194036e19"
          ),
          jnp.float64(
              "-4.0147975898570143081331651567327314879426513727063e17"
          ),
          jnp.float64(
              "-1.2224256231950866556038611331816611840005582622806e16"
          ),
          jnp.float64(
              "-3.0294978416998359252591559751192219889348243086889e14"
          ),
          jnp.float64(
              "-6.0745216676853991853181882408800942778544741309562e12"
          ),
          jnp.float64(
              "-9.7015820256487257504602368411884617546118034124958e10"
          ),
          jnp.float64("-1.1979047295103172014943051246009638849521448627102e9"),
          jnp.float64("-1.0828673187986803962453553366213081185180704377449e7"),
          jnp.float64("-64301.641879852243084848533222328247653169294636845"),
          jnp.float64("-189.82213268748512492842633694533695651725549198353"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64(
              "-7.9002903902546125415264943393831564452110384227142e24"
          ),
          jnp.float64("2.7216717773315920240388916005495044757610978850920e24"),
          jnp.float64(
              "-4.5245273523402417896591155060451596293540917864172e23"
          ),
          jnp.float64("4.8202439066648573489429084929473466336933946873201e22"),
          jnp.float64(
              "-3.6849279613215798109001886306314366669770553585217e21"
          ),
          jnp.float64("2.1439978547234949936018986580399073696813459465489e20"),
          jnp.float64(
              "-9.8217954610208432614482168402076198852581473659436e18"
          ),
          jnp.float64("3.6129050097590046114170906750606258656914749056712e17"),
          jnp.float64(
              "-1.0776853563020015840187166776539693857532426480869e16"
          ),
          jnp.float64("2.6113614804047839951277381255815992408344798790956e14"),
          jnp.float64(
              "-5.1080823514686321028999042033727350700372651758833e12"
          ),
          jnp.float64("7.9379084892776240900863370089017274672629529702211e10"),
          jnp.float64("-9.5078659631808229067561334839100023596530687299057e8"),
          jnp.float64("8.3076755949943072828599547204600503233903729927678e6"),
          jnp.float64("-47481.494773311620489995768505046981528073062064077"),
          jnp.float64("134.22451724261333460286135004576418719050265300549"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_16_16(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (16, 16)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {16, 16}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64("9.7966872884651858581341634185694369394723708865637e26"),
          jnp.float64("3.4142516780576837619942373257769804904099254785004e26"),
          jnp.float64("5.7597452957157962273386981793805876077525537337364e25"),
          jnp.float64("6.2499384441709496389025819203432173428107891919295e24"),
          jnp.float64("4.8881676452062559868110724576112853389144344377326e23"),
          jnp.float64("2.9255298613720688216536164381219155697167977042071e22"),
          jnp.float64("1.3878206073552772577757965665303256772830430449548e21"),
          jnp.float64("5.3307556557927409471980447114727488801164787730489e19"),
          jnp.float64("1.6781805866985709737513886220770968720893460612558e18"),
          jnp.float64("4.3517500134801764027685846566564419830177756277129e16"),
          jnp.float64("9.2816120806134597662568293447284432103618624178160e14"),
          jnp.float64("1.6143844226159596614969681105573735171478185389201e13"),
          jnp.float64("2.2498446522082140940732029584757169440961589133404e11"),
          jnp.float64("2.4346076228972281873789981362889644463357287218194e9"),
          jnp.float64("1.9345364388952437804873891304612412693435323577196e7"),
          jnp.float64("101158.04536470860958573562062429120329236819523579"),
          jnp.float64("263.14935216041229691603833424651263162734905040649"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64("9.7966872884651858581341634185694369394723708865637e26"),
          jnp.float64(
              "-3.3762944947694154387508413611676295478269169506811e26"
          ),
          jnp.float64("5.6281957228244949169364763547119491977971659346874e25"),
          jnp.float64(
              "-6.0298378120374261858105675025064711377439362956199e24"
          ),
          jnp.float64("4.6520518886528372567629198086130868276181058767078e23"),
          jnp.float64(
              "-2.7436721604318034485099240534178534942214727324323e22"
          ),
          jnp.float64("1.2811506425458971407576473502960132503278731843202e21"),
          jnp.float64(
              "-4.8377785592478791042818441042172984661237423209220e19"
          ),
          jnp.float64("1.4950950070246987188911251393052266880463169608818e18"),
          jnp.float64(
              "-3.7998716490318718336894856855981760254918364941421e16"
          ),
          jnp.float64("7.9288480004879669667266200942871721853771857574653e14"),
          jnp.float64(
              "-1.3463835061976317603904519846550596617283557169419e13"
          ),
          jnp.float64("1.8274559458096926772559994812299276512451729119637e11"),
          jnp.float64("-1.9206517000057456044925962664283364229033432188352e9"),
          jnp.float64("1.4774510404652744904057762482160531565929764279171e7"),
          jnp.float64("-74506.734911001449184743190367420022952796123754747"),
          jnp.float64("186.07469137747439509832699281553814014533506496989"),
      ],
  )
  return jnp.divide(numerator, denominator)


def exp64_rational_pade_17_17(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns the (17, 17)th order Pade approximant about x=0.

  N[HornerForm[PadeApproximant[2^x, {x, 1/2, {17, 17}}]], 50]

  Args:
    x: the argument of the pade appriximant polynomial
  """

  numerator = _horner_scheme(
      x[0],
      [
          jnp.float64(
              "-1.2932006792606928015969529672157750269529359655542e29"
          ),
          jnp.float64(
              "-4.5054262633095629283843160384171961020783838396294e28"
          ),
          jnp.float64(
              "-7.6129760701014740116025984727150850319865478653354e27"
          ),
          jnp.float64(
              "-8.2941804445480490852584111256199190331906903177378e26"
          ),
          jnp.float64(
              "-6.5320730130892479781328948298490325426009779215981e25"
          ),
          jnp.float64(
              "-3.9506129885502533158049617452163713881289685382379e24"
          ),
          jnp.float64(
              "-1.9022542294614916940225641608235029982836639370539e23"
          ),
          jnp.float64(
              "-7.4578146711035631713878513082240786911970127612116e21"
          ),
          jnp.float64(
              "-2.4133811259147739334190225602426899569207218054978e20"
          ),
          jnp.float64(
              "-6.4925314888598400402425980375133865345615274200280e18"
          ),
          jnp.float64(
              "-1.4543070843705627157527654415559666985057777371647e17"
          ),
          jnp.float64(
              "-2.7015725079042522305479440632344893088189438849777e15"
          ),
          jnp.float64(
              "-4.1186167746529679837888929582638755749558613578258e13"
          ),
          jnp.float64(
              "-5.0551743410028682539515107022845556983791172275797e11"
          ),
          jnp.float64("-4.8349560693001479132128621342366224650049593085891e9"),
          jnp.float64("-3.4040052197806953200676028953536242585121752809619e7"),
          jnp.float64("-157946.58258639750478934308505858614469421162188442"),
          jnp.float64("-364.80246303233187332616182356230421902658787815962"),
      ],
  )
  denominator = _horner_scheme(
      x[0],
      [
          jnp.float64(
              "-1.2932006792606928015969529672157750269529359655542e29"
          ),
          jnp.float64("4.4583577839679906192199922719990264252141589424882e28"),
          jnp.float64(
              "-7.4498491513571996261423260727646430929877114017858e27"
          ),
          jnp.float64("8.0206182095052320554793867837707339336246678003120e26"),
          jnp.float64(
              "-6.2371020514668542940472512809181744692112902175777e25"
          ),
          jnp.float64("3.7214680282845668371848139461371631705618634256983e24"),
          jnp.float64(
              "-1.7661037880009780903309034952586404000597444590808e23"
          ),
          jnp.float64("6.8169530186970611395678486705637849784734569699942e21"),
          jnp.float64(
              "-2.1692769606475299118999383138278456980382957719640e20"
          ),
          jnp.float64("5.7309995678466252337187149206742074698073805575505e18"),
          jnp.float64(
              "-1.2587618949711071648211083082995428692823217028257e17"
          ),
          jnp.float64("2.2889357378220245425568645174826697450602083891442e15"),
          jnp.float64(
              "-3.4092201219710860898629895047828910569174858065693e13"
          ),
          jnp.float64("4.0790893130692535614797099670511276143892908491590e11"),
          jnp.float64("-3.7934571203307141160301531393813955844883800350515e9"),
          jnp.float64("2.5891995118981566681926900129663241581972747108900e7"),
          jnp.float64("-116070.32263394592163326179948411717911958232087199"),
          jnp.float64("257.95429540371668624537410364798086478085424828402"),
      ],
  )
  return jnp.divide(numerator, denominator)


################################################################################
# 64-bit baselines for f=erf01(x) on [0, 1]. ###################################
################################################################################


def table_erf_64bits_0to1_baselines() -> List[Tuple[str, FinalizedFn]]:
  """Baselines for 64-bit log2(x) on (1, 2].

  Returns:
    A list of (name, finalized_fn).
  """
  return [("JAX Numpy", erf64_0_1_jscipy)]  # pytype: disable=bad-return-type  # jax-ndarray


def erf64_0_1_jscipy(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns JAX NumPy approximation."""
  return jax.scipy.special.erf(x[0])


################################################################################
# 64-bit baselines for f=leading_asymptotic * erf17(x) on [1, 7]. ##############
################################################################################


def table_erf_64bits_1to7_baselines() -> List[Tuple[str, FinalizedFn]]:
  """Baselines for 64-bit log2(x) on (1, 2].

  Returns:
    A list of (name, finalized_fn).
  """
  return [("JAX Numpy", erf64_1_7_jscipy)]  # pytype: disable=bad-return-type  # jax-ndarray


def erf64_1_7_jscipy(x: List[jnp.ndarray]) -> jnp.ndarray:
  """Returns JAX NumPy approximation."""
  return jax.scipy.special.erf(x[0])


################################################################################
# Internal. ####################################################################
################################################################################


def _horner_scheme(
    x: jnp.ndarray, coeffs: List[Optional[JnpFloat]]
) -> jnp.ndarray:
  """Produces polynomial in Horner's scheme from input and coefficients."""
  poly = coeffs.pop()
  if poly is None:
    raise NotImplementedError(
        "Case for empty first coefficient not implemented."
    )
  for c in reversed(coeffs):
    if c is None:
      poly = jnp.multiply(poly, x)
    else:
      poly = jnp.add(c, jnp.multiply(poly, x))
  return poly
