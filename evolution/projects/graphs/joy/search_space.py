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

"""A simple search space to use in unit tests.

To each operation is assigned a cost which can be used to evaluate a secondary
objective.
"""

from evolution.projects.graphs.joy import search_space_arithmetic
from evolution.projects.graphs.joy import search_space_base
from evolution.projects.graphs.joy import search_space_constants
from evolution.projects.graphs.joy import search_space_erf
from evolution.projects.graphs.joy import search_space_literals
from evolution.projects.graphs.joy import search_space_special
from evolution.projects.graphs.joy import search_space_variables

# Types.
FloatT = search_space_base.FloatT

# Required.
ProduceXOp = search_space_base.ProduceXOp
ProduceYOp = search_space_base.ProduceYOp
ProduceZOp = search_space_base.ProduceZOp
ConsumeFOp = search_space_base.ConsumeFOp
ConsumeGOp = search_space_base.ConsumeGOp

# Arithmetic.
AddOp = search_space_arithmetic.AddOp
SubOp = search_space_arithmetic.SubOp
MultOp = search_space_arithmetic.MultOp
DivOp = search_space_arithmetic.DivOp
FmaOp = search_space_arithmetic.FmaOp
SqrtOp = search_space_arithmetic.SqrtOp

# Special functions.
ExpOp = search_space_special.ExpOp

# Literals.
ZeroOp = search_space_literals.ZeroOp
OneOp = search_space_literals.OneOp
TwoOp = search_space_literals.TwoOp

# Constants.
ConstantOp = search_space_constants.ConstantOp

# Variables.
ZeroInitVariableOp = search_space_variables.ZeroInitVariableOp
TinyInitVariableOp = search_space_variables.TinyInitVariableOp
RandomInitVariableOp = search_space_variables.RandomInitVariableOp
AnchoredVariableOp = search_space_variables.AnchoredVariableOp
PositiveScale64AnchoredVariableOp = (
    search_space_variables.PositiveScale64AnchoredVariableOp
)
NegativeScale64AnchoredVariableOp = (
    search_space_variables.NegativeScale64AnchoredVariableOp
)
PositiveScale32AnchoredVariableOp = (
    search_space_variables.PositiveScale32AnchoredVariableOp
)
NegativeScale32AnchoredVariableOp = (
    search_space_variables.NegativeScale32AnchoredVariableOp
)
PositiveScale16AnchoredVariableOp = (
    search_space_variables.PositiveScale16AnchoredVariableOp
)
NegativeScale16AnchoredVariableOp = (
    search_space_variables.NegativeScale16AnchoredVariableOp
)
PositiveScale16NonAnchoredVariableOp = (
    search_space_variables.PositiveScale16NonAnchoredVariableOp
)
NegativeScale16NonAnchoredVariableOp = (
    search_space_variables.NegativeScale16NonAnchoredVariableOp
)
PositiveScale8AnchoredVariableOp = (
    search_space_variables.PositiveScale8AnchoredVariableOp
)
NegativeScale8AnchoredVariableOp = (
    search_space_variables.NegativeScale8AnchoredVariableOp
)
PositiveScale8NonAnchoredVariableOp = (
    search_space_variables.PositiveScale8NonAnchoredVariableOp
)
NegativeScale8NonAnchoredVariableOp = (
    search_space_variables.NegativeScale8NonAnchoredVariableOp
)
PositiveScale16LamarckianVariableOp = (
    search_space_variables.PositiveScale16LamarckianVariableOp
)
NegativeScale16LamarckianVariableOp = (
    search_space_variables.NegativeScale16LamarckianVariableOp
)
PositiveScale8LamarckianVariableOp = (
    search_space_variables.PositiveScale8LamarckianVariableOp
)
NegativeScale8LamarckianVariableOp = (
    search_space_variables.NegativeScale8LamarckianVariableOp
)
PositiveScale4AnchoredVariableOp = (
    search_space_variables.PositiveScale4AnchoredVariableOp
)
NegativeScale4AnchoredVariableOp = (
    search_space_variables.NegativeScale4AnchoredVariableOp
)
PositiveScale4NonAnchoredVariableOp = (
    search_space_variables.PositiveScale4NonAnchoredVariableOp
)
NegativeScale4NonAnchoredVariableOp = (
    search_space_variables.NegativeScale4NonAnchoredVariableOp
)
Scale4LogRandomInitVariableOp = (
    search_space_variables.Scale4LogRandomInitVariableOp
)
Scale8LogRandomInitVariableOp = (
    search_space_variables.Scale8LogRandomInitVariableOp
)
Scale16LogRandomInitVariableOp = (
    search_space_variables.Scale16LogRandomInitVariableOp
)

# Task-specific ops.
ErfLeadingOrderOp = search_space_erf.ErfLeadingOrderOp
ErfcAsymptoticBehaviorOp = search_space_erf.ErfcAsymptoticBehaviorOp
