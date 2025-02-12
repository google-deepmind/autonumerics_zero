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

# pylint: disable=line-too-long
r"""Worker code.

See comment in BUILD file for context.

Whether locally or on remotely, the worker(s) use the GraphEvolver class to
evolve a
population of graphs representing mathematical functions, as they search for
the correct Taylor expansion.

To run the experiment locally, see example in comment at the top of
local_config_simple.textproto.
"""
# pylint: enable=line-too-long

from absl import app
from absl import flags
from absl import logging  # pylint: disable=unused-import
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from evolution.lib.python import experiment_spec_util
from evolution.lib.python import printing
from evolution.lib.python import rng as rng_lib
from evolution.projects.graphs import graph_evolver
from evolution.projects.graphs.joy import custom_mutators
from evolution.projects.graphs.joy import evaluator as evaluator_lib

# Pointing to your project's worker spec proto is necessary to decode the
# proto extension.
from evolution.projects.graphs.joy import joy_worker_spec_pb2  # pylint: disable=unused-import

from evolution.projects.graphs.joy import search_space  # pylint: disable=unused-import
from evolution.projects.graphs.joy import simple_hasher

print_now = printing.print_now

NANOS_PER_SECOND = 1000000000

_WORKER = flags.DEFINE_string(
    "worker", "", "A text-format ProjectWorkerSpec proto."
)

_RNG_SEED = flags.DEFINE_integer(
    "rng_seed",
    -1,
    "A positive seed for the random number generator. If negative, a new seed "
    "will be generated automatically each time.",
)


def run():
  """Runs an experiment."""
  (
      spec,
      population_client_manager_spec,
      _,
  ) = experiment_spec_util.unpack_worker_spec(_WORKER.value)

  jax.config.update("jax_enable_x64", True)
  float_value = random.uniform(random.PRNGKey(0), (1000,), dtype=jnp.float64)
  print("Using precision: %s" % str(float_value.dtype))

  rng = rng_lib.RNG(
      rng_lib.GenerateRNGSeed() if _RNG_SEED.value <= 0 else _RNG_SEED.value
  )
  np_rng = np.random.RandomState(rng.UniformRNGSeed())

  if not spec.HasField("evaluator"):
    raise ValueError("Missing required evaluator in config.")
  evaluator = evaluator_lib.JoyEvaluator(spec.evaluator, rng.UniformRNGSeed())

  if spec.HasField("simple_hasher"):
    hasher = simple_hasher.SimpleHasher(spec.simple_hasher)
  else:
    hasher = None

  if spec.HasField("custom_mutator"):
    custom_mutator = custom_mutators.build_custom_mutator(
        spec=spec.custom_mutator,
        rng=np_rng,
        hasher=hasher,
        saver=None,
        op_init_params=None,
    )
  else:
    custom_mutator = None

  evolver = graph_evolver.GraphEvolver(
      spec=spec.evolver,
      population_client_manager_spec=population_client_manager_spec,
      worker_id=0,
      evaluator=evaluator,
      hasher=hasher,
      rng_seed=rng.UniformRNGSeed(),
      op_init_params=None,
      custom_mutator=custom_mutator,
  )
  evolver.evolve()


def check(spec):
  assert spec.HasField("evolver")
  assert spec.HasField("evaluator")


def optional(proto, field_str):
  if proto.HasField(field_str):
    return getattr(proto, field_str)
  else:
    return None


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  run()


if __name__ == "__main__":
  app.run(main)
