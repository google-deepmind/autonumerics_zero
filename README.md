# autonumerics_zero

AutoNumerics-Zero, a method to evolve programs that compute transcendental
functions efficiently—searching from scratch and without knowledge of numerics.

## Usage

Compilation requires Clang.

The package contains an example experiment spec. To test it, run:

```
export CC=clang++
bazel run evolution/projects/graphs/joy:run_worker -- --logtostderr --worker="$(cat evolution/projects/graphs/joy/open_source_num_ops_config.textproto)"
```

## Citing this work

Esteban Real et al., "AutoNumerics-Zero: Automated Discovery of State-of-the-Art
Mathematical Functions."
arXiv preprint arXiv:2312.08472 (2023).

```
@article{ereal2024autonumerics,
  title={AutoNumerics-Zero: Automated Discovery of State-of-the-Art
Mathematical Functions},
  author={Real, Esteban et al.},
  journal={arXiv preprint arXiv:2312.08472},
  year={2023}}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
