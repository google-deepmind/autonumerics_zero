# Real-Valued Error Bound Proofs

This directory contains C++ programs that use the
[IBEX interval arithmetic library](https://github.com/ibex-team/ibex-lib)
to prove upper bounds on the maximum relative error (MRE) of the evolved
real-valued approximations of 2^x on (0, 1].

## Programs and Results

| File | Function | Ops | Proven MRE Bound | Domain Approach | Status |
|------|----------|-----|------------------|-----------------|--------|
| `prove_mre_f2_exp2.cpp` | f2 | 2 | **$4.138 \times 10^{-2}$** | Direct [0,1] | Verified ✅ |
| `prove_mre_f3_exp2.cpp` | f3 | 3 | **$1.227 \times 10^{-3}$** | Direct [0,1] | Verified ✅ |
| `prove_mre_f4_exp2.cpp` | f4 | 4 | **$3.067 \times 10^{-4}$** | Direct [0,1] | Verified ✅ |
| `prove_mre_f5_exp2.cpp` | f5 | 5 | **$6.339 \times 10^{-6}$** | Split + Lipschitz | Verified ✅ |
| `prove_mre_f6_exp2.cpp` | f6 | 6 | **$3.995 \times 10^{-7}$** | Split + Lipschitz | Verified ✅ |
| `prove_mre_f7_exp2.cpp` | f7 | 7 | **$8.40 \times 10^{-10}$** | Split + Lipschitz | Verified ✅ |
| `prove_mre_f8_exp2.cpp` | f8 | 8 | **$1.360 \times 10^{-11}$** | Adaptive Recursive | Verified ✅ |
| `prove_mre_f9_exp2.cpp` | f9 | 9 | **$2.150 \times 10^{-13}$** | Adaptive Recursive | Verified ✅ |
| `prove_mre_f10_exp2.cpp`| f10| 10| **$6.00 \times 10^{-15}$** | Adaptive Recursive | Verified ✅ |

### Domain Approaches

- **Direct [0,1]** (f2–f4): No $1/x$ terms, so MRE is evaluated directly over $[0, 1]$ using fine interval subdivision.
- **Split + Lipschitz** (f5–f7): Singularity at $x=0$. Evaluates $[0, 0.0001]$ directly, and uses a uniform Lipschitz grid of $10^8$ intervals on $[0.0001, 1]$.
- **Adaptive Recursive** (f8–f10): Uses a dynamic recursive prover that chooses the optimal strategy automatically: direct evaluation near zero ($x < 10^{-9}$), and local Lipschitz bounding away from zero. This achieves extremely high precision with minimal steps.




## Building IBEX from source

There is no apt package for IBEX. Build from source:

```bash
cd /tmp
git clone --depth 1 https://github.com/ibex-team/ibex-lib.git
cd ibex-lib && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/tmp/ibex-install
make -j$(nproc)
make install DESTDIR=/tmp/ibex-install
```

## Compiling all programs

```bash
./build_all.sh
```

Or to compile a single program:

```bash
IBEX_PREFIX=/tmp/ibex-install/tmp/ibex-install

g++ -std=c++17 -O2 \
  -I${IBEX_PREFIX}/include \
  -I${IBEX_PREFIX}/include/ibex \
  -I${IBEX_PREFIX}/include/ibex/3rd \
  prove_mre_f7_exp2.cpp \
  ${IBEX_PREFIX}/lib/libibex.a \
  ${IBEX_PREFIX}/lib/ibex/3rd/libgaol.a \
  ${IBEX_PREFIX}/lib/ibex/3rd/libultim.a \
  -lm -o prove_mre_f7_exp2
```

## Method

Each program computes the relative error function:

    E(x) = (2^x − fN(x)) / 2^x

where fN is the N-operation evolved approximation (see paper Appendix,
Figure "exp_best_programs_sup_fig").

Two strategies are used to bound |E(x)| over (0, 1]:

1. **Direct evaluation** (f2–f4, and near-zero for f5–f10): IBEX evaluates E(x)
   on fine subintervals using interval arithmetic. The bound is the maximum
   of |E| over all subintervals.

2. **Lipschitz-based bound** (f5–f10, away from zero): First bounds |E'(x)|
   via interval arithmetic to obtain a Lipschitz constant L, then uses
   Lipschitz continuity: |E(x)| ≤ max(|E(a)|, |E(b)|) + L·(b−a) on each
   subinterval [a, b].

## Running the Proofs

You can compile and run all nine MRE proofs sequentially by running:

```bash
./run_all.sh
```

This script compiles all programs using `./build_all.sh`, runs them, and outputs their precise execution times.

## Verified Execution Log

Paste the full output of `./run_all.sh` here:

```
================================================================
1. Compiling all proof programs...
================================================================
Compiling prove_mre_f10_exp2.cpp -> prove_mre_f10_exp2
Compiling prove_mre_f2_exp2.cpp -> prove_mre_f2_exp2
Compiling prove_mre_f3_exp2.cpp -> prove_mre_f3_exp2
Compiling prove_mre_f4_exp2.cpp -> prove_mre_f4_exp2
Compiling prove_mre_f5_exp2.cpp -> prove_mre_f5_exp2
Compiling prove_mre_f6_exp2.cpp -> prove_mre_f6_exp2
Compiling prove_mre_f7_exp2.cpp -> prove_mre_f7_exp2
Compiling prove_mre_f8_exp2.cpp -> prove_mre_f8_exp2
Compiling prove_mre_f9_exp2.cpp -> prove_mre_f9_exp2
All programs compiled successfully.
Compilation completed in 15 seconds.
================================================================
2. Running and timing error bound proofs...
================================================================

----------------------------------------------------------------
Running prove_mre_f2_exp2...
----------------------------------------------------------------
The bound on the maximum relative error of f2 on [0,1] is: 0.0413813
--> Finished in 50 seconds.

----------------------------------------------------------------
Running prove_mre_f3_exp2...
----------------------------------------------------------------
The bound on the maximum relative error of f3 on [0,1] is: 0.00122662
--> Finished in 51 seconds.

----------------------------------------------------------------
Running prove_mre_f4_exp2...
----------------------------------------------------------------
The bound on the maximum relative error of f4 on [0,1] is: 0.000306732
--> Finished in 76 seconds.

----------------------------------------------------------------
Running prove_mre_f5_exp2...
----------------------------------------------------------------
The bound on the absolute value of f on [0,0.000100001] is: 6.33858e-06
The bound on the absolute value of df on [0.000100001,1] is: 0.000360737
The bound of the absolute value of f, estimated by Lipschitz, on [0.000100001,1] is: 6.32316e-06
--> Finished in 297 seconds.

----------------------------------------------------------------
Running prove_mre_f6_exp2...
----------------------------------------------------------------
The bound on the absolute value of f on [0,0.000100001] is: 3.99484e-07
The bound on the absolute value of df on [0.000100001,1] is: 0.000216321
The bound of the absolute value of f, estimated by Lipschitz, on [0.000100001,1] is: 3.98512e-07
--> Finished in 313 seconds.

----------------------------------------------------------------
Running prove_mre_f7_exp2...
----------------------------------------------------------------
The bound on the absolute value of f on [0,0.000100001] is: 8.397e-10
The bound on the absolute value of df on [0.000100001,1] is: 0.00020662
The bound of the absolute value of f, estimated by Lipschitz, on [0.000100001,1] is: 8.37837e-10
--> Finished in 317 seconds.

----------------------------------------------------------------
Running prove_mre_f8_exp2...
----------------------------------------------------------------
Proving f8 MRE <= 1.36001e-11 on [0, 1] using unified recursive prover...
SUCCESS! Proved f8 MRE <= 1.36001e-11 on [0, 1] in 2493503 steps.
--> Finished in 5 seconds.

----------------------------------------------------------------
Running prove_mre_f9_exp2...
----------------------------------------------------------------
Proving f9 MRE <= 2.15e-13 on [0, 1] using unified recursive prover...
SUCCESS! Proved f9 MRE <= 2.15e-13 on [0, 1] in 20054811 steps.
--> Finished in 47 seconds.

----------------------------------------------------------------
Running prove_mre_f10_exp2...
----------------------------------------------------------------
Proving f10 MRE <= 6e-15 on [0, 1] using unified recursive prover...
SUCCESS! Proved f10 MRE <= 6e-15 on [0, 1] in 132678423 steps.
--> Finished in 315 seconds.

================================================================
All proofs completed successfully in 1471 seconds!
================================================================

```
