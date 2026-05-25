#!/bin/bash
# Build all IBEX-based error bound proof programs.
# Requires IBEX to be installed; set IBEX_PREFIX to the install prefix.
#
# Usage:
#   ./build_all.sh                  # uses default IBEX_PREFIX
#   IBEX_PREFIX=/path/to ./build_all.sh  # custom IBEX location

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

: "${IBEX_PREFIX:=/tmp/ibex-install/tmp/ibex-install}"

if [ ! -f "${IBEX_PREFIX}/lib/libibex.a" ]; then
    echo "Error: IBEX not found at ${IBEX_PREFIX}"
    echo "Build IBEX from source first (see README.md) or set IBEX_PREFIX."
    exit 1
fi

CXXFLAGS="-std=c++17 -O2"
INCLUDES="-I${IBEX_PREFIX}/include -I${IBEX_PREFIX}/include/ibex -I${IBEX_PREFIX}/include/ibex/3rd"
LIBS="${IBEX_PREFIX}/lib/libibex.a ${IBEX_PREFIX}/lib/ibex/3rd/libgaol.a ${IBEX_PREFIX}/lib/ibex/3rd/libultim.a -lm"

for src in prove_mre_f*_exp2.cpp; do
    out="${src%.cpp}"
    echo "Compiling ${src} -> ${out}"
    g++ ${CXXFLAGS} ${INCLUDES} "${src}" ${LIBS} -o "${out}"
done

echo "All programs compiled successfully."
