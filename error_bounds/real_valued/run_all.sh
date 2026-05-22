#!/bin/bash
# Compile, run, and time all IBEX-based error bound proof programs (f2-f10).
#
# Usage:
#   ./run_all.sh                  # uses default IBEX_PREFIX
#   IBEX_PREFIX=/path/to ./run_all.sh  # custom IBEX location

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

: "${IBEX_PREFIX:=/tmp/ibex-install/tmp/ibex-install}"

echo "================================================================"
echo "1. Compiling all proof programs..."
echo "================================================================"
start_compile=$(date +%s)
./build_all.sh
end_compile=$(date +%s)
echo "Compilation completed in $((end_compile - start_compile)) seconds."

echo "================================================================"
echo "2. Running and timing error bound proofs..."
echo "================================================================"

total_start=$(date +%s)

run_proof() {
    local name="$1"
    local cmd="./$name"
    
    echo ""
    echo "----------------------------------------------------------------"
    echo "Running ${name}..."
    echo "----------------------------------------------------------------"
    
    local t_start=$(date +%s)
    $cmd
    local t_end=$(date +%s)
    local t_elapsed=$((t_end - t_start))
    
    echo "--> Finished in ${t_elapsed} seconds."
}

run_proof "prove_mre_f2_exp2"
run_proof "prove_mre_f3_exp2"
run_proof "prove_mre_f4_exp2"
run_proof "prove_mre_f5_exp2"
run_proof "prove_mre_f6_exp2"
run_proof "prove_mre_f7_exp2"
run_proof "prove_mre_f8_exp2"
run_proof "prove_mre_f9_exp2"
run_proof "prove_mre_f10_exp2"

total_end=$(date +%s)
total_elapsed=$((total_end - total_start))

echo ""
echo "================================================================"
echo "All proofs completed successfully in ${total_elapsed} seconds!"
echo "================================================================"
