#!/bin/bash
#
# run_dynamic_system.sh — Run benchmarks with the DynamicSystem strategy.
#
# This script mirrors the structure of run.sh but uses the new
# dynamic_system strategy for variance-driven pool sizing.
# Output data is written to the same CSV format so results can be
# directly compared with runs produced by run.sh.
#
# Usage:
#   ./run_dynamic_system.sh [basic|suite|evaluation]
#

if [ "$1" == "basic" ]; then
  pypy_functions="bfs mst"
  python3 synthetic_run_dynamic_system.py 500 200 pypy basic $pypy_functions
elif [ "$1" == "suite" ]; then
  jvm_functions="matrix-multiplication simple-hash word-count html-rendering"
  python3 synthetic_run_dynamic_system.py 500 100 jvm suite $jvm_functions
elif [ "$1" == "evaluation" ]; then
  pypy_functions="bfs dfs dynamic-html mst pagerank compress upload thumbnail video"
  python3 synthetic_run_dynamic_system.py 500 200 pypy evaluation $pypy_functions

  jvm_functions="matrix-multiplication simple-hash word-count html-rendering"
  python3 synthetic_run_dynamic_system.py 500 100 jvm evaluation $jvm_functions

  # Remove the state store to avoid conflicts with the next run
  cd ./database
  make redeploy-k8s

  # Remove the old checkpoints
  mc rb myminio/checkpoints --force
else
  echo "Usage: $0 [basic|suite|evaluation]"
  exit 1
fi
