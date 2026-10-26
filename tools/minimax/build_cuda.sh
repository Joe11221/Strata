#!/usr/bin/env bash
# Small developer build of the exact two translation units used by the prototype.
# CUDA 12 is required for a P100; no serving container is modified.
set -euo pipefail
cd "$(dirname "$0")/../.."
out="${1:-build-minimax}"
arch="${CUDA_ARCH:-60}"
mkdir -p "$out"
nvcc -O2 -std=c++20 -arch="sm_$arch" -DSTRATA_Q6K_EXPERTS=1 -Iinclude -Ithird_party/ggml \
  -c src/kernels/cuda/iq_kernels.cu -o "$out/iq.o"
nvcc -O2 -std=c++20 -arch="sm_$arch" -Iinclude src/minimax/engine.cu "$out/iq.o" -lcublas \
  -o "$out/strata-minimax"
