#!/usr/bin/env bash
# Benchmark the Hindsight sidecar model (Gemma E2B Q4_K_M) on CPU vs the
# Radeon 610M iGPU (Vulkan/RADV), same binary, same prompt, same settings, temp 0.
#
# Ref. claim: "Vulkan with llama.cpp is the most reliable way to offload layers
# to the 610M" -- true mechanically (ROCm has no gfx1103 support), this script
# measures what it's actually worth.
#
# Usage: ./gemma-610m-vs-cpu.sh [n_predict]   (default 128)
set -euo pipefail

BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/llama-b12627}"   # dir holding llama-cli (see README: Setup)
MODEL="${MODEL:-/path/to/models/gemma-4-E2B-it-Q4_K_M.gguf}"   # any GGUF llama.cpp can load -- set to your own path
IGPU_DEVICE="${IGPU_DEVICE:-Vulkan1}"   # the AMD device (RADV RAPHAEL_MENDOCINO) in --list-devices order; full name is rejected by --device, use the index. NEVER rely on the default pick -- a discrete NVIDIA GPU usually enumerates as Vulkan0.
THREADS="${THREADS:-12}"            # match hindsight-llm sidecar
CTX="${CTX:-8192}"
N_PREDICT="${1:-128}"
PROMPT="Explain in detail why memory bandwidth is the primary bottleneck for autoregressive LLM token generation on consumer hardware, and how this differs from the bottleneck for prefill. Cover weight streaming, arithmetic intensity, batching, and what quantization changes. Be thorough."

COMMON_ARGS=( -m "$MODEL" -c "$CTX" --threads "$THREADS" --temp 0 --flash-attn on --no-context-shift -p "$PROMPT" -n "$N_PREDICT" --no-display-prompt )

run() {  # run <label> <extra args...>
  local label="$1"; shift
  local out="/tmp/bench_${label//[^A-Za-z0-9]/_}.txt"
  echo "=============================================================="
  echo ">>> RUN: $label   (n_predict=$N_PREDICT, threads=$THREADS, ctx=$CTX, temp=0)"
  echo "=============================================================="
  ( cd "$BENCH_DIR" && LD_LIBRARY_PATH=. GGML_VK_DISABLE_NVIDIA=1 ./llama-cli "${COMMON_ARGS[@]}" "$@" </dev/null ) >"$out" 2>&1
  # this build prints a one-line summary: [ Prompt: 10.5 t/s | Generation: 34.3 t/s ]
  sed -n 's/.*Prompt: \([0-9.]*\) t\/s | Generation: \([0-9.]*\) t\/s.*/  prompt eval: \1 t\/s   generation: \2 t\/s/p' "$out" | sed 's/^/  /'
  grep -q 'Generation: .*t/s' "$out" || { echo "  RUN FAILED -- last lines:"; tail -5 "$out" | sed 's/^/    /'; return 1; }
}

# 1) CPU baseline (no offload)
run cpu -ngl 0

# 2) iGPU full offload: index-target the RADV device so the discrete GPU can't be picked
run igpu --device "$IGPU_DEVICE" -ngl 99

# 3) (optional) partial offload sanity check: half the layers
run igpu-half --device "$IGPU_DEVICE" -ngl 15

echo "=============================================================="
echo "Full logs: /tmp/bench_cpu.txt /tmp/bench_igpu.txt /tmp/bench_igpu_half.txt"
echo "Generation speed (tok/s) is the number that matters for consolidation batches."
