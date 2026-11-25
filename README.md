# Radeon 610M as a background-LLM workhorse — llama.cpp benchmark

**Purpose:** answer one practical question — *can a tiny AMD iGPU (the Radeon 610M baked into
Ryzen 9000 desktop chips) take over a small background LLM from the CPU, when the constraint is
not speed but CPU load, CPU temperature, and motherboard VRM thermals?*

## Motivation

The case: an agent-memory pipeline with no cloud LLMs. **Hindsight** runs locally as
the external memory provider for the Hermes Agent; its retain and consolidation phases drive a
small local LLM over captured conversations to distill them into recallable memory units. These
are batch jobs — heavy prefill (consolidation prompts run 10k+ tokens), latency-tolerant by
design, meant to churn away in parallel with everything else.

And it is not a Hermes-specific curiosity: Hindsight ships official integrations for
**OpenClaw** (a memory-provider plugin that replaces its built-in memory), Claude Code,
LangGraph/LangChain, CrewAI, the LiteLLM wrapper, and dozens more — plus an MCP endpoint for
any client. Any harness wired to it inherits the same background LLM, and the same question of
where to seat it.

Three ways to seat the memory LLM (Gemma E2B `Q4_K_M`, llama.cpp, `--threads 12`):

- **Option A — share the chat model.** The chat model could run the memory pipeline itself, but
  retention fires after every message and consolidation every few minutes — plus Hermes' own
  memory updates and thread summarization — so the shared endpoint is busy with background work
  precisely when the user expects an answer. And because the two workloads share no prompt
  prefix, every context switch between chat and memory processing drops the KV cache and re-runs
  a prefill of thousands of tokens from scratch.
- **Option B — dedicated sidecar on the CPU** (the current setup). It works, until the cost
  shows up: CPU usage at 80–90%, the board's VRM over-temp alarm trips, and the system bell
  starts beeping.
- **Option C — dedicated sidecar on the iGPU.** The obvious GPU — a large NVIDIA dGPU — is
  already fully committed serving the main chat, agentic, and automation models; there is no
  VRAM left to seat Gemma. That leaves the one "GPU" nobody counts as a GPU: the 2-CU Radeon
  610M. This repo measures **B vs C** head-to-head.
- **Option D — a cloud LLM.** Operationally the easiest, and what the default configs assume,
  but this pipeline reads the entire conversation to build memory — the whole point is that it
  stays local. Cloud is out by design; noted only so the option table is complete.

## Architecture at a glance

The four seating options from the Motivation — D is out by design (local-first). The benchmark
measures Option B against Option C. Note both sidecar options reach memory through the *same*
DIMMs — the iGPU has none of its own.

```mermaid
flowchart TB
    Hermes["Hermes Agent<br/>chat · agentic · automation"]

    subgraph dGPU["Discrete NVIDIA GPU"]
        CHAT["Option A — share the chat model:<br/>busy with background work when users wait;<br/>KV dropped & re-prefilled on every switch"]
        VRAM["VRAM — fully allocated<br/>no room to seat the memory LLM"]
        CHAT --- VRAM
    end

    subgraph MEM["Hindsight — local memory provider"]
        API["API + Postgres/pgvector<br/>retain · consolidate · recall"]
        SIDECAR["Gemma E2B sidecar<br/>llama.cpp server"]
        API -->|"LLM calls"| SIDECAR
    end

    subgraph CPU["CPU · 16-core Zen 5"]
        CORES["Option B — CURRENT: sidecar on 12 threads<br/>80–90% core load → VRM over-temp alarm"]
    end

    subgraph IGPU["Radeon 610M iGPU · 2 CUs · no VRAM"]
        TARGET["Option C — PROPOSED: sidecar via Vulkan<br/>full offload · ~0.4 CPU cores"]
    end

    RAM["System RAM · dual-channel DDR5<br/>every weight & KV cache lives here<br/>iGPU reaches it through a GTT window"]

    CLOUD["Option D — cloud LLM<br/>out by design: memory reads<br/>the whole conversation; staying local"]

    Hermes -->|"chat & tool calls"| CHAT
    Hermes -->|"retain after every message<br/>consolidation every few min"| API
    SIDECAR -.->|"Option B"| CORES
    SIDECAR -.->|"Option C: 2× slower, ~5% core load"| TARGET
    API -.->|"Option A"| CHAT
    API -.->|"Option D"| CLOUD
    CORES --> RAM
    TARGET -->|"same DIMMs"| RAM

    style CORES fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    style TARGET fill:#dcfce7,stroke:#16a34a,color:#14532d
    style CHAT fill:#fef9c3,stroke:#ca8a04,color:#713f12
    style RAM fill:#f3e8ff,stroke:#9333ea,color:#581c87
    style CLOUD fill:#f1f5f9,stroke:#94a3b8,color:#475569
```

If your workload is fire-and-forget and your bottleneck is *core current through the VRM*,
this repo measures whether the iGPU is a real option — with the gotchas that make or break
the experiment.

## Results (measured 2026-11-24)

Hardware: 16-core Zen 5 desktop CPU + Radeon 610M iGPU (gfx1103, 2 CUs, no VRAM)
+ dual-channel DDR5. Model: `gemma-4-E2B-it-Q4_K_M.gguf` (3.1 GB, dense in this GGUF —
`n_expert = 0`). llama.cpp b12627 (same build as the production sidecar container), Vulkan
backend from the official ggml release. Same binary, same prompt, `--temp 0`, ctx 8192:

| Run | Prompt eval | **Generation** | CPU cost during decode |
| --- | --- | --- | --- |
| CPU, `-ngl 0`, 12 threads | 689 t/s | **36.7 t/s** | 12 cores at boost clocks → VRM alarm |
| iGPU, `--device Vulkan1 -ngl 99` | 206 t/s | **17.2 t/s** | **~0.4 of one core**, 610M at 100% busy |
| iGPU, `-ngl 15` (half offload) | 188 t/s | 20.9 t/s | both partially hot — worst design |

Offload verified real, not a silent CPU fallback: `-v` log shows
`using device Vulkan1 (…RADV RAPHAEL_MENDOCINO…)` with every layer assigned to it,
and `/sys/class/drm/card1/device/gpu_busy_percent` reads 100% during decode while the llama-cli
process itself stays at ~42% of one core.

## Insights (the parts worth stealing)

1. **The naive FLOPs-ratio prediction is badly wrong.** The 610M has ~1/7 of the CPU's FP32
   throughput, and theory said "~6× slower". Measured: only **2.1× slower** on decode. Both
   paths are memory-bandwidth-bound on the *same* DIMMs (the iGPU is UMA — no VRAM, weights
   stream through the same dual-channel DDR5), so the GPU's weak compute matters far less than
   the FLOPs ratio suggests. A "weak" iGPU can hold ~50% of a 16-core Zen 5's decode speed.
2. **For VRM/thermal relief this is a genuine win: ~47% of the speed for <5% of the core load.**
   The CPU-side cost drops from "12 cores sustained at boost" (tens of amps of Vcore current →
   VRM alarm) to a single fraction of one core babysitting Vulkan command buffers. A 35 s
   consolidation batch becomes ~70 s — irrelevant for background work, priceless for thermals.
3. **Prompt eval (prefill) is the part that actually suffers** — 3.4× slower (689 → 206 t/s).
   Decode-dominated workloads benefit most; prefill-dominated ones less so.
4. **Partial offload (`-ngl` ≈ half the layers) is the worst of both worlds** — both CPU and GPU
   partially busy, no clean thermal story, only 20.9 t/s. Bench `-ngl 0`, full, and half as
   separate runs; ship full or none.
5. **Vulkan is the only sane backend for gfx1103.** ROCm has no official support for the 610M
   (`gfx1103`); RADV + llama.cpp Vulkan works out of the box. Caveat: `matrix cores: none` —
   no tensor-core acceleration, pure shader math.
6. **Device targeting is a trap.** `--list-devices` usually enumerates the *discrete* NVIDIA GPU
   as `Vulkan0` — an accidental "iGPU" benchmark silently runs on the gaming GPU (on the test
   machine the dGPU had only a few hundred MiB free, already fully occupied by the chat model). Worse,
   `--device "<full device name>"` is **rejected** even when copied verbatim from
   `--list-devices`; use the index form (`--device Vulkan1`). Guard with
   `GGML_VK_DISABLE_NVIDIA=1` as belt-and-braces.
7. **Verify the number is real.** gpu_busy% of 100 + near-zero process CPU during decode is the
   signature of a genuine iGPU decode; anything else and you may be timing a CPU fallback.

## Post-mortem: production reality one day later (2026-11-25)

We deployed Option C — the Vulkan container in this repo, `--device Vulkan0
--n-gpu-layers 99` — as the live Hindsight sidecar. **It was reverted to CPU the
same day.** The benchmark was executed correctly and its numbers replicated
(smoke test: 16.8 t/s decode); what was wrong was *which number we made the
decision on*:

1. **We led with the decode ratio; the workload is prefill-bound.** A Hindsight
   retain/consolidation call runs a 5k–16k-token prompt producing a few hundred
   tokens of JSON — roughly 85% of its GPU time is prompt eval. For that mix the
   relevant ratio was never decode's 2.1×, it was prefill — and production
   prefill on the 610M was far worse than the bench's "only" 3.4×.
2. **The bench's prefill number flattered the iGPU.** Short prompt chunks at
   ctx 8192, GPU otherwise idle, no concurrency. The real server ran long
   prompts against live KV pressure (GTT allocation churn, multiple queued
   requests, and RADV's Flash-Attention fallback that executes FA's K/V pass on
   CPU). A/B on the *identical* 3,775-token cold prompt, same model, same
   llama.cpp build: **CPU prefill 1,662 t/s vs iGPU ≈ 30 t/s effective** — the
   same consolidation call took **215 s** on the iGPU vs an estimated ~25 s on
   the CPU. From the queue's side this looked like "consolidation enters the
   queue and nothing completes": it was one 20-minute GPU round blocking every
   retain behind it, one slot deep.
3. **The thermal premise didn't hold either.** The VRM over-temp alarm *persisted*
   with the sidecar on the iGPU and the cores idle — the 610M's graphics block
   lives inside the CPU package and its steady 100%-busy current feeds the same
   power stages and the same board sensor the Vcore rail drives. Moving compute
   from the cores to the iGPU moved current between neighboring rails; it did
   not take it off the sensor.

**Lessons for anyone reusing this bench:** match the benchmark's *workload
shape*, not just the model — for agentic/memory pipelines that means long-prompt
completions measured at production context sizes with queued requests; report
prefill and decode as first-class results and pick the ratio that matches your
call's token mix; on APUs, verify what the board sensor actually sums before
assuming GPU offload is thermal relief. The iGPU result stands for what it
measures — *decode* at half CPU speed for <5% of the cores — and an
appropriately-shaped workload (decode-heavy, truly fire-and-forget) may still
want it. Ours wasn't.

## Setup

Requirements: Linux, Mesa/Vulkan (RADV), `/dev/dri/renderD*` accessible, a GGUF a few GB or smaller.

1. **Get a Vulkan-enabled llama.cpp build.** Download the official release asset
   `llama-b<NNNN>-bin-ubuntu-vulkan-x64.tar.gz` from
   [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and extract it
   so the binaries sit in `./llama-b<NNNN>/` next to this README (flat layout — the archive has
   no `bin/` subdir; run with `LD_LIBRARY_PATH=.`, which the script does). Pick the **same build
   number as the deployment you are modeling** (`docker exec <sidecar> /app/llama-server --version`)
   so the CPU baseline is comparable.

2. **Confirm the iGPU is enumerable:**

   ```bash
   LD_LIBRARY_PATH=./llama-b12627 ./llama-b12627/llama-cli --list-devices
   # Vulkan0: NVIDIA GeForce RTX ...            <- NOT the target
   # Vulkan1: AMD Ryzen ... (RADV RAPHAEL_MENDOCINO) ...  <- the 610M
   ```

   Note the index name of the AMD device; that is your `IGPU_DEVICE`. (If you only have the
   iGPU, it will be `Vulkan0`.)

3. **Point the script at your model** and run:

   ```bash
   MODEL=/path/to/model-Q4_K_M.gguf ./gemma-610m-vs-cpu.sh 128
   # knobs (env): MODEL, BENCH_DIR, IGPU_DEVICE, THREADS, CTX
   ```

   It runs three passes (CPU / full iGPU offload / half offload) with the same prompt and
   `--temp 0`, and prints prompt-eval and generation t/s per run. Full logs land in
   `/tmp/bench_{cpu,igpu,igpu_half}.txt`.

4. **Optional — verify the offload is genuine** during an iGPU run:

   ```bash
   watch -n1 cat /sys/class/drm/card1/device/gpu_busy_percent   # card1 = iGPU (check device/vendor == 0x1002)
   ```

   100% busy with near-idle process CPU = genuinely on the iGPU.

## Interpreting for your box

- This GGUF ran with all layers resident in the iGPU's GTT aperture (a large slice of system
  RAM was visible to it at rest; allocations >4 GB may need the BIOS "GTT size" raised).
- The trade only makes sense for **background/batch** workloads where wall-clock ×2 is free and
  CPU cores/VRM power are the scarce resource. Anything interactive should stay on the CPU (or a
  real GPU).
- On AMD APUs one class up (Radeon 780M/890M, 12–16 CUs), the iGPU can *beat* the CPU path —
  the conclusions here are for the 2-CU bottom-shelf part, the hardest case.

## Repo layout

- `gemma-610m-vs-cpu.sh` — the benchmark (defaults tuned for the author's box; override via env).
- `Dockerfile` — the production container we evaluated: pinned official Vulkan build +
  Mesa RADV inside, iGPU-only by construction (no NVIDIA ICD in the image; pass a single
  `/dev/dri/renderD*` at runtime). Read the Post-mortem before deploying it.
- `llama-b12627/` — not committed; download per Setup step 1.
