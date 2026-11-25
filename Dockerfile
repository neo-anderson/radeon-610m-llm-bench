# llama.cpp with the Vulkan backend (AMD RADV / Radeon 610M iGPU support).
# Official ggml release assets carry no ROCm support for gfx1103 and no
# Vulkan container exists upstream; this pins the proven b12627 Vulkan build.
#
# Build:  docker build -t llama-cpp-vulkan:b12627 .
# Run (Linux host with Mesa/RADV; pass ONLY the iGPU's render node — see README
# Device-targeting notes; find it via /dev/dri/by-path/, the PCI device of the
# CPU package is the APU):
#   docker run -d --device /dev/dri/renderD128 -v /path/to/models:/models:ro \
#     -p 127.0.0.1:8080:8080 llama-cpp-vulkan:b12627 \
#     --model /models/model-Q4_K_M.gguf --ctx-size 24576 --parallel 1 \
#     --device Vulkan0 --n-gpu-layers 99 --flash-attn on --port 8080
#
# Inside the container the AMD APU is the only Vulkan device the image can see
# (no NVIDIA ICD installed), so it is `Vulkan0` there — unlike a bare host
# where the dGPU usually enumerates first.
#
# See README.md "Post-mortem" before deploying this on a prefill-heavy
# workload — we evaluated it in production and reverted to CPU.
ARG UBUNTU=24.04

FROM ubuntu:${UBUNTU} AS fetch
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL -o /llama.tar.gz \
       https://github.com/ggml-org/llama.cpp/releases/download/b12627/llama-b12627-bin-ubuntu-vulkan-x64.tar.gz \
    && tar xzf /llama.tar.gz -C / \
    && mv /llama-b12627 /llama-build

FROM ubuntu:${UBUNTU}
# libvulkan1 = loader; mesa-vulkan-drivers = the RADV driver (libvulkan_radeon.so)
# the container's Vulkan actually runs on. GPU access itself is runtime (--device).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libvulkan1 mesa-vulkan-drivers libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=fetch /llama-build /app
ENV LD_LIBRARY_PATH=/app
ENTRYPOINT ["/app/llama-server"]
