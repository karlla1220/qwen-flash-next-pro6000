# 目标镜像: Flash-Next(SSD Stream)+ Qwen3.8-27B(DFlash2) 二合一
#
# 基座: 现有 lmsysorg/sglang:qwen38-27b-dflash2 (main 41c018a9e, CUDA 13.0.3)
# 变更:
#   1) sglang 换成 ssd-stream 官方验证的 pin commit 3df8e1e7 (含 Qwen3.8 Flash Next / qwen4_exp)
#   2) flashinfer 全家对齐 0.6.17 (官方 install.sh 同款)
#   3) 装 sglang-ssd-stream v0.1.0 wheel (entry point: sglang.srt.plugins)
#   4) DFlash2 回移: 5 个文件用 main(41c018a9e) 版本覆盖
#      (已逐行验证: 两 commit 间差异恰好 = DFlash2 那一个 commit, 无冲突)
#   5) 本地量化/KV-scale 补丁
#      - auto_round.py / gptq.py / compressed_tensors.py: 与 pin 版零漂移, 直接覆盖
#      - qwen3_5.py: pin 版 + 重新套上的 KV-scale patch

FROM lmsysorg/sglang:qwen38-27b-dflash2

ARG SGLANG_COMMIT=3df8e1e7dbc5807696622afe2929b6c33c185ca3
ARG SGLANG_MAIN_REF=41c018a9ec
ENV SGLANG_BUILD_RUST_EXTS=none
ENV SGLANG_PKG=/usr/local/lib/python3.12/dist-packages

# ---- 1) sglang 降级到 pin commit + 官方同款依赖版本 ----
RUN pip install --break-system-packages \
    "sglang @ https://github.com/sgl-project/sglang/archive/${SGLANG_COMMIT}.tar.gz#subdirectory=python" \
    "flashinfer-python==0.6.17" \
    "nvidia-modelopt==0.45.0" \
    "nvidia-cuda-cccl==13.0.85" \
    "nvidia-cuda-crt==13.0.88" \
    "nvidia-cuda-nvcc==13.0.88" \
    "nvidia-cuda-nvrtc==13.0.88" \
    "nvidia-cuda-runtime==13.0.96" \
    "nvidia-nvjitlink==13.0.88" \
    "nvidia-nvvm==13.0.88" \
 && pip install --break-system-packages --no-deps --index https://flashinfer.ai/whl "flashinfer-cubin==0.6.17" \
 && pip install --break-system-packages --no-deps --index https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache==0.6.17+cu130"

# ---- 2) SSD Stream 插件 (预编译 wheel, cp312 manylinux x86_64) ----
RUN pip install --break-system-packages \
    "https://github.com/garnermccloud/sglang-ssd-stream/releases/download/v0.1.0/sglang_ssd_stream-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "huggingface-hub>=0.36,<2"

# ---- 3) DFlash2 回移: 5 个文件覆盖为 main 版本 ----
RUN set -x; \
    for f in \
      "python/sglang/kernels/ops/speculative/dflash.py" \
      "python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py" \
      "python/sglang/srt/models/dflash.py" \
      "python/sglang/srt/speculative/dflash_utils.py" \
      "python/sglang/srt/speculative/dflash_worker_v2.py"; do \
      curl -sSL "https://raw.githubusercontent.com/sgl-project/sglang/${SGLANG_MAIN_REF}/${f}" \
        -o "${SGLANG_PKG}/${f#python/}"; \
    done; \
    python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('${SGLANG_PKG}/sglang/srt/speculative/dflash_*.py') + glob.glob('${SGLANG_PKG}/sglang/srt/models/dflash.py')]; print('DFlash2 files syntax OK')"

# ---- 4) 本地补丁 (量化 INT8-AutoRound / FP8-KV scale) ----
COPY patches/auto_round.py              ${SGLANG_PKG}/sglang/srt/layers/quantization/auto_round.py
COPY patches/gptq.py                    ${SGLANG_PKG}/sglang/srt/layers/quantization/gptq/gptq.py
COPY patches/compressed_tensors.py      ${SGLANG_PKG}/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py
COPY patches/qwen3_5.py                 ${SGLANG_PKG}/sglang/srt/models/qwen3_5.py
COPY patches/eagle_worker_v2.py         ${SGLANG_PKG}/sglang/srt/speculative/eagle_worker_v2.py
COPY patches/modelopt_quant.py.in                ${SGLANG_PKG}/sglang/srt/layers/quantization/modelopt_quant.py
COPY patches/mamba_state_scatter_triton.py.in       ${SGLANG_PKG}/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py
COPY patches/mamba_radix_cache.py.in                     ${SGLANG_PKG}/sglang/srt/mem_cache/mamba_radix_cache.py
COPY patches/spec_utils.py.in                ${SGLANG_PKG}/sglang/srt/speculative/spec_utils.py
RUN python -c "import ast; [ast.parse(open(f).read()) for f in [\
  '${SGLANG_PKG}/sglang/srt/layers/quantization/auto_round.py', \
  '${SGLANG_PKG}/sglang/srt/layers/quantization/gptq/gptq.py', \
  '${SGLANG_PKG}/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py', \
  '${SGLANG_PKG}/sglang/srt/models/qwen3_5.py', \
  '${SGLANG_PKG}/sglang/srt/speculative/eagle_worker_v2.py', \
 '${SGLANG_PKG}/sglang/srt/layers/quantization/modelopt_quant.py', \
  '${SGLANG_PKG}/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py', \
  '${SGLANG_PKG}/sglang/srt/mem_cache/mamba_radix_cache.py', \
  '${SGLANG_PKG}/sglang/srt/speculative/spec_utils.py']]; print('local patches syntax OK')"

# ---- 5) 共存补丁: 无 ssd-stream.json 的模型(如 27b)跳过 SSD Stream 正常服务 ----
COPY ssd_stream_patch.py /tmp/ssd_stream_patch.py
RUN python /tmp/ssd_stream_patch.py && rm /tmp/ssd_stream_patch.py \
 && python -c "import ast; ast.parse(open('/usr/local/lib/python3.12/dist-packages/sglang_ssd_stream/config.py').read()); print('config.py syntax OK')"
