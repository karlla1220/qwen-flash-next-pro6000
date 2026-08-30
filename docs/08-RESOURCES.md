# 08 — External links (everything this playbook was built against)

## SGLang (engine)
- Pinned commit (serves our whole stack): https://github.com/sgl-project/sglang/commit/3df8e1e7dbc5807696622afe2929b6c33c185ca3
- Source tarball used by the Dockerfile: `https://github.com/sgl-project/sglang/archive/3df8e1e7dbc5807696622afe2929b6c33c185ca3.tar.gz#subdirectory=python`
- DFlash2 backport origin (main): commit `41c018a9ec` — raw file template used in Dockerfile layer 3: `https://raw.githubusercontent.com/sgl-project/sglang/41c018a9ec/<path>`
- Base image (CUDA 13.0.3, Blackwell wheels): `lmsysorg/sglang:qwen38-27b-dflash2` — https://hub.docker.com/r/lmsysorg/sglang

## SSD Stream plugin (PLE off-GPU)
- https://github.com/garnermccloud/sglang-ssd-stream — plugin entry point `sglang.srt.plugins`
- wheel used: https://github.com/garnermccloud/sglang-ssd-stream/releases/download/v0.1.0/sglang_ssd_stream-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl

## Models (HF: use https://hf-mirror.com if huggingface.co is slow)
- vendor / unpruned profile: https://huggingface.co/garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream
- pruned profile (448E): https://huggingface.co/lovedheart/Qwen3-8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000
  - its metrics files (`aime26_metrics.json`, `gsm8k_metrics.json`, `qualification-notes.md`) are quoted in `07-PRUNED-vs-FULL.md`
- quant origin: https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4
- base model: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- FP8 PLE n-gram shards (used by both checkpoints): https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8
- AIMER pruning method (calibration-free MoE pruning): https://arxiv.org/abs/2603.18492

## Quantization / GEMM stack
- flashinfer index for `flashinfer-cubin==0.6.17`: https://flashinfer.ai/whl
- flashinfer cu130 jit-cache index: https://flashinfer.ai/whl/cu130 (also mirrored in sglang docs: https://docs.sglang.ai/whl/cu130/)
- flashinfer repo (QSA/paged wrapper reference): https://github.com/flashinfer-ai/flashinfer — external issue we hit indirectly: https://github.com/flashinfer-ai/flashinfer/issues/3628

## Upstream items we tracked (all OPEN as of our pin; watch before any pin bump)
- sglang PR #32468 — embed/lm_head freed before KV pool sizing (backported into our `eagle_worker_v2.py`): https://github.com/sgl-project/sglang/pull/32468
- sglang PR #30092 — open at pin date (context: pool-sizing related, see 04/06): https://github.com/sgl-project/sglang/pull/30092
- sglang PR #30119 — open at pin date: https://github.com/sgl-project/sglang/pull/30119
- stability port we applied (from jpezzulli's kit): commit `23e51dd` — see `06-PATCHES.md` (staging: `patches/mamba_radix_cache.py.in` etc.)

## Same-card external references (independent validation)
- validation results (4× decode 427 tok/s, MTP accept 2.58/52.7%): https://github.com/jpezzulli/pennyroyal-validation
- our analysis of its patch kit: `probe/pezz/` + `05-DATA.md §1`
- external write-up with the 512K-context claim: https://msoexpert.com/articles/qwen38-dflash2-rtx-pro-6000/
- NVIDIA TRT-LLM (MoE runner crash we avoided, why we force `flashinfer_cutlass`): https://github.com/NVIDIA/TensorRT-LLM/issues/11799

## Environment / hosting
- Docker Desktop data disk bind root (Windows side, where `/mnt/docker-desktop-disk` lives):
  the WSL2 dist backing Docker Desktop — bind sources in our bats are Linux paths resolved inside that VM; see `04-DEBUG-LOG.md` for the `MSYS_NO_PATHCONV` gotcha when launching from Git Bash
- Triton on-disk cache location: `/root/.triton` (env `TRITON_CACHE_DIR`) — persisted by bats to `/mnt/docker-desktop-disk/triton_cache`
- flashinfer cache: `/root/.cache/flashinfer` — persisted to `/mnt/docker-desktop-disk/jit_cache`
- SGLang docs (server args, speculative decoding): https://docs.sglang.ai

## Links worth re-checking after a pin bump
`#30092`, `#32468`, `#30119` merge status · flashinfer issue #3628 · base-image tag for
`qwen38-27b-dflash2` (our Dockerfile FROM line) · plugin release (we pin v0.1.0).
