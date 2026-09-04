# 06 — What every source patch does (and what it must NOT be confused with)

**Scope of this doc: `docker-target/Dockerfile` → `sglang-flash-27b:latest` only.**
The second image, `Dockerfile.qsa-fp8-fix` → `sglang-flash-ram:qsa-fp8-fix`
(`docs/09-PRIMITIVE-RAM-OFFLOAD.md`), patches a *different* runtime layout — see
"Second image" at the bottom before copying anything between the two.

Runtime = `/usr/local/lib/python3.12/dist-packages/` (pip install from the pinned tarball, layer 1).
`/sgl-workspace/sglang` is the base-image checkout of main — UNUSED by python. Every patch below
was extracted from dist-packages first (see 01-BUILD.md "Important" note).
**The last sentence is a property of THIS base image (`lmsysorg/sglang:qwen38-27b-dflash2`
+ `pip install sglang@<pin tarball>`), not of every sglang image.** On
`lmsysorg/sglang:qwen38flashnext` the `/sgl-workspace/sglang` checkout IS the live
import path — see below.

Staging files are `patches/<name>.py.in`; `COPY` rewrites them to `.py` at destination. That
extension trick is why the build's syntax-check (last RUN) always parses them before the image ships.

| File in image | Layer / origin | What it fixes | Status notes |
|---|---|---|---|
| `srt/layers/quantization/auto_round.py` | local, zero-drift vs pin | INT8 AutoRound compat | byte-ident to pin; keep for future AutoRound runs |
| `srt/layers/quantization/gptq/gptq.py` | local, zero-drift vs pin | GPTQ compat shim | same as above |
| `srt/layers/quantization/compressed_tensors/compressed_tensors.py` | local, zero-drift vs pin | compressed-tensors loader for the vendored PLE shard path | same |
| `srt/models/qwen3_5.py` | pin + re-applied KV-scale patch | fp8_e4m3 KV with scale 1.0: without this, fp8 KV silently corrupts attention (garbled output) | re-apply the patch on every pin bump |
| `srt/speculative/eagle_worker_v2.py` | backport of upstream #32468 (OPEN) | Frees draft embed/lm_head before KV pool sizing → the freed VRAM goes to the KV pool, not to allocator slack; without it FRACTION=0.99 crashes the allocator instead of unlocking 262K context. Size note: the 2.54 GB/tensor figure (and this crash) is this image's model, vocab 248320 x hidden **5120**; on the RAM-offload profile's checkpoint (hidden 2560) the same recovery is ~1.3 GB and NOT a crash-enabler — see 09 | the enabler for full-context fp8 **on this image** |
| `srt/layers/quantization/modelopt_quant.py` | FP8_PB_WO routing (fp8_pb_wo.diff) | Route FP8_PB_WO checkpoints (no hf_quant_config.json) through modelopt_mixed; stock build garbles silently (`~~~~`) | required for lovedheart; sanity probe must answer `Paris` |
| `srt/kernels/ops/mamba/mamba_state_scatter_triton.py` | ported 23e51dd | empty-checkpoint guard + step-index bounds in the scatter kernel | not on decode hot path; stability only |
| `srt/mem_cache/mamba_radix_cache.py` | ported 23e51dd | empty mamba checkpoint handling + `_get_children` fallback + track-step bounds | crash-hardening |
| `srt/speculative/spec_utils.py` | ported 23e51dd | draft-spec v2 bounds | prevents OOB draft-spec indexing under MTP |
| `sglang_ssd_stream/config.py` | build RUN patch (layer 5) | coexist: model dirs without `ssd-stream.json` (e.g. Qwen3.8-27B) skip the SSD Stream plugin instead of aborting startup | lets one image serve both families |
| `ssd_stream_patch.py` (this repo) | applied by layer 5 | idempotent config rewriter | re-run after any plugin upgrade |

Upstream patches referenced but NOT in the image: #30092, #32468, #30119 all still OPEN upstream at
the pin date. Do not expect merged fixes; track the PRs before any pin bump.

## Second image: `Dockerfile.qsa-fp8-fix` → `sglang-flash-ram:qsa-fp8-fix` (see 09)

Inverted layout, and **none of the 10 patches above belong in it**.

| | `Dockerfile` (this doc) | `Dockerfile.qsa-fp8-fix` (09) |
|---|---|---|
| Base | `lmsysorg/sglang:qwen38-27b-dflash2` | `lmsysorg/sglang:qwen38flashnext` (`593134d17`) |
| sglang from | layer 1 `pip install sglang @ <pin tarball>` | the base image's own `/sgl-workspace/sglang` checkout |
| Live import path | `/usr/local/lib/python3.12/dist-packages/sglang/` | `/sgl-workspace/sglang/python/sglang/` |
| Patch count | 9 source files + 2 SSD-Stream config edits | 3 single-file overlays |
| Build | `cd docker-target && docker build -t sglang-flash-27b:latest .` | `cd docker-target && docker build -t sglang-flash-ram:qsa-fp8-fix -f Dockerfile.qsa-fp8-fix .` |

In the other direction: the #32468 backport now lives on BOTH images (old tree →
`patches/eagle_worker_v2.py`, new tree → `patches-official/eagle_worker_v2.py`).
Do not copy either file onto the other image — see the eagle row above for the
~170-line drift.

The three overlays (`docker-target/patches-official/`, `COPY`'d straight onto the live path):

| File in image | Layer / origin | What it fixes | Status notes |
|---|---|---|---|
| `srt/layers/attention/qwen_sparse_attn_backend.py` | backport of `sgl-project/sglang@d6ff2d881e78` | casts gathered fp8-e4m3 KV rows to `q.dtype` before `sparse_gqa_fwd_interface_triton_ck`; without it any cached-prefix request under `--kv-cache-dtype fp8_e4m3` dies in Triton (`Unsupported rhs dtype fp8e4nv`) | 1 hunk, 2 code lines |
| `srt/models/qwen4_exp.py` | local (no upstream commit found as of `593134d17`) | builds `VocabParallelEmbedding` under `torch.device("cpu")` when `config.ple_offload_embedding`, killing the transient ~95 GiB GPU spike that OOMs hybrid state-cache sizing | 1 hunk; uses the `device_context()` idiom already in `utils/common.py` |
| `srt/speculative/eagle_worker_v2.py` | re-port of upstream #32468 (still OPEN upstream, verified 2026-09-05 against the PR page: open, 1 commit `a15c74d`, +13−2, unreviewed) | moves draft `init_token_map()`/`init_lm_head()` + `torch.cuda.empty_cache()` from `alloc_memory_pool()` into `EagleDraftWorker.__init__` so the draft's duplicate embed/lm_head are freed BEFORE the KV-pool profile instead of after | 3 code lines + comments, ported onto the new stock (old patch file would regress 5 drifted hunks); **not yet built into the running image, not yet A/B-tested** — see 09 for the measured ~1.3 GB expectation vs the PR's own 2.54 GB-each math |

Verified against the images, not from memory: `docker build` appears only in
`config/run_primitive_ram.sh` (comment), `docs/09` and `README.md` — **there is no
build script for this image**, `run_primitive_ram.sh` assumes the tag already
exists and `docker run` fails if it doesn't. Each Dockerfile layer is
`COPY` + an `ast.parse` **and** `import` check, so a bad overlay fails the build
instead of the first request. Side effect: those import checks bake `__pycache__`
into the image (~80 MB/layer) — visible in `docker history`, harmless.

