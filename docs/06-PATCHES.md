# 06 — What every source patch does (and what it must NOT be confused with)

Runtime = `/usr/local/lib/python3.12/dist-packages/` (pip install from the pinned tarball, layer 1).
`/sgl-workspace/sglang` is the base-image checkout of main — UNUSED by python. Every patch below
was extracted from dist-packages first (see 01-BUILD.md "Important" note).

Staging files are `patches/<name>.py.in`; `COPY` rewrites them to `.py` at destination. That
extension trick is why the build's syntax-check (last RUN) always parses them before the image ships.

| File in image | Layer / origin | What it fixes | Status notes |
|---|---|---|---|
| `srt/layers/quantization/auto_round.py` | local, zero-drift vs pin | INT8 AutoRound compat | byte-ident to pin; keep for future AutoRound runs |
| `srt/layers/quantization/gptq/gptq.py` | local, zero-drift vs pin | GPTQ compat shim | same as above |
| `srt/layers/quantization/compressed_tensors/compressed_tensors.py` | local, zero-drift vs pin | compressed-tensors loader for the vendored PLE shard path | same |
| `srt/models/qwen3_5.py` | pin + re-applied KV-scale patch | fp8_e4m3 KV with scale 1.0: without this, fp8 KV silently corrupts attention (garbled output) | re-apply the patch on every pin bump |
| `srt/speculative/eagle_worker_v2.py` | backport of upstream #32468 (OPEN) | Frees draft embed/lm_head before KV pool sizing → the freed VRAM goes to the KV pool, not to allocator slack; without it FRACTION=0.99 crashes the allocator instead of unlocking 262K context | the enabler for full-context fp8 |
| `srt/layers/quantization/modelopt_quant.py` | FP8_PB_WO routing (fp8_pb_wo.diff) | Route FP8_PB_WO checkpoints (no hf_quant_config.json) through modelopt_mixed; stock build garbles silently (`~~~~`) | required for lovedheart; sanity probe must answer `Paris` |
| `srt/kernels/ops/mamba/mamba_state_scatter_triton.py` | ported 23e51dd | empty-checkpoint guard + step-index bounds in the scatter kernel | not on decode hot path; stability only |
| `srt/mem_cache/mamba_radix_cache.py` | ported 23e51dd | empty mamba checkpoint handling + `_get_children` fallback + track-step bounds | crash-hardening |
| `srt/speculative/spec_utils.py` | ported 23e51dd | draft-spec v2 bounds | prevents OOB draft-spec indexing under MTP |
| `sglang_ssd_stream/config.py` | build RUN patch (layer 5) | coexist: model dirs without `ssd-stream.json` (e.g. Qwen3.8-27B) skip the SSD Stream plugin instead of aborting startup | lets one image serve both families |
| `ssd_stream_patch.py` (this repo) | applied by layer 5 | idempotent config rewriter | re-run after any plugin upgrade |

Upstream patches referenced but NOT in the image: #30092, #32468, #30119 all still OPEN upstream at
the pin date. Do not expect merged fixes; track the PRs before any pin bump.
