# 05 — Data summary: everything measured on this box

RTX PRO 6000 Blackwell 96GB (SM120) · SGLang `3df8e1e7` · flashinfer 0.6.17 · CUDA 13 stack
· Qwen3.8-Flash-Next family (hybrid GDN+QSA, PLE, MTP/NEXTN)

## 1. Throughout (single stream, fp8 KV unless noted)

| Workload | Value | Setup | Source |
|---|---|---|---|
| Decode @ ctx 0–128K, bs1 | **110–122 tok/s** | vendor SSD-Stream, 30 s cells, ctx 0/16K/32K/64K/128K | `llm-inference-bench/benchmark_results_mtp_draft.json` (30 s sustained, accept ON) |
| Decode @ 137K, bs1 | **142–172 tok/s** | fp8 KV 262K pool, cuda graph True | live server log |
| Decode @ 262K, bs1 | **121–152 tok/s** | fp8 KV, vendor model | live server log (17:30 instance) |
| Prefill cold 8192 | **9,198 tok/s** (0.89 s) | integrated scout | bench prefill table |
| Prefill cold 64,120 | **10,632 tok/s** (6.0 s) | bench `65536` cell | bench prefill table |
| Prefill steady-state | ~9,000 tok/s up to ~150K, then cliff (see 04 #1) | long prompt | live logs |
| External same-card ref (jpezzulli): 1× decode 171 · 4× decode 427 · 64K prefill 10,103 · MTP accept 2.58/52.7% | — | — | `probe/pezz_readme.md` |

## 2. Quality / accept

| Item | Value | Notes |
|---|---|---|
| fp8 KV needle 100K (easy, 5 probes) | **5/5**, seed 20260830, temp 0 | `kv_fp8.json` — 12–16 s each (115–190K prompt) |
| bf16 KV vs fp8 KV accept | **equal** (0.38–0.50 band, workload-dependent) — fp8 saves 3.3 GB, keep fp8 | A/B, same prompts |
| MTP accept: code/JSON | ~**87%** | matches external 2.58 mean accepted len / 52.7% |
| MTP accept: chat | ~**40%** | accept ≤0.5 ⇒ check `cuda graph: True` first |
| fp8 draft vs bf16 draft | **no accept penalty** for fp8 (–3.3 GB) | fp8 is the default |

## 3. VRAM ledger (resident after init, vendor 512E unless noted)

| Bucket | Size | Notes |
|---|---|---|
| Model weights (NVFP4 + scales) | ~32–34 GB | 512E; lovedheart 448E ≈ **14 GB less** |
| Draft MTP weights fp8 | ~3 GB (fp8) / 6 GB (bf16) | fp8 chosen |
| KV pool fp8 | 262,144 tokens × 12,288 B = **1.55 GB** | `fp8_e4m3`, no scales |
| KV pool bf16 comparison | × 24,576 B/token | 524K → 2.17 GB |
| Mamba states | ~2–3 GB (10 slots × 36 GDN layers) | `max_mamba_cache_size` auto = 5×req |
| CUDA graphs + activations | ~7–9 GB | graphs bs=[1,2] |
| PLE n-gram table | **0 GB on GPU** (47.68 GiB NVMe, io_uring O_DIRECT) | `--ple-offline-embedding` |
| Resident total | **~77 GB** / 96 GB, headroom for Triton loads | FRACTION=0.99 |

## 4. KV math (why every GB goes to context, not to KV)

Full-attn layers **12**, kv_heads 2, head_dim 256 → bytes/token =
12 × 2(K+V) × 256 × dtype: **12,288 B fp8 / 24,576 B bf16**. GDN state (36 layers, 48 heads,
128 dk/dv, float32) is context-independent: ~10 MB/slot — the reason fp8 KV is the cheapest
context lever and bf16 KV at 524K costs 4× fp8 at 262K.

## 5. Concurrency

- Pool ceiling = 262,144 tokens; N-way long-context ⇒ ~`262,144/N` resident per session (fp8/bf16
change bytes/token, not the token ceiling).
- MTP slots per running request = `num_draft_tokens+1` (auto = 5×req; cap explicitly —
auto-sizing happily over-allocs with free VRAM).
- 4× concurrent needs a KV-tier overflow (hicache host tier) — blocked today by the MTP+extra_buffer_lazy transfer bug (04 #10a).

## 6. Environment facts worth re-verifying after a rebuild

| Item | Value |
|---|---|
| Pinned sglang commit | `3df8e1e7dbc5807696622afe2929b6c33c185ca3` |
| flashinfer | `0.6.17` + `flashinfer-cubin 0.6.17` + `flashinfer-jit-cache 0.6.17+cu130` |
| SSD Stream wheel | v0.1.0 (`sglang.srt.plugins` entry point) |
| Ports | vendor 18081, loved 18082, container `sglang-flash-27b` / `sglang-loved` |
| Triton cache | `/mnt/docker-desktop-disk/triton_cache` (mounted to `/root/.triton`) |
| flashinfer cache | `/mnt/docker-desktop-disk/jit_cache` (`/root/.cache/flashinfer`) |
| Vendor snapshot | `garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream` |
| lovedheart snapshot | `lovedheart/Qwen3-8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000` |
| PLE bin sha256 | `b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3` (47.68 GiB, 128 shards = byte-perfect re-shard) |
| Official sampling | temp 1.0, top_k 20, top_p 0.95 |
| MUST NOT | `expandable_segments` (WSL2 crash) · `moe-runner auto` (TRTLLM NVFP4 crash) · kill the serving backend container |

## 7. Open experiments (this session's backlog, ~1 GPU-sit each)

1. **KV hard profile** (`scripts/kv_quality_ab.py --profile hard`) — dense-similar decoys + multi-hop at 131K.
2. **PLE-vs-disk A/B** for the 5–10 tok/s sustained decode on the 524K bf16 profile (04 #10).
3. **YaRN factor-2** (524,288 ctx, config-only) with MTP.
4. **Concurrent probe MAXREQ=2/4** — retraction behavior, accept under load.
5. Optional `--warmups` long-context script (04 #1).
