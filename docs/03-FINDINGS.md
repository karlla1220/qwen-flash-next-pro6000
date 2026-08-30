# 03 — Findings: performance & VRAM laws on the RTX PRO 6000 (96GB, SM120)

Numbers measured in our runs unless cited.

## KV / context

- **Pool ceiling** = `max_position_embeddings = 262,144` tokens. Raising `--max-total-tokens`
  beyond that is rejected; raising `FRACTION` beyond ~0.99 crashes CUDA (driver, WSL2).
  `FRACTION=0.99` + #32468 unlocks the **full context** at fp8 KV (K size 1.5GB, V 1.5GB).
- **fp8 KV** (`fp8_e4m3`, no scales provided → scaling 1.0): passes easy needle @100K (5/5).
  bf16 KV would cost exactly 2×. The hard profile (dense-similar decoys + multi-hop) is the
  open question — run `scripts/kv_quality_ab.py --profile hard`.
- YaRN factor-2 (524,288 context) is config-only and qualified by an external build on this
  card — an option, not yet tested with MTP here.

## VRAM budget (resident, weights + graph + pools)

| Model | Resident | Notes |
|---|---|---|
| RadixArk 512E, fp8 draft | ~77GB | PLE off-GPU via SSD Stream; 512E NVFP4 ≈ 32-34GB weights |
| lovedheart 448E, fp8 draft | ~14GB less | AIMER pruning removes 128 experts/48 layers; MTP pruned to 448 too |
| draft quant | fp8 saves 3.3GB, **no accept-rate penalty** vs bf16 draft (0.38-0.50 accept both) | MTP accept is workload-dependent, not draft-precision-dependent |

## VRAM water levels and the JIT cliff (read 04 #1 before touching FRACTION

- **Resident** (weights+graphs+pools) is fixed at init; what grows is the caching allocator's
 high-water (freed blocks are cached, never returned to the driver — expandable_segments is
 banned on WSL2) and first-use Trit/FlashInfer **module loads** (`cuModuleLoadData`, per-process,
 one-shot per specialization). Looks like a leak, is actually "max-seen-once + cubin bookkeeping".
- Startup pre-loads only small buckets (default warmup = 8 tokens; CUDA graph = padded small batch;
 flashinfer autotune = GEMM only). Long-context buckets therefore compile mid-request on the
 first run that crosses each length — **fixed by persisting `/root/.triton`** (04 #1, now in all bats)
. Optional full cover: `--warmups` long-context script.
- **HiCache verdict:** does NOT cure these stalls (it offloads KV only, and `MTP + extra_buffer_lazy
 + hicache` hits a known upstream IMA transfer bug). It pays off only when the KV pool can't hold
 N concurrent sessions — deferred until N-way KV overflow is actually needed.

## MTP behavior

- accept rate on this model family: **~87% on code/JSON**, ~40% on chat — matched independently
  by external builds (pennyroyal: 2.58 mean accepted length; lovedheart's 8×code = 100%).
- `QSA MTP index sharing enabled: ... layers [0]` at startup = normal, expected.
- `speculative_draft_kv_cache_dtype=None` → draft KV follows main KV dtype. With fp8 KV it stays fp8.
- accept ≤ ~0.5 ⇒ MTP barely helps: check CUDA graphs first (`cuda graph: True` required).

## Concurrency math

- Concurrency is gated by the **token pool**, not VRAM: N-way ⇒ per-session ≈ `262,144/N`
  tokens resident. fp8/bf16 changes bytes/token, not the token ceiling.
- Mamba slots: MTP needs `num_draft_tokens+1` slots per running request; `MAXREQ×5` is
  enough (auto-sizing will happily eat freed VRAM by over-allocing slots — cap explicitly).
- True N-way long-context requires hicache host tier (host RAM overflow), blocked here by a
  known MTP+extra_buffer_lazy transfer bug (see 04-DEBUG-LOG).

## Backends worth their bytes

- MoE: FlashInfer CUTLASS (`flashinfer_cutlass`) — auto can resolve to TRTLLM, which cannot run
  NVFP4 MoE (runtime `NotImplementedError` at graph capture).
- QSA decode: XQA via the FlashInfer paged wrapper (SM120 passes the `(12,0)` gate; already in pin).
- FlashInfer GDN decode/prefill: gated on `mamba-ssm-dtype bfloat16` and the verify path is
  missing the `gdn_mtp_cache_mode` route in our pin — end-to-end gain ≤ ~2-3%, path broken today.
  (Our Triton GDN decode numbers are already at parity with external FlashInfer runs).
