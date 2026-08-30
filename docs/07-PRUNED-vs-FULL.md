# 07 — Pruned (448E) vs unpruned (512E): differences and the trade-off

Both serve from the same image/profile; switch = `MODEL=loved|vendor` in the launcher.
"pruned" = lovedheart AIMER pruned checkpoint; "unpruned" = vendor RadixArk NVFP4 (512 routed experts).

## What is actually identical

| Item | Status |
|---|---|
| Architecture | 48 MoE layers, hidden 2560, `num_experts_per_tok=10`, GDN/QSA/PLE unchanged, 4.07B/layer resident |
| Quantization | NVFP4 W4A4 routed experts only; attention/shared-experts/MTP/LM-head stay BF16 |
| PLE n-gram table | Byte-ident (sha256 `b070f9…`), 47.68 GiB, streamed from NVMe, 0 VRAM in both cases |
| KV / MTP / Mamba configs | Same flags, same 262,144 native context (fp8 KV), same NEXTN params |

So the difference is exactly: **448 vs 512 routed experts per layer (64 removed/layer**, K=64), no
calibration data — expert-importance score `mean|W| / RMS(W)` per layer, drop the lowest 64, reindex
contiguously, router rows stay aligned; MTP pruned to 448 as well.

## Measured on this box

| Metric | 512E vendor | 448E lovedheart | delta |
|---|---|---|---|
| Weight resident | ~32–34 GB | ~14 GB less | frees 14 GB VRAM |
| KV pool @ 262,144 | same 1.55 GB fp8 | same (pool unchanged) | — |
| Resident total | ~77 GB | ~47–58 GB | 2 profiles both fit the card |
| Headroom for KV/other | 15 GB | 15 GB + 14 GB | 4× KV pool budget or 4–8 way concurrency at 262K/req |
| Launch probe (`Paris`) | pass | pass (needs `--quantization modelopt_mixed` + `flashinfer_cutlass`) | FP8_PB_WO routing patch required on our pin |
| Decode (fp8 KV, bs1, MTP on) | 110–222 tok/s (0–262K) | same family numbers | accept rate is workload-bound, not draft-precision-bound |

## Published quality (their evals; protocol + attribution caveats apply)

| Eval | Protocol | BF16 reference¹ | 448E NVFP4 checkpoint |
|---|---|---|---|
| GSM8K | full 1319, t0.6/top-p 0.95/max 8192, 1-shot | 97.12–97.50 (3 runs) | **97.27** (1283/1319, stop 98.86, err 0) |
| AIME26 | 30×8, t1.0, max 130k, thinking on | 100% (240/240) | **98.75 pass@1**, majority@8 100%, pass@8 100% (stop 99.16, trunc 0.83, no_answer 0.42) |

¹ Revision deltas not published — the two revisions differ in the PLE tables (FP8 vs BF16), so
treat as indicative; **there is no published GSM8K/AIME run of the unpruned 512E NVFP4 checkpoint**,
so the only apples-vs-apples statement is: pruned-NVFP4 lands **in-band** of the BF16 reference on both evals.
Qual note: "single-turn accuracy preserved in-band; long agentic generations tend to run longer than BF16."
Our own needle test (fp8 KV, 100K, easy profile) passes 5/5 — the same result for either model,
which is expected because the PLE tables are byte-ident.

## The trade, stated plainly

**Buy 448E when you need the 14 GB.** That VRAM is what everything else here is starved of:

| Want | 512E (262K ctx, fp8 KV) | 448E |
|---|---|---|
| 1× @ 262K | ✅ as-is | ✅ + spare |
| 4× @ 262K (fp8 KV = 162K/session) | needs KV overflow tier (hicache, blocked) | the freed 14 GB buys the 4-way KV pool directly — no tier needed (MAXREQ=4, MAMBA×4) |
| 1024K context (YaRN×2) | fp8 KV ≈ 3 GB — tight at 0.99 fraction | comfortably inside 14 GB slack |
| bf16 KV 524K (pool 2.17 GB) | competes with graph/activation slack | headroom-safe |
| "maximum quality" | unproven advantage on this eval table; in-band only | same in-band, plus the caveat that pruned experts were scored, not retrained |

## Decision rule

- **Default: vendor 512E** — the stock profile, one less variable in its provenance, and published accept rates are from 512E.
- **Switch to 448E** when: you need N-way concurrency or longer context and the KV pool is the binding constraint; or you want bf16 KV without the pool-size cliff; or you need a clean control to prove a slowdown is model-family-specific (the two serve as controls for each other).
- **Not a free win:** pruning is fixed and task-agnostic (no calibration data) — if your workload leans on the pruned experts, expect drift. We have no eval of that, only the in-band GSM8K/AIME and the needle 5/5 at 100K; run `kv_quality_ab.py` (hard profile) on whichever checkpoint your users live in.

## Upstream note worth carrying

The pruned repo's launcher exports `SGLANG_QSA_USE_FP8_INDEXER=1` and claims it keeps
512K-context prefill fast on their newer build. Our pin does not set that env in either bat;
carry it on the backlog for a pin bump (05 §7).
