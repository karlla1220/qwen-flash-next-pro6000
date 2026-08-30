# 04 — Field log: every dead end, the fix, and how to avoid it

Symptoms → root cause → what we did. Newest-ish first. Read this before touching flags.

---

## 1. Long-context prefill tail collapse (the "168K cliff") — TRITON JIT BUCKETS

**Symptom.** Prefill crawls at ~9000 tok/s through ~150K, then collapses to 900–2300 tok/s
over the last ~10K tokens. The first decode reads show 1–6 tok/s. Log shows a burst of
`Triton kernel '<name>' took N s to compile after serving started` +
`device-loaded after serving started (free device mem: 0.00 GiB)` — the names are QSA
graph-layout / `alloc_extend` / track-commit / mamba-scatter kernels.

**Root cause.** Triton compiles on **first use**, specialized per bucket (kernel × arch
× constexpr × divisibility-16/==1 flags of runtime args). Startup only pre-loads SMALL
buckets: the default server warmup is one request of **max_new_tokens = 8** (`http_server.py
_execute_server_warmup`), CUDA-graph capture uses padded small-batch placeholder metadata, and
flashinfer autotune covers **GEMM only** (fused_moe gemm1/gemm2). The long-context branches
only execute when a real request first crosses each length bucket → compile + `cuModuleLoadData`
are paid mid-request.

**Why it repeated on every container restart.** Triton's on-disk cache lives at
`/root/.triton` — ephemeral inside the container, so each fresh process recompiled everything.
The `--warmups` hook (`srt/entrypoints/warmup.py`, csv of registered functions run before
listening) exists upstream precisely for this, but we never configured it.

**Fix (done, in every launcher):** persist the cache —
`-v /mnt/docker-desktop-disk/triton_cache:/root/.triton` (next to the flashinfer jit_cache mount).
Same card / image / Triton version → every later container hits the cache, long-context pays only
`cuModuleLoadData` (~ms/kernel). Cache invalid = rebuild, patch the kernel file, Triton bump,
different arch → one recompile, then it re-persists. Docker Desktop *Reset/clean* wipes it like hf_cache.

**Optional upgrade (backlog, 30 min):** a warmup function that auto-sends a ~180K synthetic
prompt at startup — compiles all buckets inside engine init (startup +30–60 s, no mid-request
compile ever, kills the `device-loaded` OOM window). Needed only if new code paths appear after a
patch or cache loss.

**Residual.** After JIT coverage, a *sustained* decode slowdown at long context with a
big bf16 pool is NOT this bug (see #10).

---

## 2. Triton module-load can OOM at pool-full — the `triton_load_watch` tripwire

`SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY=1` raises on any post-ready load (for CI recipes that
assert full warmup coverage). Our `FRACTION=0.99` leaves ~0 headroom: a first-use load during
serving can legitimately OOM. Upstream's own guidance: "Pre-load it during engine init".
The persistent cache (#1) makes the load cheap; the warmup script makes it early.

---

## 3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — DO NOT USE (WSL2)

**Symptom.** Server dies at CUDA init / first graph capture (hard crash, exit 137).
**Root cause.** WSL2 + NVIDIA driver incompat with expandable_segments allocator.
**Action.** Removed from every launcher; never reintroduce. Any `PYTOR_*` env left in bats is
documentation-only and must not include expandable_segments.

---

## 4. SSD Stream plugin needs `--security-opt seccomp=unconfined`

io_uring (`io_uring_setup`/`io__register`) is blocked by the default seccomp profile → the plugin
fails to register the 47.68 GiB O_DIRECT table. All launchers carry the flag; container needs
`--ipc=host` too.

---

## 5. Patch baseline is `dist-packages`, NOT `/sgl-workspace`

The base image ships a git checkout of main at `/sgl-workspace/sglang` (UNUSED by python).
Patching against that tree silently produces a different tree (missing NvFp4EmbeddingMethod,
main-tree SM120 GEMV, etc.) that imports fine but behaves wrong. All `COPY` layers take their
source from `python3 -c "import sglang; ..."` — dist-packages.

---

## 6. MoE runner `auto` resolves to TRTLLM → `NotImplementedError` on NVFP4 MoE

Fires during CUDA-graph capture, looks like a CUDA error. Pin explicit
`--moe-runner-backend flashinfer_cutlass` (and `--speculative-moe-runner-backend` likewise).

---

## 7. FP8_PB_WO checkpoints garble silently in stock builds

Stock `modelopt_quant` has no `FP8_PB_WO` route; the lovedheart repo ships no
`hf_quant_config.json` so stock detection fails → garbled `~~~~` output. Fixed by the
`modelopt_quant.py.in` routing patch + explicit `--quantization modelopt_mixed` +
`--hf-cofig`/no-op sanity probe must return `Paris`. (Patch: `scripts/lovedheart/fp8_pb_wo.patch`).

---

## 8. lovedheart flattening: MTP in root index + bf16 PLE shards

The repo puts MTP weights in the root safetensors and stores the PLE table as 128 BF16 shards.
**Fix:** `mtp_synth.py` builds `mtp/` (vendor mtp config, `num_experts 512→448`, index maps
`mtp.*` → the flattened root files); `ple_probe.py` proved the 128 shards are a byte-perfect
re-shard of the vendor fp8 bin (sha256 `b070f9…` — 47.68 GiB) → symlink + `ssd-stream.json`, zero copy.
`deploy.sh` does index surgery (drop `ple/ple-bf16-*` entries, `weight_scale` → local scale file).

---

## 9. Port hygiene — never kill the serving backend

Port 18081 is served by a different container (`sglang-qwen` / ninfer) that may be serving the
coding session itself. Flash-Next got its own container names (`sglang-loved`, port 18082; the
`MODEL=loved` dual-profile flag). Stop/`docker rm` of other services is a hard never — check
`docker ps` before any port change.

---

## 10. Sustained long-decode slowdown on a big bf16 pool (5–10 tok/s at 187K — A/B OPEN)

**Symptom.** 262K→524K bf16 pool profile (MAXREQ=2): decode pins at 5–10 tok/s at 187K and does
not recover; fast profile (fp8 KV, 262K pool) holds 100–172 tok/s at the same context.
**What it isn't.** Not JIT (#1), not `23e51dd` stability port (index clamps only, O(1), not
allocation-impacting), not Triton parity gap.
**Top suspects.** (a) PLE SSD-stream gather blocking on the critical path via
`ticket.wait_for_launch()` (CPU-side sync in `_consume_prefetched_embeddings`) once the
47 GB table's working set escapes the WSL page cache — staging slots are 2×16 MB and
`capacity=104857 rows`; at 524K pool the row working set blows past it, so misses hit the
Docker Desktop virtual disk. (b) QSA compressed addressing at 8192 pages (524K/64) vs
4096 (262K) interacting with the indexer budget.
**Planned probe (needs GPU idle): same prompt, same container, one var at a time:
187K prompt → fp8 262K pool vs bf16 524K pool; then PLE-offload off vs on; then
`staging_per_slot` ×2.** Don't chase it otherwise.

---

## 10a. HiCache on this machine

Not the fix for #1/#10 (offloads KV only; doesn't touch Triton JIT or PLE gathers). And the
L2 host tier is broken with our stack: `MTP + extra_buffer_lazy + hicache` hits a
known upstream transfer bug (Mamba track transfer IMA), so the whole hicache route is
DEFERRED — revisit only for N-way concurrency with host-overflow KV.

## 10b. FlashInfer GDN backend (linear-attn decode) — broken path on our pin

Requires `mamba_ssm_dtype float32→bfloat16` (unapproved), `gdn_mtp_cache_mode=none`
(kills our MTP cache mode) + RecoverSSM, and the *verify* path in our pin lacks the
`gdn_mtp_cache_mode` route (AttributeError at startup). End-to-end gain ≤ ~3% (GDN is 9% of
decode bandwidth; Triton decode already at parity with external FlashInfer-GDN runs).
**Action: skipped, revisit on a sglang bump.**

## 10c. CUDA graph is mandatory for accept rate

accept < 0.5 → first check `cuda graph: True` in the log (graphs on, bs=[1,2]). Also
`QSA MTP index sharing enabled… layers [0]` at startup is normal, not a warning.

## 10d. Pinned commit: 3df8e1e7

The only tree where Qwen4-Exp + SSD Stream + NEXTN all work; DFlash2 backport copied from
main 41c018a9. The stability port 23e51dd (empty mamba checkpoint, track step bounds,
`_get_children` fallback) applied on top. #32468 (embed/lm_head freed before pool sizing)
backported into `eagle_worker_v2.py`.
