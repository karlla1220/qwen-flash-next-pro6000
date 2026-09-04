# 09 — RAM-offloaded PLE profile (primitive-ai/Qwen3.8-Flash-Next-NVFP4, WSL2 Ubuntu, no SSD Stream)

A second profile alongside the SSD-Stream one documented in `01`-`08`: same model
architecture (qwen4_exp / Qwen3.8-Flash-Next), different checkpoint, and the PLE
n-gram table lives in **host pinned RAM** instead of being streamed from NVMe.
Native `docker` in WSL2 Ubuntu, not Docker Desktop.

## Why no custom image is needed here

`--ple-offload-embedding` is not part of the `sglang_ssd_stream` plugin. It is a
native sglang flag (`sglang/srt/server_args.py`, implemented by
`Qwen4ExpPinnedHostEmbedding` in `sglang/srt/models/qwen4_exp.py`), on by default for
BF16 Qwen4-Exp on CUDA. The plugin's only job is to intercept that already-pinned
tensor and back it with NVMe io_uring reads *when the checkpoint ships an
`ssd-stream.json`*. Don't install the plugin and don't ship that manifest, and the
native path is exactly host-RAM pinning — nothing to build.

Checked directly against `lmsysorg/sglang:qwen38flashnext` (pulled 2026-09-04,
commit `593134d17a6eb0d0fc5f71a970cd2e9dc8e26e8b`, 2026-09-03):
- `sglang_ssd_stream` plugin: **absent**.
- `qwen4_exp.py` / `Qwen4ExpPinnedHostEmbedding`: present natively.
- FP8_PB_WO dispatch fix (the one `docker-target/patches/modelopt_quant.py.in`
  backports for lovedheart-style mixed precision): **already present upstream**.
- The `eagle_worker_v2.py` #32468 backport (moves `init_token_map`/`init_lm_head`
  out of `alloc_memory_pool()` so the draft's own embed/head copies are freed
  *before* KV-pool sizing profiles free memory) — **still not merged upstream**
  as of this commit. Effect if skipped: the KV budget is sized a bit more
  conservatively (measured later at ~1.3 GB on this profile — see the third
  overlay section below), not a crash. Initially skipped; ported as a third
  overlay on 2026-09-05 once the exact cost was measured (staged, not yet built).

So this profile runs the stock image unmodified. The other 7 local patches
(`auto_round.py`, `gptq.py`, `compressed_tensors.py`, `qwen3_5.py`,
`mamba_state_scatter_triton.py.in`, `mamba_radix_cache.py.in`, `spec_utils.py.in`)
were not re-checked line-by-line against this newer commit — don't copy them onto
this image blindly, the surrounding code has moved on (e.g. `eagle_worker_v2.py`
gained an `unwrap_lora_layer` call and switched `server_args.enable_dp_attention` to
`get_parallel().enable_dp_attention` that our old patch file would silently
regress). If something breaks, diagnose from the actual error first.

## QSA + fp8 KV cache: one backport needed (`docker-target/Dockerfile.qsa-fp8-fix`)

Unlike the items above, `--kv-cache-dtype fp8_e4m3` does need a patch on this image.
Reproduced live (2026-09-04): once a request hits cached prefix under fp8 KV — the
2nd+ chunk of chunked prefill, or any multi-turn/radix-cache-hit request — it
crashes with a Triton `CompilationError: ... Unsupported rhs dtype fp8e4nv` inside
the QSA sparse-attention kernel. Root cause, confirmed by reading
`qwen_sparse_attn_backend.py`'s `forward_extend()`: a fresh request with no cached
prefix passes freshly-computed bf16 k/v straight to `sparse_gqa_fwd_interface_triton`
(fine), but a request with a cached prefix instead gathers K/V *from the fp8_e4m3 KV
pool* via `index_select` and hands the fp8 tensors to
`sparse_gqa_fwd_interface_triton_ck` with no cast — `tl.dot(bf16 q, fp8 k)` doesn't
compile. Short single-chunk prompts never hit this branch, which is why the earlier
"Paris"/haiku accept tests (§ below) missed it.

Fix: `sgl-project/sglang@d6ff2d881e78d0746e0393e9860ce3de5d84de8b` — two-line change,
casts the gathered fp8 rows to `q.dtype` right before that call
(`torch.cat(k_parts).to(q.dtype)`, same for `v_parts`). Lossless: fp8 KV here has
implicit scale 1.0 (same fact behind the "no scaling factors provided" boot warning),
so widening back to bf16 is a pure dtype cast, not a real dequant. The kernel itself
is untouched. This commit is not reachable from `593134d17` (diverged history, not
part of any open PR) and not superseded by anything larger:
- The companion commit next to it, `fa862ff2` (route QSA sparse *decode* through
  trtllm-gen on SM120), is **not needed** — this image already has the equivalent
  fix under a different helper name (`is_sm100_supported() or is_sm120()` in
  `_resolve_trtllm_sparse_decode`).
- PR #36787 (draft, stacked on #36497) is a much larger SM120 *decode*-path rewrite
  (direct-paged Triton sparse decode, MQA scoring, MoE kernels, PLE NUMA pinning) —
  none of its 13 commits touch `forward_extend()`'s prefix-hit branch; unrelated to
  this bug.
- PR #37798 (part of a from-scratch 24GB-card series) solves a different, harder
  problem — new int8/int4/tiered KV quantization pool classes so a 24GB card can run
  this model at all — genuinely overkill for just unblocking fp8_e4m3 on a 96GB card.

Applied as `docker-target/Dockerfile.qsa-fp8-fix`: `FROM lmsysorg/sglang:qwen38flashnext`
+ one file overlay (`docker-target/patches-official/qwen_sparse_attn_backend.py`,
the stock file with the two `.to(q.dtype)` casts added) + a syntax/import check.
Built and tagged `sglang-flash-ram:qsa-fp8-fix`; `config/run_primitive_ram.sh`
defaults `IMAGE` to it and re-enables `--kv-cache-dtype fp8_e4m3` by default. Not yet
re-run end-to-end with a multi-chunk/multi-turn request against the patched image —
the fix was taken from the upstream author's own commit message/reasoning rather
than re-verified here, per instruction.

## PLE embedding dtype: config.json says fp8, checkpoint ships bf16

`primitive-ai/Qwen3.8-Flash-Next-NVFP4`'s `config.json` sets
`text_config.ple_embedding_dtype = "float8_e4m3fn"`, but the shipped PLE shards
(`ple-bf16-*.safetensors`) are plain BF16 with no scale tensor. In
`qwen4_exp.py`, the embedding's storage dtype is chosen as:
```python
torch.float8_e4m3fn if (quant_config.get_name() == "fp8") or config.ple_embedding_dtype == "float8_e4m3fn" else torch.bfloat16
```
Our quant is `modelopt_fp4`, not `"fp8"`, so the *only* reason this table gets
downcast is that stray config field — and the downcast is uncalibrated
(`weight_scale` is hardcoded to `torch.ones(1)`, no real scale). The boot log
says as much: `PLE checkpoint shards are torch.bfloat16 but the embedding
storage is fp8 ... downcasting is lossy`.

This is very likely a leftover/copy-paste field, not this checkpoint's intent:
the model card explicitly describes this checkpoint as the BF16-table option
and contrasts it against separately-listed "FP8-table checkpoints (official
FP8, RadixArk, Inferact)" — "the n-gram table lives in host RAM (~100 GB)"
matches serving it as BF16, not the ~48 GiB an fp8 downcast would give. Silently
quantizing an uncalibrated table is exactly the class of problem that started
this whole line of investigation (lovedheart's ngram-quantization language
quality complaint) — worth fixing, not shrugging off.

Fix, without touching the ~186GB download: `--json-model-override-args
'{"text_config": {"ple_embedding_dtype": null}}'` — verified directly (loaded
`get_config()` with and without the override, CPU-only, no GPU needed) that this
flips `ple_embedding_dtype` from `"float8_e4m3fn"` to `None`, which restores the
BF16 branch. `config/run_primitive_ram.sh` applies this by default
(`PLE_DTYPE_OVERRIDE=1`); set `PLE_DTYPE_OVERRIDE=0` to keep the fp8 downcast
(~48 GiB instead of ~95 GiB resident) if the accuracy loss turns out to be
acceptable for your use — both fit comfortably in a 163GiB WSL2 RAM budget, so
there's no capacity reason to default to fp8 here. Not yet A/B tested for output
quality difference; the fix here is "match the checkpoint author's own stated
design and avoid an unnecessary uncalibrated quantization," not a measured
quality delta.

## PLE offload + BF16 table: transient GPU OOM during hybrid-cache sizing

Turning on the BF16-table fix above (`PLE_DTYPE_OVERRIDE=1`) together with
`--ple-offload-embedding` used to OOM at boot with `RuntimeError: Not enough
GPU memory for hybrid (mamba/linear-attention) state cache` on this box (512E
model already resident at ~93.6GiB of a 97.9GiB card). Root cause, confirmed
by reading `qwen4_exp.py`/`unquant.py`/`model_loader/loader.py` live inside
the running container: `Qwen4ExpNGramEmbedding.__init__`'s
`VocabParallelEmbedding(...)` construction bottoms out in
`UnquantizedEmbeddingMethod.create_weights` (`layers/quantization/unquant.py`),
which calls `torch.empty(...)` with no `device=` kwarg. Model construction
happens inside `DefaultModelLoader.initialize_model_for_startup`'s
`with target_device:` (`target_device = torch.device("cuda")`), so this
tensor lands on GPU by ambient default-device context — the FULL PLE table,
once per `ple_layer_ids` entry, each with its own independent table
(`Qwen4ExpLayerExtensionMixin._init_qwen4_exp_layer_extensions`), sized
~48GiB (fp8) or ~95GiB (bf16) depending on `ple_embedding_dtype`. It's freed
a few lines later once `Qwen4ExpPinnedHostEmbedding.__init__` copies it to
pinned host RAM and `del embedding.weight`s the GPU original — but the
transient spike alone, plus PyTorch's CUDA caching allocator not reliably
returning freed blocks to the driver, is enough to starve the mamba-cache
sizing that runs right after model construction on a near-full card.
Confirmed empirically on this box, identical flags apart from
`PLE_DTYPE_OVERRIDE`: fp8 (~48GiB spike) boots fine at `FRACTION=0.96`; bf16
(~95GiB spike) pushes VRAM to ~99% and OOMs with the exact error above.

Fix: in `Qwen4ExpNGramEmbedding.__init__`, construct the
`VocabParallelEmbedding` under `with torch.device("cpu"):` whenever
`config.ple_offload_embedding` is set. There's no reason for a table whose
final home is pinned host RAM (`Qwen4ExpPinnedHostEmbedding`) to ever touch
the GPU in the first place; `Qwen4ExpPinnedHostEmbedding.__init__`'s own CPU
pinned-tensor allocation already explicitly passes `device="cpu"`, so it's
unaffected — it now does a CPU→pinned-CPU copy instead of GPU→pinned-CPU.
Not an upstream commit backport (no matching fix found upstream as of
`593134d17`); uses an idiom already established elsewhere in this sglang
codebase for defeating an ambient CUDA default-device context
(`utils/common.py`'s `device_context()`; `models/transformers.py:257-259`;
`lora/backend/chunked_backend.py:485,535`).

Applied as a second overlay in the SAME `docker-target/Dockerfile.qsa-fp8-fix`
(one more `COPY` of `docker-target/patches-official/qwen4_exp.py`, the stock
file extracted from the running container with the `torch.device("cpu")`
wrap added) — kept in the same image/tag as the QSA fp8-KV fix rather than a
separate Dockerfile, since both are one-file overlays on the same base image
for the same profile and there's no scenario here needing one without the
other. Rebuilt and re-tagged `sglang-flash-ram:qsa-fp8-fix` (same tag, new
layer).

## Third overlay: #32468 re-port (`eagle_worker_v2.py`) — staged, NOT yet built

Status as of 2026-09-05: **the overlay file and Dockerfile block exist in the
repo but the image has NOT been rebuilt and the running container is NOT
restarted.** All numbers below are measurements of the *unpatched* boot plus
arithmetic; the patched boot is unverified until someone builds and re-runs.

Re-checked against the actual PR (2026-09-05): [sgl-project/sglang#32468](https://github.com/sgl-project/sglang/pull/32468)
is still **OPEN**, 1 commit (`a15c74d`), +13/−2, 1 file, unreviewed (author
can't get the `run-ci` label so the gate skips real tests). The PR diff is
exactly the 3-line change we ported: `init_token_map()` + `init_lm_head()` +
`torch.cuda.empty_cache()` move from the end of `alloc_memory_pool()` to the
end of `EagleDraftWorker.__init__` (our staged file = PR semantics, ported
onto this image's stock file — the old `patches/eagle_worker_v2.py` tree would
have regressed 5 drifted stock hunks, ~170 diff lines: `get_spec()` /
`get_parallel()` refactor, `unwrap_lora_layer`, `draft_pp_context`).

What it costs in VRAM on THIS profile, measured from the running container's
boot log rather than estimated: draft weight load used 3.86 GB against a
quantized MTP checkpoint of 2.61 GB on disk (31 tensors, 5.21 GB BF16 / 2.61
GB in-file fp8) → the draft's shareable duplicates are resident at **~1.3 GB**
at profile time (vocab 248,320 x hidden 2,560 = 1.27 GB/tensor BF16; the
upstream PR's "2.54 GB each" math is hidden=5120 — the SSD-stream profile's
model, not this one). Expected effect: KV `#tokens` ~520,960 → ~630K
(+~1.3 GB / 11.4 B-per-token fp8 KV), or the same KV at lower FRACTION.
Expected NOT to happen: anything dramatic — this profile already boots
CTX=262144 SPEC=on at FRACTION=0.96 unpatched with 4.59 GB spare, so #32468
here is headroom, not an enabler (it IS an enabler on the SSD-stream image —
that's the row in `06-PATCHES.md`, different card math). PR author reports
greedy output byte-identical before/after (pure order-of-assignment move);
one behavior delta: with `--speculative-use-rejection-sampling` the
draft/target vocab-mismatch `ValueError` now fires at worker construction
instead of pool alloc (unused by this profile).

Dockerfile gate for the third overlay is stronger than syntax+import: the
build asserts `init_lm_head()` present in `EagleDraftWorker.__init__` and
absent from `alloc_memory_pool()`, so a future pin bump that silently ships
the PR upstream (or drops our overlay) fails the build loudly instead of
quietly. When that happens, delete this overlay — do not "fix" the assert.
Also on any pin bump: re-check #32468's merge status first (see 08); if
merged, the assert firing is expected, not a bug.

## Model layout: primitive-ai/Qwen3.8-Flash-Next-NVFP4

- Full 512-expert checkpoint (not pruned like lovedheart), NVFP4 experts + BF16
  tail, `hf_quant_config.json` present (quant auto-detects, unlike lovedheart).
- PLE table: `ple-bf16-00..42.safetensors`, 43 shards, plain BF16, ~95 GiB on disk,
  no scale tensor shipped alongside it. See "PLE embedding dtype" below — served
  as BF16 (~95 GiB resident) by default in this repo's launcher, not the fp8 the
  checkpoint's own `config.json` asks for.
- No dedicated `mtp/` directory — like lovedheart, the 31 `mtp.*` tensors are
  flattened into the root `model-bf16-000{10,11,12}.safetensors` shards. Unlike
  lovedheart this checkpoint is unpruned (`num_experts` still 512), so
  `scripts/primitive/mtp_synth.py` needs no 512→448 remap: it copies the root
  `config.json` verbatim and symlinks the shards containing the mtp.* keys.
- The vLLM-oriented sections of the model card (`VLLM_PLE_CPU_OFFLOAD`,
  `worker_image_disk.py`, `connector_mrv2.py`) are for `vllm/vllm-openai`, not
  sglang — ignored here.

## Reproduce

```bash
# 1. Download (~186GB; a plain local dir, not the HF blob cache, so it can be
#    bind-mounted straight into the container):
hf download primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4 \
  --exclude "worker_image_disk.py" --exclude "connector_mrv2.py" --exclude "assets/*"

# 2. Synthesize mtp/ (needed for --speculative-algorithm NEXTN):
python3 scripts/primitive/mtp_synth.py ~/models/Qwen3.8-Flash-Next-NVFP4

# 3. Build the one-file QSA fp8 KV fix on top of the official image (see below):
cd docker-target && docker build -t sglang-flash-ram:qsa-fp8-fix -f Dockerfile.qsa-fp8-fix . && cd ..

# 4. Serve (IMAGE defaults to sglang-flash-ram:qsa-fp8-fix):
PORT=8000 bash config/run_primitive_ram.sh
```

## Validated (2026-09-04, RTX PRO 6000 Blackwell Max-Q 96GB, WSL2 Ubuntu)

- `CTX=32768 MAXREQ=1 FRACTION=0.90 SPEC=off`: boots, weight load 269s, GPU mem
  ~85.7GiB used / 97.9GiB total, host RAM +~75GiB while resident. Accept test
  (`"What is the capital of France? Answer in one word."`) → `Paris`, reasoning
  intact, no garbling.
- Same, `SPEC=on DRAFT8=off`: **OOMs** — `Not enough GPU memory for hybrid
  (mamba/linear-attention) state cache`. Expected: an unquantized bf16 draft
  plus speculative buffers don't fit at `FRACTION=0.90`.
- `SPEC=on DRAFT8=on FRACTION=0.94`: boots (weight load ~330s incl. draft),
  GPU mem 88.9GiB/97.9GiB. `speculative_algorithm=EAGLE` (NEXTN), draft path
  `/model/mtp` (our synthesized dir) loads without error. Haiku-generation test:
  coherent output, `accept len: 2.62, accept rate: 0.54, cuda graph: True` — the
  synthesized draft model is producing tokens the target actually accepts, not
  just noise.
- Full `CTX=262144` not yet pushed to the limit in this pass — start from
  `FRACTION=0.94 DRAFT8=on` and back off `--max-running-requests`/context if the
  mamba-cache-sizing error above reappears.
- Post-fix (`qwen4_exp.py` PLE-offload GPU-OOM fix, rebuilt into the same
  `sglang-flash-ram:qsa-fp8-fix` tag): `CTX=262144 SPEC=on DRAFT8=on
  PLE_DTYPE_OVERRIDE=1 FRACTION=0.94` — the exact config that previously OOM'd
  with `Not enough GPU memory for hybrid (mamba/linear-attention) state cache`
  (BF16 table's transient GPU spike) — now boots cleanly at full context.
  `Load weight end`: `avail mem=12.56 GB, mem usage=81.05 GB` (vs. the
  pre-fix run's `avail mem=0.00 GB, mem usage=93.60 GB` for the *fp8* table at
  the same fraction — the fix leaves substantially more headroom, not less,
  since the table no longer touches the GPU at all). Steady-state after boot:
  `nvidia-smi` ~92GiB/97.9GiB VRAM (KV/mamba cache + cuda graphs at full
  262144 context, not a PLE spike), scheduler process RSS ~137.9GiB host RAM
  (the ~95GiB BF16 table now resident there instead of transiting the GPU).
  Accept test → `Paris`, reasoning intact.
