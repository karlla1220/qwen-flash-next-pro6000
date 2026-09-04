# Qwen3.8-Flash-Next on a single RTX PRO 6000 96GB — Playbook

Everything learned running **Qwen3.8-Flash-Next** (hybrid GDN + QSA + PLE + MTP) on one
RTX PRO 6000 Blackwell 96GB (SM120) with **SGLang + SSD Stream PLE**, distilled into a
reproducible kit: Docker image build, launch profiles, quality probes, and a field log
of every dead end we hit.

## What works today

| Result | Value | How |
|---|---|---|
| Context | **262,144 tokens (native max)** at fp8 KV | `--ple-offload-embedding` + SSD Stream PLE + #32468 backport, `FRACTION=0.99` |
| Throughout | 60-200 tok/s single stream, `cuda graph: True`, accept 0.3-0.9 | fp8 draft (`--speculative-draft-model-quantization fp8`, saves 3.3GB, no accept penalty) |
| Models served | vendor RadixArk NVFP4 (512E) and lovedheart AIMER-pruned 448E (saves 14GB VRAM) | same image, one switch in the launcher |
| VRAM | ~77GB resident (512E, PLE streamed to 0 VRAM), 15GB headroom for KV | SSD Stream plugin, io_uring O_DIRECT reads |

## Repo map

```
docker-target/            build of image sglang-flash-27b:latest (see docs/01-BUILD.md)
  Dockerfile                layered recipe: pinned sglang 3df8e1e7 + flashinfer 0.6.17 stack + plugins + patches
  patches/*.py[.in]     source patches applied at build (FP8_PB_WO routing, MTP #32468, mamba radix fixes)
  Dockerfile.qsa-fp8-fix    one-file overlay on lmsysorg/sglang:qwen38flashnext: backports the QSA
                            fp8-KV-gather dtype fix + fixes a transient GPU OOM in PLE-offload's
                            BF16-table construction (see docs/09-PRIMITIVE-RAM-OFFLOAD.md)
  patches-official/qwen_sparse_attn_backend.py, qwen4_exp.py, eagle_worker_v2.py   the patched files
                            that overlay copies in (eagle = re-port of #32468, staged 2026-09-05,
                            image not yet rebuilt with it)
config/
  run_sglang_flash_next.bat  vendor (RadixArk) profile — dual-profile MODEL=vendor|loved
  run_sglang_loved.bat       lovedheart-only launcher (container sglang-loved, port 18082)
  run_sglang_27B_dflash2.bat   Qwen3.8-27B + DFlash2 profile (coexist image)
  run_primitive_ram.sh   primitive-ai checkpoint, PLE via native --ple-offload-embedding (host RAM, no SSD stream), stock official image
scripts/
  kv_quality_ab.py       fp8 KV vs bf16 KV needle tests (easy + hard profiles)
  bench_flash.py         prefill throughput sanity probe
  lovedheart/          one-off deploy kit: download, PLE byte-ident probe, mtp/ synthesis, deploy.sh
  primitive/mtp_synth.py  mtp/ synthesis for primitive-ai/Qwen3.8-Flash-Next-NVFP4 (RAM-offload profile)
docs/
  01-BUILD.md    how to build the image
  02-RUN.md    launch profiles and every flag that matters
  03-FINDINGS.md    performance and VRAM laws on this card
  04-DEBUG-LOG.md    what broke, why, how to avoid (JIT cliff, OOMs, dead ends)
  05-DATA.md    all measured numbers + env facts + open experiments
  06-PATCHES.md      what each source patch does
  07-PRUNED-vs-FULL.md   448E pruned vs 512E vendor: diffs, evals, trade-offs
  08-RESOURCES.md    every external link: commits, PRs, models, papers, validation kits
  09-PRIMITIVE-RAM-OFFLOAD.md   RAM-offloaded PLE profile (no SSD Stream): why no custom image is needed, validated numbers
```

## RAM-offloaded PLE profile (no SSD Stream, WSL2 Ubuntu)

The rest of this README describes the SSD-Stream setup. There's a second, simpler
profile for when NVMe streaming isn't wanted: `primitive-ai/Qwen3.8-Flash-Next-NVFP4`
served with sglang's native `--ple-offload-embedding` (the PLE table pins to host
RAM, ~95GiB resident by default — BF16, matching the checkpoint's shipped shards;
`PLE_DTYPE_OVERRIDE=0` keeps a smaller, uncalibrated fp8 downcast at ~48GiB instead
if RAM budget is tighter than ~100GB) — no SSD Stream plugin, on the stock
`lmsysorg/sglang:qwen38flashnext` image plus one small overlay
(`docker-target/Dockerfile.qsa-fp8-fix`) that backports three fixes: making
`--kv-cache-dtype fp8_e4m3` survive QSA's cached-prefix path, preventing a
transient GPU OOM when the BF16 PLE table is combined with
`--ple-offload-embedding`, and a staged (not-yet-built) re-port of #32468 that
frees the draft's duplicate embed/lm_head before KV-pool sizing. See `docs/09-PRIMITIVE-RAM-OFFLOAD.md` for what does
and doesn't need patching and what was validated; `config/run_primitive_ram.sh` +
`scripts/primitive/mtp_synth.py` to reproduce.

## Reproduce in 5 steps

1. Build the image (`docker build -t sglang-flash-27b:latest docker-target`) — see `docs/01-BUILD.md`.
2. Place models in `~/.cache/huggingface/hub` (HF_ENDPOINT=https://hf-mirror.com if GitHub/HF is slow).
3. For lovedheart: run the one-off deploy kit (`docs/02-RUN.md` §B) — index surgery, symlink PLE
   (byte-ident re-shard, zero copy), `mtp/` synthesis. 10 s.
4. Launch: `run_sglang_flash_next.bat` (vendor) or `run_sglang_loved.bat` (pruned).
5. Accept test:
```bash
curl.exe http://127.0.0.1:18081/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"temperature":0,"max_tokens":512}'
# -> "Paris" in content => FP8_PB_WO dispatch is live. Garbled text => patch not applied.
```

## The five non-negotiables

1. `--security-opt seccomp=unconfined` — SSD Stream's io_uring is blocked by the default seccomp.
2. **Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — hard driver crash on WSL2 (`docs/04-DEBUG-LOG.md` §3). Also never `docker rm` a running service container you did not start.
3. KV pool cannot exceed `max_position_embeddings=262,144`; `FRACTION=0.99` + the #32468 backport unlocks exactly that.
4. FP8_PB_WO layers need the routing patch (stock builds emit **silent garbled output**, no error).
5. KV dtype: fp8_e4m3 passes needle tests; bf16 KV only worth it if the hard profile (decoys + multi-hop) says otherwise.

Plus one housekeeping rule: keep the two cache mounts in every launcher — `jit_cache` (flashinfer
autotune) and `triton_cache` (`/root/.triton`) — so long-context never pays JIT again per restart
(`docs/04-DEBUG-LOG.md` §1.5).
