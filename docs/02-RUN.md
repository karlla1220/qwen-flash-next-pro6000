# 02 — Running & deploying

## A. Vendor model (RadixArk / garnermccloud SSD-Stream rep)

1. Build image (01-BUILD).
2. Model is in HF cache (`hf-mirror.com` mirror recommended for GitHub/HF access:
`HF_ENDPOINT=https://hf-mirror.hf-mirror.com` + `hf download garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream`).
3. `run_sglang_flash_next.bat` (host port 18081). It frees VRAM by stopping the other
   local servers (e.g. ninfer), then starts the Flash-Next profile.

Key flags (all validated against the pinned tree):

```
--quantization modelopt_fp4
--moe-runner-backend flashinfer_cutlass          (auto resolves to TRTLLM for some configs → crashes NVFP4 MoE)
--kv-cache-dtype fp8_e4m3 --page-size 64
--mamba-radix-cache-strategy extra_buffer_lazy --mamba-track-interval 64
--max-running-requests 1 --chunked-prefill-size 2048 --mem-fraction-static 0.99 --max-total-tokens 262144
--speculative-algorithm NEXTN --speculative-draft-model-path <snap>/mtp
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--speculative-moe-runner-backend flashinfer_cutlass --speculative-draft-model-quantization fp8
--ple-offload-embedding                                   # the whole reason 262K fits: PLE table off the GPU
--host 0.0.0.0 --port 8000
```
Container side: `-d --gpus all --ipc=host --security-opt seccomp=unconfined` (io_uring).

## B. lovedheart (448E pruned) — one-off deploy kit in `scripts/lovedheart/`

The model repo flattens MTP into the root index and stores the PLE table as 128 BF16
shards, so `mtp/` and a streamable PLE bin do not exist. The kit fixes both:

```bash
# 1. (if not already downloaded) scripts/lovedheart/download.sh   → HF cache hub layout
# 2. ple_probe.py: byte-identity check of the loved shards vs the vendor fp8 bin
#   (result: 128 shards re-shard the vendor bin byte-perfectly → symlink, zero copy)
# 3. deploy.sh (idempotent):
#      - index surgery: drop the 128 ple-bf16 entries, weight_scale -> local scale file
#      - symlink  ple/qwen3.8-flash-next-ple-fp8.bin  -> vendor fp8 bin  (47.68 GiB)
#      - write ssd-stream.json (same table params as vendor)
#      - sanity: index-referenced files missing = []
# 4. mtp_synth.py: builds <snap>/mtp/ (config from vendor mtp config with num_experts
#        512→448, index = 31 mtp.* keys -> the 3 flattened root files). NEXTN then works
#      ("The MTP layer is pruned to 448 as well — --speculative-algo NEXTN remains usable")
```

Then `run_sglang_loved.bat` (its own container `sglang-loved`, port 18082, `MODEL=loved`,
`--quantization modelopt_mixed` explicit — the checkpoint ships no `hf_quant_config.json`,
so stock auto-detection fails).

Accept test (same as README) must return **`Paris`** — stock builds of FP8_PB_WO models
garble silently; our patched image must not.

## C. Dual-profile switch

Inside one bat, top-of-file `MODEL=vendor|loved` switches: snapshot path, served name,
quant method (`modelopt_fp4` vs explicit `modelopt_mixed`), PLE offload flags. All else identical.
