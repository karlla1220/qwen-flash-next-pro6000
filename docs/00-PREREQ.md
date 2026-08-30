# 00 — Host prerequisites (read before building or launching anything)

## Hardware this playbook targets
- GPU: one Blackwell (SM120) card ≥ 96 GB VRAM (we use RTX PRO 6000 96GB). The 262K context
 profile only fits with PLE streamed off GPU (see `05-DATA.md` §3). Anything smaller → fp8 KV +
 `MAXREQ=1` at a fraction of the context; anything larger → raise `FRACTION` first, then KV pool.
- System RAM: ≥ 64 GB (weights stream through host during load; the PLE table never enters VRAM
 but the container reads it via io_uring O_DIRECT).
- NVMe on the same volume as the model weights (io_uring O_DIRECT on network mounts = disaster).

## Software stack
- Docker Desktop with WSL2 backend (this machine). bat files assume: data-disk path
 `/mnt/docker-desktop-disk` (Docker Desktop's own ext4 volume, NOT a host mount),
 `--ipc=host`, and `--security-opt seccomp=unconfined` (io_uring is blocked by default seccomp —
 `04-DEBUG-LOG.md` §4).
- NVIDIA driver on the host matching the image's CUDA 13 wheel set; no `cuda-toolkit` needed on host.
- Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (driver crash on WSL2 — `04 §3`).

## Disk budget (free, on the Docker Desktop data disk)
| Item | ~Size |
|---|---|
| Vendor 512E profile (weights + PLE bin 47.68 GiB included) | 135 GB |
| lovedheart 448E profile (PLE = symlink to vendor, 0 copies) | 135 GB |
| Triton cache dir (grows once) | 100–300 MB |
| flashinfer autotune cache (grows once) | tens of MB |
| HF download staging (`hf download` blobs) | up to 2× the model before cleanup |
**Plan for ~150 GB** per profile, + staging headroom. Re-shard path (`deploy.sh`) symlinks the PLE
table so it costs no extra space.

## Portability notes
- The three launchers are Windows cmd (.bat). On a Linux host: drop `^` line continuations → `\`,
 swap `/mnt/docker-desktop-disk/*` bind sources to your own paths, keep every flag identical.
- If you are on a different Docker Desktop machine, the data-disk root may live on another drive —
 check with `wsl -d docker-desktop ls /mnt` (paths in bats are VM-internal Linux paths).
- Host directory bootstrap (first run): `mkdir -p /mnt/docker-desktop-disk/{hf_cache,jit_cache,triton_cache}`
 inside the docker-desktop WSL dist.

## Known external dependencies (links in `08-RESOURCES.md`)
- Base image tag `lmsysorg/sglang:qwen38-27b-dflash2` — if it disappears, rebuild FROM the nearest
 `lmsysorg/sglang:*` CUDA-13 tag and re-apply layer 1 (sglang pin + dependency versions) unchanged.
- GitHub archive tarball for the pinned commit; flashinfer.ai whl indices; model repos on
 huggingface.co (mirror: hf-mirror.com).

## Reproduce checklist (the 5 steps of README in order)
1. `00` (this file) prerequisites green → 2. `01-BUILD.md` (build + import checks) →
3. model placement (`02` A/B: download, then lovedheart deploy.sh if using 448E) →
4. launch the right bat → 5. `Paris` probe + `scripts/bench_flash.py` prefill ~9k, decode ≈
05-DATA §1 band.
