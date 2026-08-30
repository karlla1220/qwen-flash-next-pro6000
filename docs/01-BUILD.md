# 01 — Building the image `sglang-flash-27b:latest`

One command:

```bash
cd docker-target && docker build -t sglang-flash-27b:latest .
```

Windows Git Bash, Docker Desktop + WSL2: same command, `cd docker-target` and run it.
If the last layers (COPY/RUN syntax) are cached by buildx, only the changed layers rebuild.

## What the image does (layer map of the Dockerfile)

| Layer | What | Why |
|---|---|---|
| 0 Base | `lmsysorg/sglang:qwen38-27b-dflash2` (CUDA 13.0.3) | a base that already contains CUDA13 runtime for the Blackwell wheels |
| 1 sglang pin | pip install `sglang @ .../archive/3df8e1e7...tar.gz` + nvidia cuda 13 wheels + `flashinfer-python==0.6.17` (and matching `flashinfer-cubin/jit-cache` `cu130`) | the only commit where qwen4_exp + SSD Stream + NEXTN are all functional |
| 2 SSD Stream | wheel `sglang_ssd_stream-0.1.0` (sglang plugin entry point) + `huggingface-hub` | streams the PLE n-gram table from NVMe, 0 VRAM, io_uring O_DIRECT |
| 3 DFlash2 backport | 5 dflash_* files copied from main `41c018a9ec` | enables the *27B + DFlash2* profile in the same image (this card also runs Qwen3.8-27B) |
| 4 patches | COPY of 9 source files (see `06-PATCHES.md`) + syntax-check each file | FP8_PB_WO routing; #32468; mamba radix + spec_utils stability fixes |
| 5 coexist patch | ssd_stream config rewrite for models without `ssd-stream.json` (27B) | lets one image serve both model families without re-launching |

Runtime layout inside the image (authoritative):

```
/usr/local/lib/python3.12/dist-packages/
    sglang/          ← pip installed from the pinned tarball, NOT the checkout
    sglang_ssd_stream/
    flashinfer/ …
/sgl-workspace/sglang   ← leftover base-image git checkout (main branch)
```
> **Important:** always take `dist-packages` as the patch baseline. `/sgl-workspace` is main, and patching against it silently produces a file with the wrong tree (e.g. SM120 GEMV, no NvFp4EmbeddingMethod). We learned this the hard way (see `04-DEBUG-LOG.md` #3).

Verify after build (no GPU needed):

```bash
docker run --rm --entrypoint python sglang-flash-27b:latest -c "import inspect; from sglang.srt.layers.quantization import modelopt_quant as m; print('FP8_PB_WO dispatch:', 'fp8_pb_wo_config' in inspect.getsource(m))"
docker run --rm --entrypoint python sglang-flash-27b:latest -c "import inspect; from sglang.srt.speculative import eagle_worker_v2 as e; print('mtp embed defered:', 'free_embs' in inspect.getsource(e) or 'free' in inspect.getsource(e))"
```
