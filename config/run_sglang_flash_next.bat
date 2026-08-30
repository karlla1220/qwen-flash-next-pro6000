@echo off
setlocal
rem ============================================================
rem SGLang + Qwen3.8-Flash-Next + SSD Stream PLE
rem Target : single RTX Pro 6000 Blackwell 96GB (SM120)
rem Profile: EXACT validated RTX PRO 6000 profile from
rem          sglang-ssd-stream cli.py (_rtx_pro_args):
rem          131K context, FP8 KV, MTP 3/1/4, 1 request.
rem          (262K context is DGX Spark-only, do not use here.)
rem Recipe : garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream
rem          (NVFP4 experts + bf16 dense/attention + MTP;
rem           47.68GiB FP8 PLE table streams from SSD via
rem           the ssd_stream plugin - only ~64MiB resident)
rem Image  : sglang-flash-27b:latest
rem          (sglang pinned to 3df8e1e7 - vendor-validated commit;
rem           DFlash2 backported; local quant patches applied;
rem           ssd_stream plugin no-ops without ssd-stream.json)
rem PORT   : host port 18081 (container port 8000)
rem NOTE   : needs --security-opt seccomp=unconfined (io_uring).
rem DOCKERDISK: model weights live on the Docker Desktop data disk
rem          (/mnt/docker-desktop-disk, ext4) - standard HF cache
rem          layout, 227/227 files hash-verified.
rem          NOTE: Docker Desktop Reset/Clean may wipe the data disk.
rem ============================================================

set "NAME=sglang-flash"
set "PORT=18081"
rem CTX: with DRAFT8=on the KV pool reaches the native 262144 cap
rem (#32468 backport + FRACTION=0.99). bf16 draft caps out at ~116K.
rem KV FP8 cost: 100K=1.17GB, 262K=3.00GB
set "CTX=262144"
rem POOL: KV pool size (max_total_tokens). Defaults to CTX. For concurrency
rem >1, set POOL ~= MAXREQ * expected per-session length so several long
rem sessions coexist; CTX stays the per-session cap (native max 262144).
rem Example dual-262K: MAXREQ=2 SPEC=off CTX=262144 POOL=524288

set "POOL=262144"
if not defined POOL set "POOL=%CTX%"
rem MAXREQ: concurrent running requests. Each request costs
rem MAMBA slots (5 with MTP / 4 without) and shares the CTX-sized KV
rem pool, so with MAXREQ=2 keep CTX<=200000 (~100K per session).
rem Mamba VRAM grows ~1.45GB per extra concurrent request (MTP).
rem Set MAXREQ=2 to try dual concurrency (DRAFT8=on strongly advised:
rem bf16 draft steals 3.5GB that the extra mamba slots need).
set "MAXREQ=1"
rem SPEC: on = MTP + FP8-quantized draft (~165 tok/s, draft costs ~3.4GB
rem        instead of 6.66GB; 200K context feasible).
rem        If draft fp8 quant fails at load, set DRAFT8=off (fallback
rem        to bf16 draft, then CTX must drop to ~100000).
rem      off = no MTP, max VRAM for KV (~90-130 tok/s).
set "SPEC=on"
set "DRAFT8=on"
rem hicache: host-RAM prefix cache. Sized for the 30GB Docker VM
rem (VM cap = 50pct of 61.6GB RAM, no .wslconfig). 12GB ~ 1M tokens
rem of FP8-KV prefix cache. Do not exceed 16.
set "HICACHE=12"
rem Pool sizing: 0.99 raises KV budget by ~0.9GB, enough to give the full
rem requested CTX (was clamped to 162880 at 0.985). Graph-capture peak
rem headroom measured 3.25GB at capture start, so 0.99 keeps ~2.3GB margin.
rem Do NOT go past 0.99 (131K profile OOM'd at Docker overhead).
set "FRACTION=0.99"
rem MAMBA slots: MTP needs num_draft_tokens+1 = 5 slots per running
rem request (4 with SPEC=off); total = slots_per_req * MAXREQ. 4 slots
rem with SPEC=on serves 0 requests ("Can not alloc mamba cache").
rem Predefine MAMBA to override the auto value.
if not defined MAMBA (
  if /i "%SPEC%"=="on" ( set /a "MAMBA=5*MAXREQ" ) else ( set /a "MAMBA=4*MAXREQ" )
)
rem !! HICACHE DISABLED (2026-08-29): sglang 3df8e1e7 transfer_mamba backup
rem    kernel (kvcacheio/transfer_mamba.cuh:184) hits illegal memory access
rem    with MTP packed-KV + extra_buffer_lazy. Re-enable only after
rem    upstream fix; GPU-side radix prefix cache still works meanwhile.
rem MODEL: vendor = garnermccloud NVFP4-SSD-Stream (validated, current)
rem        loved  = lovedheart AIMER-pruned 448-expert NVFP4+FP8_PB_WO
rem                 (needs rebuilt image w/ FP8_PB_WO dispatch patch;
rem                  PLE served from the vendor bin via our ssd-stream.json,
rem                  snapshot deployed by probe/lovedheart/deploy.sh)
set "MODEL=vendor"
set "CACHE=/root/.cache/huggingface"
if /i "%MODEL%"=="loved" (
  set "SNAP=%CACHE%/hub/models--lovedheart--Qwen3-8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000/snapshots/0e63a89e8ce4b0ee1d909d4cbabf6898e43a8add"
  set "SERVED=Qwen3.8-Flash-Next-Pruned"
  set "PLE_FLAGS=--ple-offload-embedding"
  rem lovedheart ships config.json-only MIXED_PRECISION (FP8_PB_WO layers):
  rem must route explicitly through modelopt_mixed, NOT modelopt_fp4.
  set "QUANT=modelopt_mixed"
) else (
  set "SNAP=%CACHE%/hub/models--garnermccloud--Qwen3.8-Flash-Next-NVFP4-SSD-Stream/snapshots/83325b75b7cb498ef5d7a5477171cadf92ad21f5"
  set "SERVED=Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
  set "PLE_FLAGS="
  set "QUANT=modelopt_fp4"
)

echo ^>^> Stopping other GPU backends (Flash-Next needs the whole card)...
docker stop ninfer >nul 2>&1
docker stop sglang-qwen >nul 2>&1

rem === Wait for GPU VRAM to fully release before starting ===
rem docker stop returns as soon as the process exits, but the CUDA context / VRAM
rem is freed by the driver a few seconds later (slow on WSL2+WDDM). Starting a new
rem CUDA app before that causes: "CUDA driver error: unknown error" at torch.ones.

set "SPEC_FLAGS="
if /i "%SPEC%"=="on" (
  set "SPEC_FLAGS=--speculative-algorithm NEXTN --speculative-draft-model-path %SNAP%/mtp --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3"
)
if /i "%SPEC%%DRAFT8%"=="onon" (
  set "SPEC_FLAGS=%SPEC_FLAGS% --speculative-draft-model-quantization fp8"
  echo ^>^> MTP on, draft quantized to FP8 (saves ~3.3GB VRAM)
) else if /i "%SPEC%"=="on" (
  echo ^>^> MTP on, bf16 draft (needs CTX ~= 100000)
) else (
  echo ^>^> MTP speculative decoding disabled (frees ~6.66GB VRAM for KV)
)

echo ^>^> Starting SGLang (Flash-Next + SSD Stream, validated profile) on :%PORT% ...
docker rm -f %NAME% >nul 2>&1
docker run -d --name %NAME% --gpus all ^
  -p %PORT%:8000 ^
  --ipc=host ^
  --security-opt seccomp=unconfined ^
  -v /mnt/docker-desktop-disk/hf_cache:/root/.cache/huggingface ^
  -v /mnt/docker-desktop-disk/jit_cache:/root/.cache/flashinfer ^
  rem Persist Triton disk cache: long-context bucket specializations (QSA
  rem layout / mamba scatter / track-commit kernels) are compiled on first
  rem use mid-request otherwise (prefill tail drops to ~1K tok/s).
  -v /mnt/docker-desktop-disk/triton_cache:/root/.triton ^
  -e "HF_ENDPOINT=https://hf-mirror.com" ^
  -e "SGLANG_DISABLE_CUDA_IPC=1" ^
  -e "CUDA_IPC_HANDLE_CACHE_DISABLE=1" ^
  sglang-flash-27b:latest ^
  sglang serve ^
  --trust-remote-code ^
  --model-path %SNAP% ^
  --served-model-name %SERVED% ^
  --quantization %QUANT% ^
  --fp4-gemm-backend flashinfer_cutlass ^
  --kv-cache-dtype fp8_e4m3 ^
  --page-size 64 ^
  --mamba-radix-cache-strategy extra_buffer_lazy ^
  --mamba-track-interval 64 ^
  --mamba-ssm-dtype float32 ^
  --max-mamba-cache-size %MAMBA% ^
  --chunked-prefill-size 2048 ^
  --max-running-requests %MAXREQ% ^
  --cuda-graph-max-bs-decode %MAXREQ% ^
  --context-length %CTX% ^
  --max-total-tokens %POOL% ^
  --mem-fraction-static %FRACTION% ^
  %PLE_FLAGS% ^
  %SPEC_FLAGS% ^
  --allow-auto-truncate ^
  --enable-multimodal ^
  --reasoning-parser auto ^
  --tool-call-parser qwen3_coder ^
  --sampling-defaults model ^
  --host 0.0.0.0 --port 8000

if errorlevel 1 (
  echo.
  echo [FAILED] container start failed. Check: docker logs %NAME%
  pause
  exit /b 1
)

timeout /t 3 /nobreak >nul
echo.
echo ^>^> Container state:
docker ps -a --filter "name=%NAME%" --format "{{.Names}}  {{.Status}}"
echo.
echo ^>^> First logs (watch which phase it stalls in):
docker logs --tail 25 %NAME%
echo.
echo ============================================================
echo Next steps:
echo   watch startup : docker logs -f %NAME%
echo   health check  : curl.exe http://127.0.0.1:%PORT%/health
echo   test request  : curl.exe http://127.0.0.1:%PORT%/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen3.8-Flash-Next-NVFP4-SSD-Stream\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with: SSD Stream works\"}],\"max_tokens\":32}"
echo   (always use 127.0.0.1, NOT localhost - wslrelay can hijack)
echo   (wait until "The server is fired up"; first start compiles
echo    flashinfer kernels - slower)
echo Rollback (back to 27B):
echo   docker stop %NAME% ^&^& start switch-to-ninfer.bat
echo   (or run run_sglang_dflash2.bat for SGLang 27B)
echo ============================================================
pause
