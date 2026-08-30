@echo off
setlocal
rem ============================================================
rem SGLang + Qwen3.8-27B-FP8 + DFlash2 speculative decoding
rem Target : single RTX Pro 6000 Blackwell 96GB (SM120)
rem Recipe : SGLang cookbook rtx6000 + DFlash2 draft (incoai),
rem          block size 8 (7 draft tokens per verify step)
rem Tuning : 3 concurrent (1 main 262k + 2 subagents 128k), FP8 KV,
rem          mamba pool pinned to 32 slots (ratio flag removed)
rem Image  : lmsysorg/sglang:qwen38-27b-dflash2
rem          (sglang git main, includes DFlash2 PR #35371;
rem           build with F:\MyAI\deepseek\dflash2.ps1 build)
rem WARNING: removes sglang-qwen and KILLS the currently serving
rem          LLM endpoint. Run deliberately, from this terminal.
rem NOTE   : first start after switching images takes ~1-2 extra
rem          minutes (flashinfer 0.6.17 JIT kernel compile).
rem PATCH  : bind-mounts patches\compressed_tensors.py over the image copy
rem          (fixes weight-only FP8 compressed-tensors crash:
rem           "Other method (CompressedTensorsW4A16Sparse24) is not supported"
rem           + ignore-list name mismatch for model.language_model.* entries).
rem          Still present in sglang main as of 2026-08-23. Remove when fixed.
rem PATCH2 : bind-mounts patches\qwen3_5.py adding load_kv_cache_scales.
rem          ONLY for FP8 Pessoa model (uses its calibrated K/V scales via
rem          SGLANG_KV_SCALES_JSON). Disabled while running INT8 models.
rem PATCH3 : bind-mounts patches\gptq.py + patches\auto_round.py.
rem          auto_round.py: from_config never read packed_modules_mapping, so
rem          the fused in_proj_ba module (b+a = 96 rows) missed its BF16
rem          exclusion and crashed marlin repack ("size_n = 96 is not
rem          divisible by tile_n_size = 64") on
rem          Minachist/Qwen3.8-27B-INT8-AutoRound. Now wired through.
rem PORT   : host port 18081 (moved from 8081; use 127.0.0.1, never localhost
rem          - wslrelay.exe can hijack IPv6 loopback and black-hole requests).
rem DOCKERDISK: model weights + flashinfer JIT cache live on the Docker
rem          Desktop data disk (/mnt/docker-desktop-disk, ext4, inside the
rem          docker VM) for fast container I/O. WSL-path mounts from the
rem          Windows CLI resolve to EMPTY dirs (distro mount service
rem          missing in this setup), and F:\ NTFS via 9p only does
rem          ~155 MB/s. One-time setup: setup_docker_disk_cache.bat.
rem          If you UPDATE Minachist INT8 / incoai DFlash2 on
rem          F:\HuggingFaceCache, re-run that script first.
rem          NOTE: Docker Desktop Reset/Clean may wipe the data disk.
rem ============================================================

echo ^>^> Removing sglang-qwen (kills current LLM endpoint)...
docker rm -f sglang-qwen >nul 2>&1

echo.
echo ^>^> Starting SGLang (FP8 + DFlash2 block-8) on :18081 ...
docker run -d --name sglang-qwen --gpus all ^
  -p 18081:8000 ^
  --ipc=host ^
  -v /mnt/docker-desktop-disk/hf_cache:/root/.cache/huggingface ^
  -v "E:\Desktop\LLM\patches\gptq.py:/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/gptq/gptq.py:ro" ^
  -v "E:\Desktop\LLM\patches\auto_round.py:/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/auto_round.py:ro" ^
  -v /mnt/docker-desktop-disk/jit_cache:/root/.cache/flashinfer ^
  rem Persist Triton disk cache across container restarts (JIT bucket
  rem specializations compiled on first use, otherwise re-payed per launch).
  -v /mnt/docker-desktop-disk/triton_cache:/root/.triton ^
  -e "HF_ENDPOINT=https://hf-mirror.com" ^
  -e "SGLANG_DISABLE_CUDA_IPC=1" ^
  -e "CUDA_IPC_HANDLE_CACHE_DISABLE=1" ^
  lmsysorg/sglang:qwen38-27b-dflash2 ^
  sglang serve ^
  --trust-remote-code ^
  --model-path Minachist/Qwen3.8-27B-INT8-AutoRound ^
  --chat-template /root/.cache/huggingface/chat_templates/qwen3.8-froggeric-v22.3.jinja ^
  --served-model-name qwen3.8-27b ^
  --mm-feature-transport cpu ^
  --mem-fraction-static 0.92 ^
  --attention-backend flashinfer ^
  --chunked-prefill-size 4096 ^
  --reasoning-parser qwen3 ^
  --tool-call-parser qwen3_coder ^
  --max-running-requests 2 ^
  --max-mamba-cache-size 10 ^
  --speculative-algorithm DFLASH ^
  --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 ^
  --speculative-num-draft-tokens 8 ^
  --mamba-radix-cache-strategy extra_buffer ^
  --mamba-ssm-dtype float32 ^
  --host 0.0.0.0 --port 8000


if errorlevel 1 (
  echo.
  echo [FAILED] container start failed. Check: docker logs sglang-qwen
  pause
  exit /b 1
)

timeout /t 3 /nobreak >nul
echo.
echo ^>^> Container state:
docker ps -a --filter "name=sglang-qwen" --format "{{.Names}}  {{.Status}}"
echo.
echo ^>^> First logs (watch which phase it stalls in):
docker logs --tail 25 sglang-qwen
echo.
echo ============================================================
echo Next steps:
echo   watch startup : docker logs -f sglang-qwen
echo   health check  : curl.exe http://127.0.0.1:18081/health
echo   (always use 127.0.0.1, NOT localhost - see PORT note above;
echo   (wait until "The server is fired up" appears in logs;
echo    first DFlash2 start is slower - flashinfer JIT compile)
echo ============================================================
pause
