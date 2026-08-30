#!/bin/bash
set -u
D=/root/.cache/huggingface/hub/models--lovedheart--Qwen3-8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000
for P in layer-00008-experts-0256-0383.safetensors layer-00025-experts-0384-0511.safetensors layer-00027-experts-0128-0255.safetensors; do
  OID=$(python3 -c "
import json
m=json.load(open('/probe/lovedheart/manifest.json'))
print([x['oid'] for x in m['lfs'] if x['path']=='$P'][0])")
  echo "GET $P -> sha256,$OID"
  curl --http1.1 -fSL --retry 5 --retry-all-errors --retry-delay 3 -C - \
    -o "$D/blobs/sha256,$OID.part" \
    "https://hf-mirror.com/lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000/resolve/main/$P" \
  && mv "$D/blobs/sha256,$OID.part" "$D/blobs/sha256,$OID" \
  && echo "DONE $P" || echo "FAIL $P"
done
