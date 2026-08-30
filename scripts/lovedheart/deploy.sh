#!/bin/bash
# lovedheart deployment: verify new blobs -> snapshot links -> scale copy -> index surgery -> ple symlink -> ssd-stream.json
set -eu
HF=/root/.cache/huggingface/hub
LOV=$HF/models--lovedheart--Qwen3-8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000
VROOT=$HF/models--garnermccloud--Qwen3.8-Flash-Next-NVFP4-SSD-Stream
COMMIT=0e63a89e8ce4b0ee1d909d4cbabf6898e43a8add
SNAP=$LOV/snapshots/$COMMIT
VBIN_SHA=b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3

echo "== 1) verify the 3 retried blobs =="
python3 - <<'PY'
import hashlib, json, os
HF="/root/.cache/huggingface/hub"
D=HF+"/models--lovedheart--Qwen3-8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000"
man=json.load(open("/probe/lovedheart/manifest.json"))
want={"layer-00008-experts-0256-0383.safetensors","layer-00025-experts-0384-0511.safetensors","layer-00027-experts-0128-0255.safetensors"}
def sha256_of(p):
    h=hashlib.new("sha256")
    with open(p,"rb") as f:
        for c in iter(lambda: f.read(1<<24), b""): h.update(c)
    return h.hexdigest()
ok=True
for x in man["lfs"]:
    if x["path"] in want:
        p=f"{D}/blobs/sha256,{x['oid']}"
        got=sha256_of(p) if os.path.exists(p) else None
        good=got==x["oid"]; ok=ok and good
        print(("OK  " if good else "BAD "),x["path"])
raise SystemExit(0 if ok else 1)
PY

echo "== 2) snapshot symlinks (all files) =="
python3 - <<'PY'
import json, os
D="/root/.cache/huggingface/hub/models--lovedheart--Qwen3-8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000"
SNAP=D+"/snapshots/0e63a89e8ce4b0ee1d909d4cbabf6898e43a8add"
man=json.load(open("/probe/lovedheart/manifest.json"))
os.makedirs(SNAP,exist_ok=True)
n=0
for x in man["lfs"]+man["raw"]:
    blob=f"{D}/blobs/sha256,{x['oid']}" if "oid" in x else f"{D}/blobs/{x['sha1']}"
    assert os.path.exists(blob), "missing blob "+blob
    dst=os.path.join(SNAP,x["path"])
    d=os.path.dirname(dst)
    if d!=SNAP: os.makedirs(d,exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst): os.remove(dst)
    os.symlink(os.path.relpath(blob,d),dst); n+=1
print("links:",n)
PY

echo "== 3) scale file + index surgery + ple symlink + ssd-stream.json =="
python3 - <<PY
import json, os
HF="/root/.cache/huggingface/hub"
SNAP=HF+"/models--lovedheart--Qwen3-8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000/snapshots/0e63a89e8ce4b0ee1d909d4cbabf6898e43a8add"
VSNAP=HF+"/models--garnermccloud--Qwen3.8-Flash-Next-NVFP4-SSD-Stream/snapshots/83325b75b7cb498ef5d7a5477171cadf92ad21f5"
VBLOB=HF+"/models--garnermccloud--Qwen3.8-Flash-Next-NVFP4-SSD-Stream/blobs/sha256,$VBIN_SHA"

# 3a. vendor scale file copied in (byte-identical scale)
src=os.path.join(VSNAP,"model-plefp8-scale.safetensors")
dst=os.path.join(SNAP,"model-plefp8-scale.safetensors")
open(dst,"wb").write(open(src,"rb").read())
print("scale file copied:",os.path.getsize(dst),"bytes")

# 3b. index surgery
idx_path=os.path.join(SNAP,"model.safetensors.index.json")
assert os.path.islink(idx_path)
idx=json.load(open(os.path.realpath(idx_path)))
wm=idx["weight_map"]
removed=0; freed=0
for k in list(wm):
    if ".ngram_embedding.shard_" in k and k.endswith(".weight"):
        del wm[k]; removed+=1
SCALE_KEY="model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
assert SCALE_KEY in wm, "scale tensor missing from index"
wm[SCALE_KEY]="model-plefp8-scale.safetensors"
idx["metadata"]["total_size"]-=removed*2500012*160
if os.path.islink(idx_path): os.remove(idx_path)
json.dump(idx,open(idx_path,"w"),indent=2)
print("index: removed",removed,"shard entries; total_size now",idx["metadata"]["total_size"])

# 3c. ple bin symlink (relative, into vendor blobs)
os.makedirs(os.path.join(SNAP,"ple"),exist_ok=True)
pl=os.path.join(SNAP,"ple","qwen3.8-flash-next-ple-fp8.bin")
if os.path.islink(pl) or os.path.exists(pl): os.remove(pl)
os.symlink(os.path.relpath(VBLOB,os.path.dirname(pl)),pl)
print("ple link ->",os.readlink(pl))

# 3d. ssd-stream.json mirroring vendor table
tbl={"format":"sglang-ssd-stream","source":{"model":"lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000","revision":"0e63a89e8ce4b0ee1d909d4cbabf6898e43a8add"},
 "tables":[{"bytes":51200245760,"columns":160,"dtype":"float8_e4m3fn","layer":0,
   "path":"ple/qwen3.8-flash-next-ple-fp8.bin","row_start":0,"rows":320001536,
   "sha256":"$VBIN_SHA"}],"version":1}
json.dump(tbl,open(os.path.join(SNAP,"ssd-stream.json"),"w"),indent=2)
print("ssd-stream.json written")

# 3e. final sanity: every index file exists in snapshot
missing=[f for f in set(wm.values()) if not os.path.exists(os.path.join(SNAP,f))]
print("index-referenced files missing:",missing)
PY

echo "=== DEPLOY READY ==="
