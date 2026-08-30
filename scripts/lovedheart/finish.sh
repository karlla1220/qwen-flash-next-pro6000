#!/bin/bash
# Finish phase: verify all blobs + create snapshot symlinks. Run inside container.
set -u
D=/root/.cache/huggingface/hub/models--lovedheart--Qwen3-8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000
COMMIT=0e63a89e8ce4b0ee1d909d4cbabf6898e43a8add
SNAP="$D/snapshots/$COMMIT"
mkdir -p "$SNAP"

python3 - <<'PYEOF'
import hashlib, json, os
man = json.load(open("/probe/lovedheart/manifest.json"))
D = "/root/.cache/huggingface/hub/models--lovedheart--Qwen3-8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000"
COMMIT = "0e63a89e8ce4b0ee1d909d4cbabf6898e43a8add"
SNAP = D + "/snapshots/" + COMMIT

def sha256_of(p):
    h = hashlib.new("sha256")
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()

bad, ok, missing = [], 0, []
for x in man["lfs"]:
    p = f"{D}/blobs/sha256,{x['oid']}"
    if not os.path.exists(p):
        missing.append(x["path"]); continue
    if sha256_of(p) == x["oid"]:
        ok += 1
    else:
        bad.append(x["path"])

for x in man["raw"]:
    p = f"{D}/blobs/{x['sha1']}"
    if not os.path.exists(p):
        missing.append(x["path"]); continue
    data = open(p, "rb").read()
    gh = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    if gh == x["sha1"]:
        ok += 1
    else:
        bad.append(x["path"])

print(f"VERIFY ok={ok} bad={len(bad)} missing={len(missing)}", flush=True)
for b in bad[:30]: print("  BAD:", b)
for b in missing[:30]: print("  MISSING:", b)

if not bad and not missing:
    import pathlib
    for x in man["lfs"] + man["raw"]:
        blob = f"{D}/blobs/sha256,{x['oid']}" if 'oid' in x else f"{D}/blobs/{x['sha1']}"
        dst = os.path.join(SNAP, x["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True) if os.path.dirname(dst) != SNAP else None
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.relpath(blob, os.path.dirname(dst)), dst)
    n = sum(1 for _ in pathlib.Path(SNAP).rglob("*"))
    print(f"SYMLINKS OK total_entries={n}")
    print("=== ALL DONE ===")
PYEOF
