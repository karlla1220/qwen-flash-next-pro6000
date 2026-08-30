#!/bin/bash
# Download lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000 into HF cache layout.
# CPU-only container; hf_cache mounted at /root/.cache/huggingface.
set -u
REPO="lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000"
MIRROR="https://hf-mirror.com"
MAN=/probe/lovedheart/manifest.json
COMMIT="$(python3 -c "import json;print(json.load(open('$MAN'))['commit'])")"
ORG=$(echo "$REPO" | cut -d/ -f1 | tr '.' '-')
NAME=$(echo "$REPO" | cut -d/ -f2 | tr '.' '-')
D=/root/.cache/huggingface/hub/models--$ORG--$NAME
export D REPO MIRROR
mkdir -p "$D/blobs" "$D/snapshots/$COMMIT" "$D/refs"
echo "$COMMIT" >"$D/refs/main"

python3 - "$MAN" >/tmp/jobs.tsv <<'EOF'
import json, sys
man = json.load(open(sys.argv[1]))
for x in man["lfs"]:
    print(f"lfs\t{x['path']}\tblobs/sha256,{x['oid']}\t{x['size']}\t{x['oid']}")
for x in man["raw"]:
    print(f"raw\t{x['path']}\tblobs/{x['sha1']}\t{x['size']}\t{x['sha1']}")
EOF

dl_one() {
  local _type="$1" path="$2" rel="$3" size="$4"
  local out="$D/$rel"
  [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ] && return 0
  local url="$MIRROR/$REPO/resolve/main/$path"
  curl -fsSL --retry 8 --retry-delay 5 --connect-timeout 20 -o "$out.part" "$url" || {
    echo "$path" >>/tmp/dl_fail.log
    rm -f "$out.part"
    return 0
  }
  local got=$(stat -c%s "$out.part" 2>/dev/null || echo 0)
  if [ "$got" != "$size" ]; then
    echo "$path" >>/tmp/dl_fail.log
    rm -f "$out.part"
    return 0
  fi
  mv -f "$out.part" "$out"
}
export -f dl_one

echo "[total jobs] $(grep -c . /tmp/jobs.tsv)"
: >/tmp/dl_fail.log
cat /tmp/jobs.tsv | xargs -P 3 -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r t p r s h <<< "{}"; dl_one "$t" "$p" "$r" "$s"'

# serial retry of any failures
if [ -s /tmp/dl_fail.log ]; then
  echo "=== retrying $(wc -l </tmp/dl_fail.log) failures serially ==="
  while read -r p; do
    awk -F'\t' -v p="$p" '$2==p{printf "%s\t%s\t%s\n",$1,$3,$4}' /tmp/jobs.tsv |
      while IFS=$'\t' read -r t r s; do dl_one "$t" "$p" "$r" "$s"; done
  done </tmp/dl_fail.log
fi
echo "=== DOWNLOAD PHASE DONE blobs=$(find "$D/blobs" -maxdepth 1 -type f -not -name '*.part' | wc -l) ==="

# symlinks: snapshot path -> ../../blobs/<...>
COMMIT_DIR="$D/snapshots/$COMMIT"
while IFS=$'\t' read -r _type path rel size _hash; do
  mkdir -p "$COMMIT_DIR/$(dirname "$path")"
  depth=$(awk -F/ '{print NF-1}' <<<"$path")
  up=$(printf '../%.0s' $(seq 1 "$depth"))
  ln -sfn "${up}${rel}" "$COMMIT_DIR/$path"
done </tmp/jobs.tsv
echo "=== SYMLINKS DONE links=$(find "$COMMIT_DIR" -type l | wc -l) ==="

python3 - <<'EOF'
import hashlib, json, os
man = json.load(open("/probe/lovedheart/manifest.json"))
D = os.environ["D"]
bad, ok = [], 0
def hf(p, algo):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
    return h.hexdigest()
for x in man["lfs"]:
    p = f"{D}/blobs/sha256,{x['oid']}"
    try:
        ok += hf(p,"sha256")==x["oid"]
    except Exception:
        bad.append(x["path"])
for x in man["raw"]:
    p = f"{D}/blobs/{x['sha1']}"
    try:
        data=open(p,"rb").read()
        gh=hashlib.sha1(b"blob %d\0"%len(data)+data).hexdigest()
        ok += gh==x["sha1"]
    except Exception: bad.append(x["path"])
print(f"VERIFY ok={ok} bad={len(bad)}")
for b in bad[:30]: print("  BAD:", b)
EOF
echo "=== ALL DONE ==="
