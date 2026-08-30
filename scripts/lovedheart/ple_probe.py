import json, os, struct

LOV_B = "/root/.cache/huggingface/hub/models--lovedheart--Qwen3-8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000/blobs"
VROOT = "/root/.cache/huggingface/hub/models--garnermccloud--Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
VEN = VROOT + "/snapshots/" + open(VROOT + "/refs/main").read().strip()
VB = os.path.realpath(os.path.join(VEN, "ple", "qwen3.8-flash-next-ple-fp8.bin"))
ROWS_PER_SHARD = 2500012


def hdr(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


man = json.load(open("/probe/lovedheart/manifest.json"))
blob = lambda oid: os.path.join(LOV_B, "sha256," + oid)
ple = sorted([y for y in man["lfs"] if "plefp8" in y["path"]], key=lambda y: y["path"])

# fingerprint: 160 bytes at row 7777 of chosen shards
fp = {}
for idx in [0, 1, 63, 127]:
    for x in ple:
        p = blob(x["oid"])
        h, o = hdr(p)
        hit = False
        for k, v in h.items():
            if k.endswith("shard_%d.weight" % idx):
                with open(p, "rb") as f:
                    f.seek(o + v["data_offsets"][0] + 7777 * 160)
                    fp[idx] = f.read(160)
                hit = True
                break
        if hit:
            break

print("fingerprints:", {i: fp[i][:8].hex() for i in sorted(fp)})

needles = {fp[i]: i for i in fp}
found = {}
CH = 1 << 26
tail = b""
pos = 0
with open(VB, "rb") as f:
    while True:
        b = f.read(CH)
        if not b:
            break
        buf = tail + b
        base = pos - len(tail)
        for n, i in needles.items():
            if i in found:
                continue
            j = buf.find(n)
            if j >= 0:
                found[i] = base + j
        tail = b[-160:]
        pos += len(b)

for i in [0, 1, 63, 127]:
    exp = i * ROWS_PER_SHARD * 160
    got = found.get(i)
    if got == exp:
        verdict = "IDENTICAL_ORDER"
    elif got is not None:
        verdict = "MOVED"
    else:
        verdict = "NOT_FOUND_requantized"
    print("shard%-4d found_at=%s expected_if_concat=%s %s" % (i, got, exp, verdict))
