import json, os, shutil

S = "/root/.cache/huggingface/hub/models--lovedheart--Qwen3-8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000/snapshots/0e63a89e8ce4b0ee1d909d4cbabf6898e43a8add"
V = "/root/.cache/huggingface/hub/models--garnermccloud--Qwen3.8-Flash-Next-NVFP4-SSD-Stream/snapshots/83325b75b7cb498ef5d7a5477171cadf92ad21f5"
M = os.path.join(S, "mtp")
os.makedirs(M, exist_ok=True)


def sub512(o):
    n = 0
    if isinstance(o, dict):
        for k in list(o):
            if k == "num_experts" and o[k] == 512:
                o[k] = 448; n += 1
            else:
                r = sub512(o[k])
                if isinstance(r, int): n += r
        return n
    if isinstance(o, list):
        for x in o:
            r = sub512(x)
            if isinstance(r, int): n += r
        return n
    return n


# 1. config: vendor mtp config, 512 -> 448 everywhere
cfg = json.load(open(os.path.join(V, "mtp", "config.json")))
repl = sub512(cfg)
json.dump(cfg, open(os.path.join(M, "config.json"), "w"), indent=2)
print("config.json written, num_experts 512->448 x", repl)
# verify no stray 512 expert counts
def find512(o, path=""):
    hits = []
    if isinstance(o, dict):
        for k, v in o.items():
            p = path + "." + k
            if v == 512: hits.append(p)
            hits += find512(v, p)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            hits += find512(v, path + "[%d]" % i)
    return hits
print("remaining literal-512 fields:", find512(cfg))

# 2. index: mtp.* subset from main snapshot index (already surgery'd)
idx = json.load(open(os.path.join(S, "model.safetensors.index.json")))
mtp_map = {k: v for k, v in idx["weight_map"].items() if k.startswith("mtp.")}
assert len(mtp_map) == 31, len(mtp_map)
json.dump({"metadata": {"total_size": 0}, "weight_map": mtp_map},
          open(os.path.join(M, "model.safetensors.index.json"), "w"), indent=2)
print("index.json written, keys:", len(mtp_map))

# 3. weight files: copy (real files) - draft loads from its own dir
for f in sorted(set(mtp_map.values())):
    src = os.path.realpath(os.path.join(S, f))
    dst = os.path.join(M, f)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(src, M), dst)
    print("weight link", f, "->", os.readlink(dst), os.path.getsize(src) // 2**20, "MiB")

# 4. small companion files copied from vendor mtp (tokenizer etc.)
for f in ["chat_template.jinja", "generation_config.json", "hf_quant_config.json",
          "merges.txt", "tokenizer_config.json", "tokenizer.json",
          "preprocessor_config.json", "video_preprocessor_config.json", "vocab.json"]:
    src = os.path.join(V, "mtp", f)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(M, f))

# 5. vendor mtp hf_quant_config may map 512-expert layers; check + neuter if so
hq = os.path.join(M, "hf_quant_config.json")
if os.path.exists(hq):
    j = json.load(open(hq))
    q = j.get("quantization", j)
    print("mtp hf_quant_config:", q.get("quant_algo"), "layers:", len(q.get("quantized_layers", {})))

print("=== MTP DIR READY:", sorted(os.listdir(M)))
