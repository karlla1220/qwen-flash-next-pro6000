#!/usr/bin/env python3
"""Build <snap>/mtp/ for primitive-ai/Qwen3.8-Flash-Next-NVFP4 so
--speculative-draft-model-path can point at it.

Unlike scripts/lovedheart/mtp_synth.py, this checkpoint is the full,
unpruned 512-expert model (same num_experts as the base architecture), so no
512->448 remap is needed: mtp/config.json is just an unmodified copy of the
root config.json (mirrors garnermccloud's repackaged mtp/config.json, which
is also a verbatim copy of its own root config for the same reason).

Usage: python3 mtp_synth.py <snapshot_dir>
  (snapshot_dir = the flat directory `hf download --local-dir` produced,
   e.g. ~/models/Qwen3.8-Flash-Next-NVFP4)
"""
import json
import os
import shutil
import sys

S = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/models/Qwen3.8-Flash-Next-NVFP4"))
M = os.path.join(S, "mtp")
os.makedirs(M, exist_ok=True)

# 1. config: verbatim copy, num_experts already 512 (unpruned) so no remap
shutil.copyfile(os.path.join(S, "config.json"), os.path.join(M, "config.json"))
print("config.json copied verbatim (num_experts unchanged)")

# 2. index: mtp.* / model.mtp.* subset from the root index
idx = json.load(open(os.path.join(S, "model.safetensors.index.json")))
mtp_map = {k: v for k, v in idx["weight_map"].items()
           if k.startswith("mtp.") or k.startswith("model.mtp.")}
print("mtp weight_map keys found:", len(mtp_map))
assert len(mtp_map) > 0, "no mtp.* keys found in root index - checkpoint layout changed?"
total = sum(
    os.path.getsize(os.path.join(S, f))
    for f in set(mtp_map.values())
) if mtp_map else 0
json.dump({"metadata": {"total_size": 0}, "weight_map": mtp_map},
          open(os.path.join(M, "model.safetensors.index.json"), "w"), indent=2)
print("index.json written, keys:", len(mtp_map))

# 3. weight files: symlink real shards (draft loads from its own dir)
for f in sorted(set(mtp_map.values())):
    src = os.path.realpath(os.path.join(S, f))
    dst = os.path.join(M, f)
    assert os.path.exists(src), f"missing shard referenced by index: {f}"
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(src, M), dst)
    print("weight link", f, "->", os.readlink(dst),
          os.path.getsize(src) // 2**20, "MiB")

# 4. companion files copied from the same checkpoint's own root
for f in ["chat_template.jinja", "generation_config.json", "hf_quant_config.json",
          "merges.txt", "tokenizer_config.json", "tokenizer.json",
          "preprocessor_config.json", "video_preprocessor_config.json", "vocab.json"]:
    src = os.path.join(S, f)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(M, f))

# 5. sanity: every index-referenced file exists in mtp/
missing = [f for f in set(mtp_map.values())
           if not os.path.exists(os.path.join(M, f))]
print("index-referenced files missing:", missing)
assert not missing

print("=== MTP DIR READY:", sorted(os.listdir(M)))
