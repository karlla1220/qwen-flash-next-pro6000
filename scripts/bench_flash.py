"""Flash-Next concurrency bench: lengths x concurrency, wall-clock tok/s."""
import json, time, urllib.request, sys
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:18081/v1/chat/completions"
SERVED = "Qwen3.8-Flash-Next-Pruned"
GEN = 128
WORDS = "apple banana cherry date elder fig grape melon nut orange pear quince raisin".split()

def prompt(seed, tokens):
    # ~1 token/word; unique filler per (seed) to defeat radix sharing across lanes
    n = max(0, tokens - 4)
    body = " ".join(WORDS[(i + seed) % len(WORDS)] for i in range(n))
    return "[%d] %s. Question: What fruit is named after a note in music? Answer in one word." % (seed, body)

def one(args):
    seed, tokens = args
    body = json.dumps({"model": SERVED, "messages": [{"role": "user", "content": prompt(seed, tokens)}],
                    "temperature": 0.0, "max_tokens": GEN, "stream": False}).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=360).read())
    dt = time.perf_counter() - t0
    u = r["usage"]
    return u["prompt_tokens"], u["completion_tokens"] if u.get("completion_tokens") else GEN, dt, r.get("id")

def run(L, C):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(C) as ex:
        res = list(ex.map(one, [(i * 7 + 1, L) for i in range(C)]))
    wall = time.perf_counter() - t0
    pt = sum(x[0] for x in res) / len(res)
    gt = sum(x[1] for x in res)
    per_avg = sum(x[2] for x in res) / len(res)
    agg = gt / wall
    return pt, gt, wall, per_avg, agg

COMBOS = [(16384, 1), (16384, 2), (16384, 3), (16384, 4),
        (32728, 1), (32728, 2), (32728, 3), (32728, 4),
        (98728, 1), (98728, 2), (98728, 3), (98728, 4)]

print("%-6s %-3s %8s %8s %8s %8s %8s %8s" % ("len", "con", "prompt", "gen", "wall_s", "e2e_avg", "req_tok/s", "agg_tok/s"))
for L, C in COMBOS:
    try:
        pt, gt, wall, per_avg, agg = run(L, C)
        print("%-6d %-3d %8d %8d %8.2f %8.2f %8.1f %8.1f" % (L, C, pt, gt, wall, per_avg, gt/C/wall, agg), flush=True)
    except Exception as e:
        print("%-6d %-3d FAILED %s" % (L, C, e), flush=True)
