# -*- coding: utf-8 -*-
"""
KV-cache quality A/B: long-context needle retrieval.

Same fixed-seed 100K-token haystack with needles at 20% / 35% / 50% / 65% /
80% depth. Run once per KV-dtype config (fp8 vs bf16) and diff the JSON.
Greedy decoding so runs are comparable; scoring is exact-match on hidden
facts and verbatim serial codes.

Usage:
  python kv_quality_ab.py                          # 100K context, default
  python kv_quality_ab.py --length 60000           # shorter haystack
  python kv_quality_ab.py --out fp8.json           # save results
Then diff two saved JSON files, or just read the score lines.
"""
import argparse
import json
import random
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import requests  # noqa: E402

FILLER = [
    "The depot recorded seventeen inbound crates before noon, most of them tagged for the northern branch.",
    "Rain shifted the harvest schedule by two days, and the silo supervisor rewrote the delivery manifest.",
    "An audit of the cooling loop found mineral deposits near the third valve, but pressure stayed nominal.",
    "The night crew swapped the conveyor belt without interrupting the sorting queue.",
    "Insurance paperwork for the warehouse annex arrived late, listing three unsigned amendments.",
    "Field notes mention a faint signal drifting across the eastern ridge at odd hours.",
    "The archive room catalogued forty boxes of receipts dating back to the previous decade.",
    "A freight delay in the coastal corridor pushed two inspections into the following week.",
    "Maintenance logs show the backup generator ran a twelve minute test cycle on a Tuesday.",
    "The survey team remeasured the fence line and found a discrepancy of exactly eleven meters.",
    "Inventory flagged a surplus of pallet jacks and a shortage of shrink wrap.",
    "The cafeteria moved lunch service earlier to match the new shift rotation.",
    "Storm damage to the loading dock roof was patched with borrowed tarpaulin and good intentions.",
    "A subcontractor invoice double-charged the pallet rental line, which accounting later reversed.",
    "The old pump house hums audibly whenever the reservoir drops below the midline marker.",
    "Training exercises occupied the north yard for three consecutive mornings.",
    "The quality board approved a minor tolerance change for the brushed finish samples.",
    "A driver reported a wandering axle on route nine; the truck was parked for inspection.",
    "The ledger shows a small, recurring refund under a vendor name no one recognizes.",
    "Summer heat warped the plastic bins stacked against the west wall.",
    "The communications test reached the hill relay on the second attempt.",
    "Crew rotation charts changed color coding, causing a week of minor confusion.",
    "A fox was seen near the generator shed on four separate nights.",
    "The spare parts locker was reorganized alphabetically, then abandoned halfway through.",
    "Two inspectors disagreed about the grading of the timber shipment and escalated it politely.",
    "The depot's water pipe froze again at the same elbow joint as last winter.",
    "A calendar mixup sent the safety refresher course two weeks earlier than planned.",
    "Fog closed the access road until mid-morning, delaying the milk run.",
    "The new label printer uses ribbons nobody else in the building orders.",
    "An anonymous note in the suggestion box requested better staplers, signed 'the third shift'.",
    "Weighbridge calibration certificates were renewed with a one kilogram correction noted.",
    "The old loading scale still drifts by a few grams when it rains.",
    "A bird nested in the rafters above bay four, and management decided to wait it out.",
    "Quarterly forecasts were revised twice, mostly due to rounding disagreements.",
    "The gate camera reboots itself shortly after every thunderstorm.",
    "Night shift prefers the quiet radio channel, which newer staff keep finding by accident.",
    "A pallet of returnables went out with the wrong colored tag and came back a week later.",
    "The break room whiteboard still lists a committee that finished its work last spring.",
]

SYSTEM = (
    "你是一个仔细阅读档案的助手。只根据给定材料回答问题,答案尽量简短,不要解释。"
    "如果材料中没有答案,回答“未找到”。"
)


def build_needles(rng):
    """Fixed-format hidden facts + verbatim serial codes."""
    code = lambda n: f"{chr(65+rng.randrange(26))}{chr(65+rng.randrange(26))}{rng.randrange(10)}-{rng.randrange(1000,9999)}-{chr(65+rng.randrange(26))}{chr(65+rng.randrange(26))}"
    n1 = f"Kestrel-{rng.randrange(2,9)} 站点的备用电源熔断编号是 {code(rng)}。"
    n2 = f"档案编号 {rng.randrange(70000,79999)} 的运单在 {rng.randrange(3,27)} 日由车队「{['晨星','远山','青梧'][rng.randrange(3)]}」签收。"
    n3 = f"第 {rng.randrange(4,58)} 号仓库的湿度报警阈值被临时调整为 {rng.randrange(41,66)}%RH。"
    s1 = f"随附校验码:{code(rng)}(请勿在公开场合复述)。"
    s2 = f"本批次追溯序列号:{code(rng)}。"
    return [
        (n1, "Kestrel 站点的备用电源熔断编号是什么?只输出编号本身。", "fact-1"),
        (s1, "档案中提到的随附校验码是什么?只输出校验码本身。", "serial-1"),
        (n2, "档案编号 7xxxx 段的运单是在几日、由哪个车队签收的?", "fact-2"),
        (s2, "本批次的追溯序列号是什么?只输出序列号本身。", "serial-2"),
        (n3, "仓库的湿度报警阈值被临时调整为多少?只输出数字和百分号。", "fact-3"),
    ]


def needle_answer(needle):
    """Ground-truth fragments that must appear in the answer."""
    if "熔断编号" in needle:
        return [re.search(r"[A-Z]{2}\d-\d{4}-[A-Z]{2}", needle).group(0)]
    if "校验码" in needle or "序列号" in needle:
        return [re.search(r"[A-Z]{2}\d-\d{4}-[A-Z]{2}", needle).group(0)]
    if "签收" in needle:
        m = re.search(r"(\d{1,2}) 日由车队「(\S+?)」", needle)
        return [m.group(1), m.group(2)]
    if "湿度" in needle:
        m = re.search(r"(\d{2})%RH", needle)
        return [m.group(1)]
    return [needle]


def build_context(target_tokens, chars_per_token, rng):
    paras = []
    chars = 0
    budget = int(target_tokens * chars_per_token)
    while chars < budget:
        block = " ".join(rng.sample(FILLER, k=len(FILLER)))
        paras.append(block)
        chars += len(block) + 2
    return paras


def normalize(text):
    return re.sub(r"[\s,.,。,:：;；\-_]+", "", text).upper()


def build_hard(seed):
    """Hard profile: dense same-shape rows, near-miss decoys, multi-hop.
    Returns (rows, tasks) where tasks = (needle_rows[], question, expected[], key)."""
    rng = random.Random(seed)
    fleets = ["晨星", "远山", "青梧", "长夏", "拾光"]
    code = lambda: f"{chr(65+rng.randrange(26))}{chr(65+rng.randrange(26))}{rng.randrange(10)}-{rng.randrange(1000,9999)}-{chr(65+rng.randrange(26))}{chr(65+rng.randrange(26))}"
    wb = lambda: f"WB-{rng.randrange(1000,7999)}-{rng.randrange(100,999)}"

    def row(w=None, fleet=None, code_v=None, day=None, tag=""):
        return (f"运单 {w or wb()}:站点 S{rng.randrange(10,89)},"
                f"车队「{fleet or rng.choice(fleets)}」,签收 {day or rng.randrange(1,29)} 日,"
                f"校验码 {code_v or code()}{tag}")

    # T1: exact code among single-char decoys
    t1_wb = f"WB-{rng.randrange(8100,8199)}-{rng.randrange(900,999)}"
    t1_code = f"{chr(65+rng.randrange(26))}{chr(65+rng.randrange(26))}{rng.randrange(10)}-{rng.randrange(1000,8999)}-{chr(65+rng.randrange(26))}{'K'}"
    def mutate(s, i):
        chars = list(s)
        c = chars[i]
        nxt = chr((ord(c) - 48) % 10 + 48) if c.isdigit() else chr((ord(c) - 65 + 1) % 26 + 65)
        chars[i] = nxt
        return "".join(chars)
    t1_rows = [(row(t1_wb, code_v=t1_code), 0.20)]
    for i, d in [(1, 0.17), (4, 0.185), (6, 0.215), (10, 0.23)]:
        t1_rows.append((row(f"WB-{rng.randrange(8100,8199)}-{rng.randrange(100,899):03d}",
                            code_v=mutate(t1_code, i)), d))
    t1 = (t1_rows, f"运单 {t1_wb} 的校验码是什么?只输出校验码本身。", [t1_code], "exact-vs-decoys")

    # T2: multi-hop chain waybill -> station -> gate -> rule
    t2_wb, t2_st, t2_gate, t2_days = "WB-7731-101", f"S{rng.randrange(40,60)}", f"K-{rng.randrange(900,960)}", rng.randrange(2,6)
    t2_rows = [
        (row(t2_wb, fleet=rng.choice(fleets)) + f",卸货站点 {t2_st}", 0.12),
        (f"站点档案:{t2_st} 的闸口号为 {t2_gate}。", 0.58),
        (f"调度条例:凡闸口以 K-9 开头者,转运时限为 {t2_days} 日。", 0.62),
        ("调度条例:凡闸口以 K-8 开头者,转运时限为 9 日。", 0.64),
    ]
    t2 = (t2_rows, f"运单 {t2_wb} 的转运时限是几日?只输出数字。", [str(t2_days)], "multi-hop")

    # T4: verbatim 16-hex among near-miss hex decoys
    t4_wb = f"WB-{rng.randrange(8200,8299)}-{rng.randrange(400,499)}"
    t4_hex = f"{rng.getrandbits(64):016X}"
    def hexmut(s, n):
        pos = rng.sample(range(16), n)
        return "".join(c if i not in pos else f"{(int(c,16)+3)%16:X}" for i, c in enumerate(s))
    t4_rows = [(row(t4_wb) + f",溯源码 {t4_hex}", 0.50)]
    for i, d in [(46, 0.47), (48, 0.485), (52, 0.515), (54, 0.53)]:
        t4_rows.append((row(f"WB-{rng.randrange(8200,8299)}-{rng.randrange(100,899):03d}")
                        + f",溯源码 {hexmut(t4_hex, 3)}", d/100))
    t4 = (t4_rows, f"运单 {t4_wb} 的溯源码是什么?逐字符原样输出。", [t4_hex], "verbatim-16hex")

    # T3: count rare tagged subset (mark exactly N rows with 急件 tag)
    t3_n = 17
    t3_rows = [(row(f"WB-{9000+i}", fleet="远山", tag=",急件"), None) for i in range(t3_n)]
    t3 = (t3_rows, "车队「远山」名下标注为急件的运单共有多少单?只输出数字。", [str(t3_n)], "count")

    # T5: temporal order across far depths
    t5_labels = ["闸门检修开始", "雾警发布", "发电机试机"]
    t5_times = [f"03:{rng.randrange(30,58)}", f"01:{rng.randrange(10,58)}", f"05:{rng.randrange(1,20)}"]
    order = sorted(range(3), key=lambda i: t5_times[i])
    t5_rows = [(f"值班记录 {t5_times[i]}:{t5_labels[i]}。", d)
               for i, d in zip(range(3), [0.30, 0.70, 0.94])]
    t5 = (t5_rows, "值班记录中哪一件事发生得最早?只输出事件名称。",
          [t5_labels[order[0]]], "temporal-order")

    tasks = [t1, t2, t3, t4, t5]
    specials = {}
    for task in tasks:
        for r, d in task[0]:
            if d is None:
                specials.setdefault("count", []).append(r)
            else:
                specials.setdefault(round(d, 3), []).append(r)
    return None, specials, tasks


def build_context_hard(target_tokens, chars_per_token, rng, _filler, specials):
    """Unique random rows (no reuse), ~8 rows per paragraph block."""
    paras, chars = [], 0
    budget = int(target_tokens * chars_per_token)
    while chars < budget:
        blk = "\n".join(
            f"运单 WB-{rng.randrange(1000,7999)}-{rng.randrange(100,999)}:站点 S{rng.randrange(10,89)},"
            f"车队「{rng.choice(['晨星','远山','青梧','长夏','拾光'])}」,签收 {rng.randrange(1,29)} 日,"
            f"校验码 {chr(65+rng.randrange(26))}{chr(65+rng.randrange(26))}{rng.randrange(10)}-{rng.randrange(1000,9999)}-{chr(65+rng.randrange(26))}{chr(65+rng.randrange(26))}"
            for _ in range(8))
        paras.append(blk)
        chars += len(blk) + 2
    for d, rows in sorted((kv for kv in specials.items() if kv[0] != "count")):
        idx = max(1, int(len(paras) * d))
        for j, r in enumerate(rows):
            paras.insert(idx + j, "* " + r)
    for j, r in enumerate(specials.get("count", [])):
        paras.insert(2 + j * max(3, len(paras) // 20), "* " + r)
    return "\n\n".join(paras)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["easy", "hard"], default="easy",
                    help="hard: dense same-shape rows, decoys, multi-hop, count, ordering")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--model", default=None, help="default: first served model")
    ap.add_argument("--length", type=int, default=100000, help="haystack tokens")
    ap.add_argument("--chars-per-token", type=float, default=6.17,
                    help="English filler calibration measured by llm-inference-bench")
    ap.add_argument("--seed", type=int, default=20260830, help="fixed => configs comparable")
    ap.add_argument("--depths", default="0.20,0.35,0.50,0.65,0.80")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="reasoning model: budget must cover CoT + short answer")
    ap.add_argument("--temperature", type=float, default=0.0, help="greedy for comparability")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    sess = requests.Session()

    info = sess.get(base + "/get_server_info", timeout=10).json()
    ctx_cap = info.get("context_len") or info.get("max_total_num_tokens")
    model = args.model or (info.get("model_path") or "?")
    served = sess.get(base + "/v1/models", timeout=10).json()["data"][0]["id"]
    print(f"[server] model={model} served={served} context_len={ctx_cap}")
    if args.length + 512 > (ctx_cap or 0):
        sys.exit(f"[!] length {args.length} exceeds server context {ctx_cap}")

    rng = random.Random(args.seed)
    if args.profile == "hard":
        filler, specials, tasks = build_hard(args.seed)
        needles = [("", q, k) for (_rows, q, _e, k) in tasks]
        expected_map = {k: e for (_rows, _q, e, k) in tasks}
        depths = [0.5] * len(tasks)
        context = build_context_hard(args.length, args.chars_per_token, rng, filler, specials)
    else:
        needles = build_needles(rng)
        tasks = None
        depths = [float(x) for x in args.depths.split(",")]
        assert len(depths) == len(needles), "need one depth per needle"
        paras = build_context(args.length, args.chars_per_token, rng)
        for (needle, _q, _k), d in zip(needles, depths):
            idx = max(1, int(len(paras) * d))
            paras.insert(idx, "* " + needle)
        context = "\n\n".join(paras)
    est_tokens = int(len(context) / args.chars_per_token)
    print(f"[haystack] ~{est_tokens:,} tokens, {len(context):,} chars, "
          f"{len(needles)} needles at {depths}")

    results, score = [], 0
    for n, (needle, q, key) in enumerate(needles):
        payload = {
            "model": served,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": context + "\n\n---\n问题:" + q},
            ],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        t0 = time.time()
        r = sess.post(base + "/v1/chat/completions", json=payload, timeout=args.timeout)
        r.raise_for_status()
        j = r.json()
        m = j["choices"][0]["message"]
        content = (m.get("content") or "").strip()
        reasoning = (m.get("reasoning_content") or "").strip()
        # Reasoning model: the answer may live in content, or the CoT quotes
        # the retrieved needle if content ran out. Score against both.
        merged = content + "\n" + reasoning
        ans = content if content else (reasoning[-160:] + "~")
        usage = j.get("usage", {})
        truth = expected_map[key] if tasks else needle_answer(needle)
        norm = normalize(merged)
        ok = all(normalize(x) in norm for x in truth)
        score += ok
        results.append({
            "key": key, "depth": depths[n], "question": q, "expected": truth,
            "answer": ans, "ok": ok, "elapsed_s": round(time.time() - t0, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        })
        print(f"  [{'OK ' if ok else 'MISS'}] {key:16s} "
              f"expect={truth} ans={ans[:60]!r} ({time.time()-t0:.1f}s)")

    print(f"[score] {score}/{len(needles)}  tag={args.tag}")
    out = args.out or f"kv_quality_{args.tag}.json"
    json.dump({"tag": args.tag, "seed": args.seed, "length": args.length,
               "temperature": args.temperature, "score": f"{score}/{len(needles)}",
               "results": results},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
