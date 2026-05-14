#!/usr/bin/env bash
# Identify which 26 W8A8 MoE specs failed to compile by diffing BF16
# vs W8A8 checkpoint entries. The errors are all the same root cause:
#   "ub overflow, requires 2621440 bits while 1572864 bits available"
#   "Failed to run BiShengHIR pipeline"
# i.e. Triton/BiSheng Unified Buffer exhaustion on certain
# (hidden, intermediate, num_experts) combinations under W8A8 tiling.
# We need to know which model configs are affected so they can be
# documented as "no W8A8 data on this SoC".
#
# Usage on the NPU host:
#     tools/diff_moe_w8a8_failures.sh
#     tools/diff_moe_w8a8_failures.sh /custom/data/root
set -u

cd "$(dirname "$0")/.." || exit 1
ROOT="${1:-data}"

python3 - "${ROOT}" <<'PY'
import json, os, sys, collections
root = sys.argv[1]
ckpt = os.path.join(root, "moe", "moe_checkpoint.json")
if not os.path.exists(ckpt):
    print(f"(missing) {ckpt}")
    sys.exit(1)
with open(ckpt) as f:
    completed = list(json.load(f).get("completed", []))

def parse(key):
    # key format: tokens_hidden_inter_experts_topk_local_quant_modelname...
    # quant_type may be "bf16" or "w8a8_dynamic", model name follows.
    parts = key.split("_")
    if "bf16" in parts:
        i = parts.index("bf16")
        quant = "bf16"
    elif "w8a8" in parts and "dynamic" in parts:
        i = parts.index("w8a8")
        quant = "w8a8_dynamic"
    else:
        return None
    head = parts[:i]
    tail = parts[i+(2 if quant == "w8a8_dynamic" else 1):]
    if len(head) < 6:
        return None
    return {
        "tokens":      int(head[0]),
        "hidden":      int(head[1]),
        "intermediate": int(head[2]),
        "experts":     int(head[3]),
        "topk":        int(head[4]),
        "local":       int(head[5]),
        "quant":       quant,
        "model":       "_".join(tail),
        "raw":         key,
    }

parsed = [p for p in (parse(k) for k in completed) if p is not None]
bf16 = {(p["tokens"], p["hidden"], p["intermediate"], p["experts"],
         p["topk"], p["local"], p["model"]) for p in parsed if p["quant"] == "bf16"}
w8a8 = {(p["tokens"], p["hidden"], p["intermediate"], p["experts"],
         p["topk"], p["local"], p["model"]) for p in parsed if p["quant"] == "w8a8_dynamic"}

missing = sorted(bf16 - w8a8)
print(f"BF16 specs       : {len(bf16)}")
print(f"W8A8 specs       : {len(w8a8)}")
print(f"BF16 with no W8A8: {len(missing)}    <-- the failing 26")
print()

# Group missing by model to see which models are entirely affected.
by_model = collections.defaultdict(list)
for spec in missing:
    by_model[spec[6]].append(spec)
print("=== failures grouped by model ===")
for model, specs in sorted(by_model.items()):
    n = len(specs)
    n_bf16 = sum(1 for x in bf16 if x[6] == model)
    print(f"{model:24s}  {n:3d} of {n_bf16:3d} W8A8 specs failed")
    # print unique (hidden, inter, experts, topk) quadruples so we can
    # see whether it's all token sizes or only some.
    shapes = sorted({(s[1], s[2], s[3], s[4]) for s in specs})
    for h, i, e, t in shapes:
        toks = sorted(s[0] for s in specs if (s[1], s[2], s[3], s[4]) == (h, i, e, t))
        print(f"    hidden={h} inter={i} experts={e} topk={t}  failed tokens={toks}")
PY
