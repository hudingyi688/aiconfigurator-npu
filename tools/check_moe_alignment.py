#!/usr/bin/env python3
"""GLM-5 MoE DispatchFFNCombine: profiler ground truth vs bench 对齐脚本。

与 check_alignment.py(GEMM) 互补，覆盖 FusedMC2 融合算子(占 profiler ~20% 时间)。

为什么不能直接套 GEMM 脚本的口径(关键):
    by_op_family.csv 把不同 (phase, ep) 的同 num_tokens 行聚合求 median，
    会把 decode(小 M、ep8/10) 和 prefill(大计算量、ep0/16) 混成一个污染中位数。
    本脚本改用 detail.csv，按 (phase, ep, num_tokens) 严格分组后再比，
    且 decode / prefill 分开报告 —— 因为同样 num_tokens=256，prefill(每 token
    attend 累积 KV) 和 decode 是不同物理量，混比会制造假"低估"。

ep 维度局限(诚实标注，已实测推翻"ep-insensitive"假设):
    bench 表 ep∈{2,4,8,32}，profiler 实测 ep∈{0,8,10,16} —— 几乎不重叠。
    且 dispatch 延迟【强 ep 依赖】(实测 ntok=8: ep2=1711us / ep8=867 / ep32=301，
    差 5.7×)，所以【不能】跨 ep 夹取近似比对 —— 那会制造 +100%~+190% 的假偏差。
    本脚本只在【严格同 ep】的点上计入匹配度统计；其余点仅列出、标 ep≠ 不计入。
    现实: 严格同 ep 的可比点极少(仅 decode ep8 的 ntok=1/2)。结论 ——
    MoE dispatch 的可信度主要来自【采集方式】(ep32 实采 microbench 直接替换 clamp，
    见 ep32 的 10.8x 修正)，而非 profiler-vs-bench 匹配度(几乎无干净可比点)。

用法:
    python3 tools/check_moe_alignment.py
    python3 tools/check_moe_alignment.py --detail <path> --bench-moe <path>

设计原则(同 check_alignment.py):
    - profiler 是标准答案(跨 run 中位数)
    - 不依赖运行时框架，纯 csv 解析
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

OP = "DispatchFFNCombine"


def parse_ntok(shape: str) -> int | None:
    """profiler input_shapes 第一字段是 'num_tokens,6144'。"""
    try:
        return int(shape.split(";")[0].split(",")[0])
    except (ValueError, IndexError):
        return None


def load_profiler(path: Path) -> list[dict]:
    """detail.csv -> 每条 DispatchFFNCombine 记录(带 phase/ep/num_tokens)。"""
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["op_type"] != OP:
                continue
            ntok = parse_ntok(r["input_shapes"])
            if ntok is None:
                continue
            out.append({
                "phase": r["phase"],
                "ep": int(r["ep"]),
                "ntok": ntok,
                "count": int(r["count"]),
                "median_us": float(r["median_us"]),
            })
    return out


def load_bench(path: Path) -> dict:
    """moe_dispatch_combine_perf.txt -> {(ep, dtype, ntok): latency_us}。"""
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (int(r["ep_size"]), r["dtype"], int(r["num_tokens"]))
            out[key] = float(r["latency_us"])
    return out


def bench_lookup(bench: dict, ep: int, dtype: str, ntok: int) -> float:
    """查 bench 在【严格同 ep】下的 latency；ntok 缺失时线性插值/clamp。

    不做跨 ep 夹取 —— dispatch 强 ep 依赖，跨 ep 比对无意义。
    该 ep 在 bench 中不存在时返回 nan。
    """
    pts = sorted((nt, lat) for (e, dt, nt), lat in bench.items()
                 if e == ep and dt == dtype)
    if not pts:
        return float("nan")
    xs = [p[0] for p in pts]
    if ntok <= xs[0]:
        return pts[0][1]
    if ntok >= xs[-1]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= ntok <= x1:
            t = (ntok - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detail", type=Path,
                    default=repo / "docs/profiler_alignment/groundtruth/detail.csv")
    ap.add_argument("--bench-moe", type=Path,
                    default=repo / "src/aiconfigurator_npu/systems/data/"
                    "ascend_910b/vllm-ascend/0.18.0/moe_dispatch_combine_perf.txt")
    ap.add_argument("--dtype", default="w8a8_dynamic",
                    help="生产量化 dtype (默认 w8a8_dynamic)")
    args = ap.parse_args()

    if not args.detail.exists():
        raise SystemExit(f"detail.csv missing: {args.detail}")
    if not args.bench_moe.exists():
        raise SystemExit(f"bench moe missing: {args.bench_moe}")

    prof = load_profiler(args.detail)
    bench = load_bench(args.bench_moe)
    print(f"profiler DispatchFFNCombine 记录: {len(prof)}")
    print(f"bench keys: {len(bench)}  (ep={sorted({k[0] for k in bench})}, "
          f"dtype={sorted({k[1] for k in bench})})")
    print(f"对齐用 dtype: {args.dtype}\n")

    bench_eps = sorted({k[0] for k in bench})
    # 按 phase 分组报告：profiler shape 不带 dtype，生产是 w8a8，统一用 --dtype 查 bench
    for phase in ("decode", "prefill"):
        rows = [r for r in prof if r["phase"] == phase]
        if not rows:
            continue
        print(f"{'='*92}\n=== {phase.upper()} ===")
        print(f"{'ep':>4}{'ntok':>6}{'count':>8}{'prof_us':>10}"
              f"{'bench_us':>10}{'diff':>9}  note")
        diffs = []  # 仅严格同 ep 的偏差计入统计
        for r in sorted(rows, key=lambda r: (r["ep"], r["ntok"])):
            same_ep = r["ep"] in bench_eps
            if same_ep:
                b_us = bench_lookup(bench, r["ep"], args.dtype, r["ntok"])
                diff = (b_us - r["median_us"]) / r["median_us"] * 100
                sev = "OK" if abs(diff) < 20 else ("WARN" if abs(diff) < 50 else "FAIL")
                diffs.append(diff)
                print(f"{r['ep']:>4}{r['ntok']:>6}{r['count']:>8}"
                      f"{r['median_us']:>10.1f}{b_us:>10.1f}{diff:>+8.1f}%  EP= {sev}")
            else:
                # ep 不在 bench 网格 → 跨 ep 不可比，仅列出 profiler 值
                print(f"{r['ep']:>4}{r['ntok']:>6}{r['count']:>8}"
                      f"{r['median_us']:>10.1f}{'--':>10}{'--':>9}  ep∉bench, 不可比")
        if diffs:
            a = [abs(d) for d in diffs]
            print(f"  [严格同 ep 匹配度] n={len(diffs)} "
                  f"avg{mean(diffs):+.1f}% |avg|{mean(a):.1f}% |max|{max(a):.1f}%")
        else:
            print("  [严格同 ep 匹配度] 无可比点 (该 phase 的 profiler ep 都不在 bench 网格)")

    print(f"\n{'='*92}")
    print("口径说明:")
    print("  - 匹配度只在【严格同 ep】点上算 (dispatch 强 ep 依赖，跨 ep 夹取会造假偏差)")
    print("  - decode: num_tokens=batch×(1+MTP) 的小 M，与 bench 标定场景一致")
    print("  - prefill: num_tokens 是 per-rank chunk token，每 token attend 累积 KV，")
    print("    与 decode 同 num_tokens 不同物理量；生产 prefill MoE 在单请求建模另算。")
    print("  - bench ep∈{2,4,8,32}，profiler ep∈{0,8,10,16}，干净可比点极少 →")
    print("    MoE dispatch 可信度主要靠【采集方式】(ep32 实采 microbench)，非匹配度。")


if __name__ == "__main__":
    main()
