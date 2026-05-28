#!/usr/bin/env python3
"""GLM-5 collect bench vs profiler ground truth 对齐脚本。

用法:
    python3 tools/check_alignment.py
    python3 tools/check_alignment.py --groundtruth docs/profiler_alignment/groundtruth/by_op_family.csv
    python3 tools/check_alignment.py --bench-gemm systems/data/ascend_910b/vllm-ascend/0.18.0/gemm_perf.txt

输出:
    每个有对应 bench 数据的 (op_type, shape) 一行，列出 profiler 实测 vs
    bench 实测的偏差。整体偏差分布给出 avg / median / |max|。

设计原则:
    - groundtruth 是"标准答案"（生产 profiler 跨多个 run 的中位数）
    - 修改 collector / 重新采集后跑这个脚本，秒看精度变化
    - 不依赖任何运行时框架，纯 csv 解析
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def load_groundtruth(path: Path) -> dict:
    """Return {(family, npu_op_type, shape): {'median_us': x, 'calls': n}}."""
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (r['family'], r['npu_op_type'], r['input_shapes'])
            out[key] = {
                'median_us': float(r['median_us']),
                'avg_us': float(r['avg_us']),
                'calls': int(r['total_calls']),
            }
    return out


def load_bench_gemm(path: Path) -> dict:
    """Return {(M, N, K, dtype_label): mean_latency_us}."""
    out = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (int(r['m']), int(r['n']), int(r['k']), r['gemm_dtype'])
            out[key].append(float(r['latency']) * 1000.0)   # ms -> us
    return {k: mean(v) for k, v in out.items()}


def parse_gemm_shape(shape_str: str) -> tuple | None:
    """parse profiler 'Input Shapes' field for MatMul / QuantBatchMatmul.

    Returns (M, N, K) or None if shape isn't a 2D MatMul.
    Handles both ND (2-dim) and FRACTAL_NZ (4-dim) weight layouts.
    """
    parts = shape_str.strip().strip('"').split(';')
    if len(parts) < 2:
        return None
    try:
        a = [int(x) for x in parts[0].split(',')]
        b = [int(x) for x in parts[1].split(',')]
    except ValueError:
        return None
    if len(a) != 2:
        return None
    m, k_in = a
    if len(b) == 4 and b[2] == 16 and b[3] == 32:
        # FRACTAL_NZ: (N/32, K/16, 16, 32) -> logical (N=a*d, K=b*c)
        if b[1] * b[2] == k_in:
            return m, b[0] * b[3], k_in
    elif len(b) == 2:
        if b[0] == k_in:
            return m, b[1], k_in
        if b[1] == k_in:
            return m, b[0], k_in
    return None


def align_gemm(groundtruth: dict, bench: dict) -> list:
    """Compare profiler GEMM ops vs bench measurements."""
    rows = []
    for (family, op_type, shape), gt in groundtruth.items():
        if family != 'GEMM':
            continue
        parsed = parse_gemm_shape(shape)
        if parsed is None:
            continue
        m, n, k = parsed
        dtype = 'float16' if op_type == 'MatMulV2' else 'w8a8_dynamic'
        bench_us = bench.get((m, n, k, dtype))
        if bench_us is None:
            rows.append({
                'op': op_type, 'm': m, 'n': n, 'k': k, 'dtype': dtype,
                'profiler_us': gt['median_us'], 'bench_us': None,
                'diff_pct': None, 'calls': gt['calls'],
                'status': 'BENCH_MISS',
            })
            continue
        diff_pct = (bench_us - gt['median_us']) / gt['median_us'] * 100
        rows.append({
            'op': op_type, 'm': m, 'n': n, 'k': k, 'dtype': dtype,
            'profiler_us': gt['median_us'], 'bench_us': bench_us,
            'diff_pct': diff_pct, 'calls': gt['calls'],
            'status': ('OK' if abs(diff_pct) < 20 else
                       'WARN' if abs(diff_pct) < 50 else 'FAIL'),
        })
    rows.sort(key=lambda r: -r['calls'])
    return rows


def print_report(rows: list) -> None:
    print(f"\n{'op':<22} {'M':>5} {'N':>6} {'K':>6} {'dtype':<14} "
          f"{'calls':>6} {'profiler(us)':>12} {'bench(us)':>10} {'diff':>8} {'status':>6}")
    print('-' * 110)
    matched = []
    miss_count = 0
    for r in rows:
        if r['status'] == 'BENCH_MISS':
            miss_count += 1
            print(f"  {r['op']:<20} {r['m']:>5} {r['n']:>6} {r['k']:>6} "
                  f"{r['dtype']:<14} {r['calls']:>6} {r['profiler_us']:>12.2f} "
                  f"{'(MISS)':>10} {'--':>8} {'MISS':>6}")
        else:
            matched.append(r)
            print(f"  {r['op']:<20} {r['m']:>5} {r['n']:>6} {r['k']:>6} "
                  f"{r['dtype']:<14} {r['calls']:>6} {r['profiler_us']:>12.2f} "
                  f"{r['bench_us']:>10.2f} {r['diff_pct']:>+7.1f}% {r['status']:>6}")

    print('-' * 110)
    if matched:
        diffs = [r['diff_pct'] for r in matched]
        abs_d = [abs(d) for d in diffs]
        print(f"\n  matched: {len(matched)}, bench MISS: {miss_count}")
        print(f"  avg diff   : {mean(diffs):+.1f}%")
        print(f"  median diff: {sorted(diffs)[len(diffs)//2]:+.1f}%")
        print(f"  avg |diff| : {mean(abs_d):.1f}%")
        print(f"  max |diff| : {max(abs_d):.1f}%")
        for lo, hi in [(0, 10), (10, 20), (20, 50), (50, 100), (100, 1e9)]:
            n = sum(1 for d in abs_d if lo <= d < hi)
            label = f"|diff| ∈ [{lo}, {'∞' if hi > 1e8 else hi})%"
            print(f"  {label:<20s}: {n}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--groundtruth', type=Path,
        default=repo / 'docs/profiler_alignment/groundtruth/by_op_family.csv',
    )
    parser.add_argument(
        '--bench-gemm', type=Path,
        default=repo / 'systems/data/ascend_910b/vllm-ascend/0.18.0/gemm_perf.txt',
    )
    args = parser.parse_args()

    if not args.groundtruth.exists():
        raise SystemExit(f'groundtruth missing: {args.groundtruth}')
    if not args.bench_gemm.exists():
        raise SystemExit(f'bench gemm missing: {args.bench_gemm}')

    gt = load_groundtruth(args.groundtruth)
    bench = load_bench_gemm(args.bench_gemm)
    print(f'groundtruth: {len(gt)} (family, op_type, shape) entries')
    print(f'bench gemm:  {len(bench)} (M, N, K, dtype) keys')

    rows = align_gemm(gt, bench)
    print_report(rows)


if __name__ == '__main__':
    main()
