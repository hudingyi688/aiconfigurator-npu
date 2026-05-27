#!/usr/bin/env python3
"""Minimal reproducer for npu_quant_matmul / npu_dynamic_quant.

Goal: figure out which combination of (M, N, K, weight_layout, scale_dtype)
this CANN install actually accepts, before continuing with the full
W8A8 GEMM sweep. Run on the NPU host.

Usage:
    PYTHONPATH=collector python3 tools/probe_quant_matmul.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch

os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", "")
torch.npu.config.allow_internal_format = True
torch.npu.set_device(0)
import torch_npu  # noqa: E402

DEV = "npu:0"
BF16 = torch.bfloat16
I8 = torch.int8


def try_call(label, run):
    try:
        out = run()
        torch.npu.synchronize()
        if isinstance(out, torch.Tensor):
            print(f"  [OK]   {label:60s} -> {tuple(out.shape)} {out.dtype}")
        else:
            print(f"  [OK]   {label:60s} -> {out!r}")
        return True
    except Exception as e:
        msg = repr(e).split("\n", 1)[0][:200]
        print(f"  [FAIL] {label:60s} -> {type(e).__name__}: {msg}")
        return False


print("=" * 70)
print("[1] sanity: trivial bf16 add")
print("=" * 70)
try_call("torch.zeros + 1", lambda: (torch.zeros(8, dtype=BF16, device=DEV) + 1).sum())


print()
print("=" * 70)
print("[2] npu_dynamic_quant on small bf16")
print("=" * 70)
for shape in [(1, 6144), (16, 6144), (1024, 6144)]:
    x = torch.randn(*shape, dtype=BF16, device=DEV)

    def go(x=x):
        q, s = torch_npu.npu_dynamic_quant(x)
        return q

    try_call(f"dynamic_quant x={shape}", go)


print()
print("=" * 70)
print("[3] npu_quant_matmul: smallest possible (M=1, N=K=128)")
print("=" * 70)
M, N, K = 1, 128, 128
x_q = torch.randint(-128, 127, (M, K), dtype=I8, device=DEV)
pertoken_scale = torch.rand(M, dtype=BF16, device=DEV) * 0.1 + 0.01
weight_KN = torch.randint(-128, 127, (K, N), dtype=I8, device=DEV)
weight_NK = torch.randint(-128, 127, (N, K), dtype=I8, device=DEV)
ws_1d_bf16 = torch.rand(N, dtype=BF16, device=DEV) * 0.1 + 0.01
ws_1d_fp32 = ws_1d_bf16.to(torch.float32)

# Layout / scale-dtype matrix
for w_layout, w in [("(K,N)", weight_KN), ("(N,K)", weight_NK)]:
    for sd_label, scale in [("bf16", ws_1d_bf16), ("fp32", ws_1d_fp32)]:
        try_call(
            f"qmm M=1 N=128 K=128 weight={w_layout} scale={sd_label}",
            lambda w=w, scale=scale: torch_npu.npu_quant_matmul(
                x_q, w, scale,
                pertoken_scale=pertoken_scale,
                bias=None,
                output_dtype=BF16,
            ),
        )


print()
print("=" * 70)
print("[4] npu_quant_matmul on the GLM-5 logits shape that crashed")
print("=" * 70)
for M, N, K in [(1, 24576, 6144), (1, 77440, 6144), (1, 154880, 6144)]:
    x_q = torch.randint(-128, 127, (M, K), dtype=I8, device=DEV)
    ps = torch.rand(M, dtype=BF16, device=DEV) * 0.1 + 0.01
    w_KN = torch.randint(-128, 127, (K, N), dtype=I8, device=DEV)
    w_NK = torch.randint(-128, 127, (N, K), dtype=I8, device=DEV)
    ws_bf16 = torch.rand(N, dtype=BF16, device=DEV) * 0.1 + 0.01
    ws_fp32 = ws_bf16.to(torch.float32)

    for w_layout, w in [("(K,N)", w_KN), ("(N,K)", w_NK)]:
        for sd_label, scale in [("bf16", ws_bf16), ("fp32", ws_fp32)]:
            try_call(
                f"qmm M={M} N={N} K={K} w={w_layout} scale={sd_label}",
                lambda w=w, scale=scale, ps=ps: torch_npu.npu_quant_matmul(
                    x_q, w, scale,
                    pertoken_scale=ps,
                    bias=None,
                    output_dtype=BF16,
                ),
            )


print()
print("=" * 70)
print("[5] try going through vllm-ascend's apply() at smallest shape")
print("=" * 70)
try:
    import vllm_ascend  # noqa
    from vllm_ascend.platform import NPUPlatform
    NPUPlatform.import_kernels()
    from vllm_ascend.quantization.methods.w8a8_dynamic import (
        AscendW8A8DynamicLinearMethod,
    )
    qm = AscendW8A8DynamicLinearMethod()

    class _Layer(torch.nn.Module):
        pass
    layer = _Layer()
    layer.weight = torch.nn.Parameter(
        torch.randint(-128, 127, (128, 128), dtype=I8, device=DEV),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.rand(128, 1, dtype=BF16, device=DEV) * 0.1 + 0.01,
        requires_grad=False,
    )
    layer.weight_offset = torch.nn.Parameter(
        torch.zeros(128, 1, dtype=BF16, device=DEV),
        requires_grad=False,
    )
    print("  layer constructed")

    try:
        qm.process_weights_after_loading(layer)
        print("  process_weights_after_loading OK")
    except Exception as e:
        print(f"  process_weights_after_loading FAIL: {repr(e)[:150]}")

    x = torch.randn(1, 128, dtype=BF16, device=DEV)
    try:
        out = qm.apply(layer, x, bias=None, tp_rank=0)
        torch.npu.synchronize()
        print(f"  apply() OK  out={tuple(out.shape)}")
    except Exception as e:
        traceback.print_exc()
        print(f"  apply() FAIL: {repr(e)[:200]}")
except Exception as e:
    print(f"  vllm-ascend setup failed: {e}")


print()
print("Done. The first row in section [3] / [4] that prints OK tells us")
print("which weight layout + scale dtype works. We'll lock the collector")
print("on that combination next.")
