"""E2E prefill TTFT validation for the profiler-derived DSA model.

Builds the production GLM-5 prefill config (disagg prefill worker) and runs the
context-phase static breakdown, isolating the DSA (context_attention) portion,
to compare against:
  - the OLD SLO doc (2026-05-30), which used the nh=4 synthetic silicon, and
  - the profiler ground truth (single-rank DSA attn-core 152.5ms@10k / 392.3@20k).
Not a pytest test (needs the full systems DB); run directly.
"""
import importlib.resources as ir

from aiconfigurator_npu.sdk import common, config
from aiconfigurator_npu.sdk.models import get_model
from aiconfigurator_npu.sdk.perf_database import get_database
from aiconfigurator_npu.sdk.inference_session import InferenceSession
from aiconfigurator_npu.sdk.backends.factory import get_backend

MODEL = "zai-org/GLM-5"
SYSTEM, BACKEND, VERSION = "ascend_910b", "vllm-ascend", "0.18.0"
ROOT = str(ir.files("aiconfigurator_npu") / "systems")


def build(tp, ep, dp):
    mc = config.ModelConfig(
        tp_size=tp, pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.w8a8_dynamic,
        moe_quant_mode=common.MoEQuantMode.w8a8_dynamic,
        kvcache_quant_mode=common.KVCacheQuantMode.float16,
        fmha_quant_mode=common.FMHAQuantMode.float16,
        moe_tp_size=1, moe_ep_size=ep, attention_dp_size=dp,
        is_disagg_prefill=True,
    )
    model = get_model(MODEL, mc, "vllm")
    db = get_database(SYSTEM, BACKEND, VERSION, ROOT)
    db.set_default_database_mode(common.DatabaseMode.HYBRID)
    return InferenceSession(model, db, get_backend("vllm")), db


def run_ctx(tp, ep, dp, isl):
    sess, _ = build(tp, ep, dp)
    rc = config.RuntimeConfig(batch_size=1, beam_width=1, isl=isl, osl=1)
    # context-only breakdown
    backend = sess._backend
    cdict, _, gdict, _ = backend._run_static_breakdown(
        sess._model, sess._database, rc, "static_ctx", 32, 1.0
    )
    dsa = cdict.get("context_attention", 0.0)
    total = sum(cdict.values())
    return dsa, total, cdict


if __name__ == "__main__":
    print(f"{'档':>3} {'isl':>6} {'cfg':>12} {'DSA(ms)':>9} {'prefill总(ms)':>12} {'DSA占比':>7}")
    cases = [
        # tp * dp == moe_tp * moe_ep
        ("档1", 10000, 16, 32, 2),   # 16*2 == 1*32
        ("档2", 20480, 16, 32, 2),
        ("档3", 40960, 32, 32, 1),   # 32*1 == 1*32
        ("档4", 81920, 32, 32, 1),
    ]
    for tag, isl, tp, ep, dp in cases:
        dsa, total, cdict = run_ctx(tp, ep, dp, isl)
        print(f"{tag:>3} {isl:>6} tp{tp}/ep{ep}/dp{dp} {dsa:>9.1f} {total:>12.1f} {dsa/total*100:>6.0f}%")
        top = sorted(cdict.items(), key=lambda kv: -kv[1])[:6]
        for n, v in top:
            print(f"        {n:32} {v:8.1f}ms")
