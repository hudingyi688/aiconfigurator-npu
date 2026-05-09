"""
Convert aiconfigurator-npu TensorCast CSV format to aiconfigurator perf .txt format.

Usage:
    python tools/convert_to_aiconfigurator.py \
        --input-dir systems/data/ascend_910b/vllm-ascend/0.18.0 \
        --output-dir /path/to/aiconfigurator/systems/data/ascend_910b/vllm-ascend/0.18.0 \
        --device "Ascend 910B" \
        --framework vllm-ascend \
        --version 0.18.0
"""

import argparse
import csv
import os
import re
import sys


# Quant type mapping: TensorCast → aiconfigurator
GEMM_QUANT_MAP = {
    "bf16": "float16",
    "bfloat16": "float16",
    "w8a8_dynamic": "sq",
    "w8a8_static": "sq",
    "fp8": "fp8",
    "float16": "float16",
}

MOE_QUANT_MAP = {
    "bf16": "float16",
    "bfloat16": "float16",
    "w8a8_dynamic": "float16",  # aiconfigurator MoEQuantMode has no sq; map to float16 with W8A8 latency
    "w8a8_static": "float16",
    "fp8": "fp8",
    "float16": "float16",
}

ATTN_QUANT_MAP = {
    "bf16": "float16",
    "bfloat16": "float16",
    "float16": "float16",
    "fp8": "fp8",
}


def _parse_input_shapes(shapes_str: str) -> list[list[int]]:
    """Parse 'M,N;K,L' → [[M,N],[K,L]]"""
    return [[int(x) for x in s.split(",")] for s in shapes_str.split(";")]


def _us_to_ms(us: float) -> float:
    return us / 1000.0


def convert_gemm(input_path: str, output_path: str, device: str, framework: str, version: str) -> int:
    """Convert MatMulV2.csv + QuantBatchMatmulV3.csv → gemm_perf.txt"""
    rows_out = []

    for fname in ("MatMulV2.csv", "QuantBatchMatmulV3.csv"):
        fpath = os.path.join(input_path, fname)
        if not os.path.exists(fpath):
            print(f"  [skip] {fname} not found")
            continue

        with open(fpath, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                quant_raw = row.get("Quant Type", "bf16").strip().lower()
                quant = GEMM_QUANT_MAP.get(quant_raw)
                if quant is None:
                    print(f"  [warn] unknown quant type '{quant_raw}' in {fname}, skipping row")
                    continue

                shapes = _parse_input_shapes(row["Input Shapes"])
                # MatMul: input0=[M,K], input1=[K,N] → output=[M,N]
                # shapes[0] = [M, K], shapes[1] = [K, N]  (or [N, K] transposed)
                if len(shapes) < 2:
                    continue
                m = shapes[0][0]
                k = shapes[0][1]
                n = shapes[1][1] if len(shapes[1]) > 1 else shapes[1][0]

                latency_ms = _us_to_ms(float(row["Average Duration(us)"]))

                rows_out.append({
                    "framework": framework,
                    "version": version,
                    "device": device,
                    "op_name": "gemm",
                    "kernel_source": "vllm_ascend_default",
                    "gemm_dtype": quant,
                    "m": m,
                    "n": n,
                    "k": k,
                    "latency": latency_ms,
                })

    if not rows_out:
        print("  [warn] no GEMM rows converted")
        return 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["framework", "version", "device", "op_name", "kernel_source",
                  "gemm_dtype", "m", "n", "k", "latency"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  gemm_perf.txt: {len(rows_out)} rows")
    return len(rows_out)


def convert_moe(input_path: str, output_path: str, device: str, framework: str, version: str) -> int:
    """Convert GroupedMatmul_MoE_BF16/W8A8.csv → moe_perf.txt"""
    rows_out = []

    for fname in ("GroupedMatmul_MoE_BF16.csv", "GroupedMatmul_MoE_W8A8.csv"):
        fpath = os.path.join(input_path, fname)
        if not os.path.exists(fpath):
            print(f"  [skip] {fname} not found")
            continue

        with open(fpath, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                quant_raw = row.get("Quant Type", "bf16").strip().lower()
                quant = MOE_QUANT_MAP.get(quant_raw)
                if quant is None:
                    print(f"  [warn] unknown quant type '{quant_raw}' in {fname}, skipping row")
                    continue

                num_tokens = int(row["Num Tokens"])
                hidden_size = int(row["Hidden Size"])
                inter_size = int(row["Intermediate Size"])
                num_experts = int(row["Num Experts"])
                topk = int(row["TopK"])
                ep_size = int(row["EP Size"])
                local_experts = int(row["Local Experts"])
                moe_tp_size = max(1, num_experts // local_experts // ep_size) if local_experts > 0 else 1

                latency_ms = _us_to_ms(float(row["Average Duration(us)"]))

                rows_out.append({
                    "framework": framework,
                    "version": version,
                    "device": device,
                    "op_name": "moe",
                    "kernel_source": "vllm_ascend_fused_moe",
                    "moe_dtype": quant,
                    "num_tokens": num_tokens,
                    "hidden_size": hidden_size,
                    "inter_size": inter_size,
                    "topk": topk,
                    "num_experts": num_experts,
                    "moe_tp_size": moe_tp_size,
                    "moe_ep_size": ep_size,
                    "distribution": "power_law_1.2",
                    "latency": latency_ms,
                })

    if not rows_out:
        print("  [warn] no MoE rows converted")
        return 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["framework", "version", "device", "op_name", "kernel_source",
                  "moe_dtype", "num_tokens", "hidden_size", "inter_size", "topk",
                  "num_experts", "moe_tp_size", "moe_ep_size", "distribution", "latency"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  moe_perf.txt: {len(rows_out)} rows")
    return len(rows_out)


def convert_context_attention(input_path: str, output_path: str, device: str, framework: str, version: str) -> int:
    """Convert FusedInferAttentionScore.csv → context_attention_perf.txt"""
    fpath = os.path.join(input_path, "FusedInferAttentionScore.csv")
    if not os.path.exists(fpath):
        print("  [skip] FusedInferAttentionScore.csv not found")
        return 0

    rows_out = []
    with open(fpath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_type = row.get("Op Type", "attention_context").strip()
            if "generation" in op_type.lower():
                continue  # skip decode rows if mixed

            batch = int(row["Batch"])
            seq_len = int(row["Seq Len"])
            num_heads = int(row["Num Heads"])
            num_kv_heads_raw = int(row["Num KV Heads"])
            head_size = int(row["Head Size"])
            # 0 means MHA (kv_heads == heads) in aiconfigurator convention
            num_kv_heads = 0 if num_kv_heads_raw == 0 or num_kv_heads_raw == num_heads else num_kv_heads_raw

            latency_ms = _us_to_ms(float(row["Average Duration(us)"]))

            rows_out.append({
                "framework": framework,
                "version": version,
                "device": device,
                "op_name": "context_attention",
                "kernel_source": "vllm_ascend_flash_attn",
                "batch_size": batch,
                "isl": seq_len,
                "num_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_size,
                "beam_width": 1,
                "attn_dtype": "float16",
                "kv_cache_dtype": "float16",
                "step": 0,
                "latency": latency_ms,
            })

    if not rows_out:
        print("  [warn] no context attention rows converted")
        return 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["framework", "version", "device", "op_name", "kernel_source",
                  "batch_size", "isl", "num_heads", "num_key_value_heads", "head_dim",
                  "beam_width", "attn_dtype", "kv_cache_dtype", "step", "latency"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  context_attention_perf.txt: {len(rows_out)} rows")
    return len(rows_out)


def convert_generation_attention(input_path: str, output_path: str, device: str, framework: str, version: str) -> int:
    """Convert FusedInferAttentionScore_Decode.csv → generation_attention_perf.txt"""
    fpath = os.path.join(input_path, "FusedInferAttentionScore_Decode.csv")
    if not os.path.exists(fpath):
        print("  [skip] FusedInferAttentionScore_Decode.csv not found")
        return 0

    rows_out = []
    with open(fpath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch = int(row["Batch"])
            seq_len = int(row["Seq Len"])
            num_heads = int(row["Num Heads"])
            num_kv_heads_raw = int(row["Num KV Heads"])
            head_size = int(row["Head Size"])
            num_kv_heads = 0 if num_kv_heads_raw == 0 or num_kv_heads_raw == num_heads else num_kv_heads_raw

            latency_ms = _us_to_ms(float(row["Average Duration(us)"]))

            # generation attention: step = context_len, isl = 1 (one new token)
            # aiconfigurator stores s = isl + step, so step = seq_len - 1
            step = max(0, seq_len - 1)

            rows_out.append({
                "framework": framework,
                "version": version,
                "device": device,
                "op_name": "generation_attention",
                "kernel_source": "vllm_ascend_flash_attn",
                "batch_size": batch,
                "isl": 1,
                "num_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_size,
                "beam_width": 1,
                "attn_dtype": "float16",
                "kv_cache_dtype": "float16",
                "step": step,
                "latency": latency_ms,
            })

    if not rows_out:
        print("  [warn] no generation attention rows converted")
        return 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["framework", "version", "device", "op_name", "kernel_source",
                  "batch_size", "isl", "num_heads", "num_key_value_heads", "head_dim",
                  "beam_width", "attn_dtype", "kv_cache_dtype", "step", "latency"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  generation_attention_perf.txt: {len(rows_out)} rows")
    return len(rows_out)


def convert_dsa_module(input_path: str, output_path_ctx: str, output_path_gen: str,
                       device: str, framework: str, version: str) -> int:
    """Convert FusedInferAttentionScore_MLA.csv → dsa_context_module_perf.txt
    and FusedInferAttentionScore_Decode_MLA.csv → dsa_generation_module_perf.txt.

    The TensorCast MLA CSV must have been produced by collect_mla.py with the
    Architecture column present (added in the updated collector).
    """
    ctx_rows: list[dict] = []
    gen_rows: list[dict] = []

    file_map = {
        "FusedInferAttentionScore_MLA.csv": "context",
        "FusedInferAttentionScore_Decode_MLA.csv": "generation",
    }

    for fname, phase in file_map.items():
        fpath = os.path.join(input_path, fname)
        if not os.path.exists(fpath):
            print(f"  [skip] {fname} not found")
            continue

        with open(fpath, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                batch = int(row["Batch"])
                seq_len = int(row["Seq Len"])
                num_heads = int(row["Num Heads"])
                architecture = row.get("Architecture", "GlmMoeDsaForCausalLM").strip()
                latency_ms = _us_to_ms(float(row["Average Duration(us)"]))

                base = {
                    "framework": framework,
                    "version": version,
                    "device": device,
                    "kernel_source": "vllm_ascend_mla",
                    "batch_size": batch,
                    "num_heads": num_heads,
                    "gemm_type": "float16",
                    "mla_dtype": "float16",
                    "kv_cache_dtype": "float16",
                    "architecture": architecture,
                    "latency": latency_ms,
                }

                if phase == "context":
                    base["op_name"] = "dsa_context_module"
                    base["isl"] = seq_len
                    ctx_rows.append(base)
                else:
                    base["op_name"] = "dsa_generation_module"
                    base["isl"] = 1
                    base["step"] = max(0, seq_len - 1)
                    gen_rows.append(base)

    total = 0

    if ctx_rows:
        os.makedirs(os.path.dirname(output_path_ctx), exist_ok=True)
        ctx_fields = ["framework", "version", "device", "op_name", "kernel_source",
                      "batch_size", "isl", "num_heads", "gemm_type", "mla_dtype",
                      "kv_cache_dtype", "architecture", "latency"]
        with open(output_path_ctx, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ctx_fields)
            writer.writeheader()
            writer.writerows(ctx_rows)
        print(f"  dsa_context_module_perf.txt: {len(ctx_rows)} rows")
        total += len(ctx_rows)

    if gen_rows:
        os.makedirs(os.path.dirname(output_path_gen), exist_ok=True)
        gen_fields = ["framework", "version", "device", "op_name", "kernel_source",
                      "batch_size", "isl", "num_heads", "gemm_type", "mla_dtype",
                      "kv_cache_dtype", "architecture", "step", "latency"]
        with open(output_path_gen, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=gen_fields)
            writer.writeheader()
            writer.writerows(gen_rows)
        print(f"  dsa_generation_module_perf.txt: {len(gen_rows)} rows")
        total += len(gen_rows)

    if not ctx_rows and not gen_rows:
        print("  [warn] no DSA module rows converted")

    return total


def main():
    parser = argparse.ArgumentParser(description="Convert aiconfigurator-npu CSV data to aiconfigurator txt format")
    parser.add_argument("--input-dir", required=True, help="Input directory with TensorCast CSV files")
    parser.add_argument("--output-dir", required=True, help="Output directory for aiconfigurator txt files")
    parser.add_argument("--device", default="Ascend 910B", help="Device name string in output files")
    parser.add_argument("--framework", default="vllm-ascend", help="Framework name")
    parser.add_argument("--version", default="0.18.0", help="Framework version")
    args = parser.parse_args()

    print(f"Converting: {args.input_dir} → {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    total = 0
    total += convert_gemm(
        args.input_dir,
        os.path.join(args.output_dir, "gemm_perf.txt"),
        args.device, args.framework, args.version,
    )
    total += convert_moe(
        args.input_dir,
        os.path.join(args.output_dir, "moe_perf.txt"),
        args.device, args.framework, args.version,
    )
    total += convert_context_attention(
        args.input_dir,
        os.path.join(args.output_dir, "context_attention_perf.txt"),
        args.device, args.framework, args.version,
    )
    total += convert_generation_attention(
        args.input_dir,
        os.path.join(args.output_dir, "generation_attention_perf.txt"),
        args.device, args.framework, args.version,
    )

    total += convert_dsa_module(
        args.input_dir,
        os.path.join(args.output_dir, "dsa_context_module_perf.txt"),
        os.path.join(args.output_dir, "dsa_generation_module_perf.txt"),
        args.device, args.framework, args.version,
    )

    print(f"\nDone. Total rows written: {total}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
