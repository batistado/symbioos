#!/usr/bin/env python3
"""Targeted Qwen2.5-7B-Instruct Utilization-Gated Streaming Micro-Prefill Ablation."""

import json
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pilot"))
import phase1_multimodel_devtools as p1  # noqa: E402


def main() -> None:
    mid = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, local_files_only=True).to("mps").eval()
    p1.warmup_mps_buckets(model, device="mps")
    templates = p1.build_base_prompt_templates(model, tok, [1024, 1280, 1536, 1792], device="mps")
    kv_b = p1.cache_kv_bytes(templates[1024]) // 1024

    seeds = [42, 137]
    workloads = {sd: p1.generate_devtool_workload(sd, num_episodes=16) for sd in seeds}

    configs = [
        ("symbioos_sched", 0, None, "SymbioOS-Sched (No Stream)"),
        ("symbioos_full", 256, None, "SymbioOS-Full (Ungated 256)"),
        ("symbioos_full", 128, 0.55, "SymbioOS-Full-Gated (128tok, util<0.55)"),
    ]
    out_rows = []
    for pol, ch, gate, lbl in configs:
        runs = []
        for sd in seeds:
            res = p1.run_devtool_policy(
                model_id=mid,
                policy=pol,
                seed=sd,
                episodes=workloads[sd],
                model=model,
                tokenizer=tok,
                base_templates=templates,
                kv_bytes_per_tok=kv_b,
                max_concurrency=10,
                hbm_cap_mb=500.0,
                cpu_cores=3,
                nvme_bw_mb_s=250.0,
                micro_chunk_tokens=ch,
                device="mps",
                stream_util_gate=gate,
            )
            runs.append(res)
        s = p1.summarize_group({"label": lbl}, runs)
        out_rows.append(s)
        print(f"{lbl:40s} | P50={s['ttft_p50_ms']:7.1f}ms P95={s['ttft_p95_ms']:7.1f}ms | Hit={s['hbm_hit_rate_pct']:5.1f}% | JCT={s['episode_jct_mean_sec']:5.2f}s | Chunks={s['micro_chunks_prefilled']}")

    out_path = Path(__file__).resolve().parent.parent / "results" / "qwen7b_gated_stream_ablation.json"
    out_path.write_text(json.dumps(out_rows, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
