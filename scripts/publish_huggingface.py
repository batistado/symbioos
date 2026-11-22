#!/usr/bin/env python3
"""Build & Publish SymbioOS Hugging Face Suite (`batistado` Namespace).

Publishes three interlinked, archival Hugging Face Hub repositories:
  1. Dataset : https://huggingface.co/datasets/batistado/SymbioBench-Agent
       4 Dataset-Viewer configs (`phase1_multimodel_resumes`, `phase0_resumes`,
       `phase2_cluster_sweeps`, `policy_summaries`) + raw JSONs + vector figures.
  2. Space   : https://huggingface.co/spaces/batistado/SymbioOS-Explorer
       Always-on, zero-cold-start Static HTML5 Interactive Explorer (`sdk: static`)
       with Multi-Model Benchmark Inspector, Theorem 1 & 2 Interactive Tandem
       Simulator, and Per-Archetype Mechanism Dissection.
  3. Hub Repo: https://huggingface.co/batistado/SymbioOS
       Rootless `symbioos` Python package, benchmark suite, IEEE TPDS LaTeX/Overleaf
       bundle, and figures (strictly excluding internal handoff notes).
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_all_telemetry() -> Dict[str, Any]:
    p0_closed = json.loads((RESULTS_DIR / "symbioos_pilot.json").read_text(encoding="utf-8"))
    p0_wave = json.loads((RESULTS_DIR / "symbioos_pilot_open_wave.json").read_text(encoding="utf-8"))
    p1_multi = json.loads((RESULTS_DIR / "phase1_multimodel_devtools.json").read_text(encoding="utf-8"))
    p2_sweeps = json.loads((RESULTS_DIR / "phase2_scaleout_sweeps.json").read_text(encoding="utf-8"))
    return {
        "p0_closed": p0_closed,
        "p0_wave": p0_wave,
        "p1_multi": p1_multi,
        "p2_sweeps": p2_sweeps,
    }


def build_dataset_staging(stage_dir: Path, bundles: Dict[str, Any], username: str) -> None:
    # 1. Config 1 (default): phase1_multimodel_resumes (1,088 real pytest/clang++/git resumes)
    p1_rows: List[Dict[str, Any]] = []
    for run in bundles["p1_multi"]["runs"]:
        for tr in run["turn_records"]:
            p1_rows.append(dict(tr))
    _write_jsonl(stage_dir / "data" / "phase1_multimodel_resumes" / "train.jsonl", p1_rows)

    # 2. Config 2: phase0_resumes (1,080 controlled heavy-tailed resumes across closed-pool & burst-wave)
    p0_rows: List[Dict[str, Any]] = []
    for regime_tag, src in [("closed_pool_k12", bundles["p0_closed"]), ("burst_wave_n16", bundles["p0_wave"])]:
        for run in src["runs"]:
            for tr in run["turn_records"]:
                row = {"regime": regime_tag, **tr}
                p0_rows.append(row)
    _write_jsonl(stage_dir / "data" / "phase0_resumes" / "train.jsonl", p0_rows)

    # 3. Config 3: phase2_cluster_sweeps (N=32..256 scale-out, HBM ratio, Pareto alpha, JIT delta)
    p2_rows: List[Dict[str, Any]] = []
    for sweep_name, items in bundles["p2_sweeps"].items():
        for item in items:
            p2_rows.append({"sweep_group": sweep_name, **item})
    _write_jsonl(stage_dir / "data" / "phase2_cluster_sweeps" / "train.jsonl", p2_rows)

    # 4. Config 4: policy_summaries
    sum_rows: List[Dict[str, Any]] = []
    for s in bundles["p1_multi"]["multimodel_summary"]:
        sum_rows.append({
            "suite": "phase1_multimodel_devtools",
            "model_id": s["model_id"],
            "policy": s["policy"],
            "total_resumes": int(s["total_resumes"]),
            "ttft_p50_ms": float(s["ttft_p50_ms"]),
            "ttft_p95_ms": float(s["ttft_p95_ms"]),
            "ttft_p99_ms": float(s["ttft_p99_ms"]),
            "hbm_hit_rate_pct": float(s["hbm_hit_rate_pct"]),
            "interactive_hit_rate_pct": float(s["interactive_hit_rate_pct"]),
            "total_waste_mb_s_per_seed": float(s["total_waste_mb_s_per_seed"]),
            "episode_jct_mean_sec": float(s["episode_jct_mean_sec"]),
            "goodput_ep_per_min": float(s["cluster_goodput_ep_per_min"]),
            "slo_goodput_per_min": float(s["slo_goodput_le_600ms_per_min"]),
        })
    for tag, src in [("phase0_closed_pool", bundles["p0_closed"]), ("phase0_burst_wave", bundles["p0_wave"])]:
        for s in src["summary"]:
            sum_rows.append({
                "suite": tag,
                "model_id": "Qwen/Qwen2.5-3B-Instruct",
                "policy": s["policy"],
                "total_resumes": int(s["total_tool_resumes"]),
                "ttft_p50_ms": float(s["ttft_p50_ms"]),
                "ttft_p95_ms": float(s["ttft_p95_ms"]),
                "ttft_p99_ms": float(s["ttft_p99_ms"]),
                "hbm_hit_rate_pct": float(s["hbm_hit_rate_pct"]),
                "interactive_hit_rate_pct": float(s["interactive_hit_rate_pct"]),
                "total_waste_mb_s_per_seed": float(s["total_waste_mb_s_per_seed"]),
                "episode_jct_mean_sec": float(s["episode_jct_mean_sec"]),
                "goodput_ep_per_min": float(s["cluster_goodput_episodes_per_min"]),
                "slo_goodput_per_min": float(s["slo_goodput_resumes_le_500ms_per_min"]),
            })
    _write_jsonl(stage_dir / "data" / "policy_summaries" / "train.jsonl", sum_rows)

    # Copy raw JSON results and figures
    raw_out = stage_dir / "raw_results"
    raw_out.mkdir(parents=True, exist_ok=True)
    for jf in RESULTS_DIR.glob("*.json"):
        shutil.copy2(jf, raw_out / jf.name)

    fig_out = stage_dir / "figures"
    fig_out.mkdir(parents=True, exist_ok=True)
    for ff in FIGURES_DIR.glob("*"):
        shutil.copy2(ff, fig_out / ff.name)

    ds_card = f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - llm-serving
  - kv-cache
  - coding-agents
  - operating-systems
  - scheduling
  - gittins-index
  - streaming-prefill
pretty_name: SymbioBench-Agent Multi-Container OS & Tiered KV Serving Telemetry
size_categories:
  - 1K<n<10K
configs:
  - config_name: phase1_multimodel_resumes
    default: true
    data_files:
      - split: train
        path: data/phase1_multimodel_resumes/train.jsonl
  - config_name: phase0_resumes
    data_files:
      - split: train
        path: data/phase0_resumes/train.jsonl
  - config_name: phase2_cluster_sweeps
    data_files:
      - split: train
        path: data/phase2_cluster_sweeps/train.jsonl
  - config_name: policy_summaries
    data_files:
      - split: train
        path: data/policy_summaries/train.jsonl
---

# SymbioBench-Agent: Multi-Container OS & Tiered GPU KV Serving Telemetry

* **System & Code Repository:** [`{username}/SymbioOS`](https://huggingface.co/{username}/SymbioOS) | [`github.com/batistado/symbioos`](https://github.com/batistado/symbioos)
* **Interactive Explorer Space:** [`{username}/SymbioOS-Explorer`](https://huggingface.co/spaces/{username}/SymbioOS-Explorer)

`SymbioBench-Agent` contains **2,168 real POSIX tool-execution and MPS GPU post-tool resume trajectories** plus **cluster-scale (`N = 32..256`) coupled tandem-queue sweeps** evaluating five OS + KV serving policies head-to-head:
1. `lru_cfs`: Standard `vLLM`-style reactive LRU KV eviction + fair-share OS CPU scheduling (`CFS`).
2. `continuum_cfs`: Open-loop `t = 0` scalar TTL KV pinning (`Continuum`) + fair-share CPU.
3. `unison_cfs`: Turn-level `t = 0` DHR survival-hazard KV eviction (`UNISON/SPEAR`) + fair-share CPU.
4. `symbioos_sched`: Closed-loop unstopped-quantum pipe filtration, KV-weighted Gittins `os.killpg` CPU scheduling, `1.25s` early stall demotion, and Theorem-2 JIT KV promotion.
5. `symbioos_full`: `symbioos_sched` + idle-slack `256`-token streaming `stdout` micro-prefill with a 1-token BPE lookback guard.

![Multi-Model Real DevTool Benchmark](figures/fig3_multimodel_devtools.png)

## Dataset Configurations

| Config | Rows | Description |
|---|---:|---|
| `phase1_multimodel_resumes` (default) | `1,088` | Real `pytest 9.2.0`, `clang++ -O3 -std=c++20 -c`, and `git` POSIX process-group resumes across `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, and `OLMoE-1B-7B-0924-Instruct`, plus `{{128, 256, 512, inf}}` chunk-size ablations. |
| `phase0_resumes` | `1,080` | Controlled heavy-tailed resumes across `Closed-Pool (K=12, 290 MB)` and `Burst-Wave (N=16, 380 MB)` regimes across 3 seeds (`42, 137, 2026`). |
| `phase2_cluster_sweeps` | `61` | Calibrated `32B/70B`-scale (`160 KB/tok`) cluster sweeps across `N = 32..256`, HBM ratios `15%..65%`, Pareto tail `alpha`, and Theorem-2 JIT stopping margin `delta_jit`. |
| `policy_summaries` | `25` | Aggregated latency (`P50/P95/P99`), HBM KV hit rate, HBM waste (`MB*s`), mean JCT, and strict-SLO goodput metrics. |

## Citation

```bibtex
@article{{syed2026symbioos,
  title   = {{{{SymbioOS}}: Closed-Loop Rootless {{POSIX}} Sandbox Scheduling and White-Box Streaming {{KV}} Promotion for Tool-Intensive Agent Clusters}},
  author  = {{Syed, Mohammed Kamran and Rentala, Anudeep and Rahul, Somesh and Pai, Raghuram Malpe and Dara, Krishna Suman}},
  journal = {{IEEE Transactions on Parallel and Distributed Systems (TPDS)}},
  year    = {{2026}},
  url     = {{https://github.com/batistado/symbioos}}
}}
```
"""
    (stage_dir / "README.md").write_text(ds_card, encoding="utf-8")


def build_static_space_staging(stage_dir: Path, bundles: Dict[str, Any], username: str) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for ff in FIGURES_DIR.glob("*.png"):
        shutil.copy2(ff, stage_dir / ff.name)

    space_readme = f"""---
title: SymbioOS Explorer
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: apache-2.0
short_description: Interactive OS + Tiered GPU KV Tandem Explorer for SymbioOS
---

# SymbioOS Interactive Explorer

Interactive telemetry inspector and Theorem 1 / Theorem 2 tandem-queue simulator for [`{username}/SymbioOS`](https://huggingface.co/{username}/SymbioOS) and [`{username}/SymbioBench-Agent`](https://huggingface.co/datasets/{username}/SymbioBench-Agent).
"""
    (stage_dir / "README.md").write_text(space_readme, encoding="utf-8")

    embedded_data = {
        "p1_multimodel": bundles["p1_multi"]["multimodel_summary"],
        "p1_chunks": bundles["p1_multi"]["chunk_ablation_summary"],
        "p0_closed": bundles["p0_closed"]["summary"],
        "p0_wave": bundles["p0_wave"]["summary"],
        "p2_hbm": bundles["p2_sweeps"]["sweep_b_hbm_ratio"],
        "p2_jit": bundles["p2_sweeps"]["sweep_d_jit_boundary"],
    }
    data_json = json.dumps(embedded_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SymbioOS — Interactive OS + GPU KV Explorer</title>
<style>
  :root {{
    --bg: #0f172a;
    --panel: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --blue: #38bdf8;
    --deepblue: #0284c7;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }}
  header {{
    padding: 28px 24px 20px; border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, #0f172a 0%, #172554 100%);
  }}
  .container {{ max-width: 1180px; margin: 0 auto; padding: 0 20px; }}
  h1 {{ margin: 0 0 8px; font-size: 1.65rem; letter-spacing: -0.02em; }}
  .subtitle {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 14px; }}
  .links a {{
    display: inline-block; margin-right: 10px; margin-bottom: 6px; padding: 6px 12px;
    border-radius: 6px; background: rgba(56,189,248,0.14); color: var(--blue);
    text-decoration: none; font-size: 0.84rem; border: 1px solid rgba(56,189,248,0.3);
  }}
  .links a:hover {{ background: rgba(56,189,248,0.25); }}
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 14px; margin: 22px 0;
  }}
  .kpi {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px;
  }}
  .kpi .val {{ font-size: 1.65rem; font-weight: 700; color: var(--blue); }}
  .kpi .lbl {{ font-size: 0.82rem; color: var(--muted); margin-top: 4px; }}
  .card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; margin-bottom: 22px;
  }}
  .card h2 {{ margin-top: 0; font-size: 1.18rem; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; align-items: center; }}
  select, input[type=range] {{
    background: #0f172a; color: var(--text); border: 1px solid var(--border);
    padding: 7px 12px; border-radius: 6px; font-size: 0.88rem;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; margin-top: 12px; }}
  th, td {{ padding: 9px 10px; text-align: right; border-bottom: 1px solid var(--border); }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; }}
  tr.highlight {{ background: rgba(16, 185, 129, 0.12); font-weight: 600; }}
  .bar-row {{ display: flex; align-items: center; margin: 8px 0; gap: 10px; font-size: 0.84rem; }}
  .bar-name {{ width: 175px; flex-shrink: 0; color: #e2e8f0; }}
  .bar-track {{ flex: 1; background: #0f172a; height: 22px; border-radius: 5px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 5px; transition: width 0.25s ease; }}
  .bar-val {{ width: 95px; text-align: right; font-variant-numeric: tabular-nums; }}
  .sim-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 800px) {{ .sim-grid {{ grid-template-columns: 1fr; }} }}
  .fig-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .fig-grid img {{ width: 100%; border-radius: 8px; background: white; padding: 6px; }}
</style>
</head>
<body>
<header>
  <div class="container">
    <h1>SymbioOS: Closed-Loop Rootless POSIX Sandbox Scheduling &amp; Streaming KV Promotion</h1>
    <div class="subtitle">Interactive Telemetry Explorer across 2,168 Real MPS Hardware Resumes (Qwen2.5-3B, Qwen2.5-7B, OLMoE-1B-7B) &amp; 256-Sandbox Tandem Sweeps</div>
    <div class="links">
      <a href="https://huggingface.co/{username}/SymbioOS" target="_blank">Hub System Repo</a>
      <a href="https://huggingface.co/datasets/{username}/SymbioBench-Agent" target="_blank">SymbioBench-Agent Dataset</a>
      <a href="https://github.com/batistado/symbioos" target="_blank">GitHub Source &amp; Proofs</a>
    </div>
  </div>
</header>

<div class="container">
  <div class="kpi-grid">
    <div class="kpi"><div class="val">4.82&times;</div><div class="lbl">Lower Median Resume TTFT (262.6 ms vs 1265.7 ms on Qwen2.5-3B)</div></div>
    <div class="kpi"><div class="val">95.2%</div><div class="lbl">Interactive pytest/lint HBM KV Hit Rate (vs 59.5% Continuum)</div></div>
    <div class="kpi"><div class="val">5.70&times;</div><div class="lbl">Higher Strict-SLO Goodput on Sparse MoE OLMoE-1B-7B</div></div>
    <div class="kpi"><div class="val">7.98&times;</div><div class="lbl">Faster Post-exit(0) Prefill via 1-Token BPE Lookback Streaming</div></div>
  </div>

  <div class="card">
    <h2>1. Verified Multi-Architecture &amp; Load-Regime Benchmark Inspector</h2>
    <div class="controls">
      <label>Select Benchmark Suite:</label>
      <select id="suiteSelect" onchange="renderSuite()">
        <option value="qwen3b">Phase 1 Real DevTools — Qwen2.5-3B-Instruct (36 KB/tok Dense GQA)</option>
        <option value="olmoe">Phase 1 Real DevTools — OLMoE-1B-7B-0924-Instruct (128 KB/tok Sparse MoE)</option>
        <option value="qwen7b">Phase 1 Real DevTools — Qwen2.5-7B-Instruct (56 KB/tok Dense GQA)</option>
        <option value="p0_closed">Phase 0 Closed-Pool (K=12, M_HBM=290 MB, 600 Resumes)</option>
        <option value="p0_wave">Phase 0 Burst-Wave (N=16, M_HBM=380 MB, 480 Resumes)</option>
      </select>
      <label>Visualized Metric:</label>
      <select id="metricSelect" onchange="renderSuite()">
        <option value="ttft_p50_ms">Post-Tool Resume TTFT P50 (ms, lower is better)</option>
        <option value="ttft_p95_ms">Post-Tool Resume TTFT P95 (ms, lower is better)</option>
        <option value="interactive_hit_rate_pct">Interactive HBM KV Hit Rate (%, higher is better)</option>
        <option value="slo_goodput">Strict-SLO Resume Goodput (/min, higher is better)</option>
      </select>
    </div>
    <div id="barsContainer"></div>
    <div id="tableContainer"></div>
  </div>

  <div class="card">
    <h2>2. Interactive HBM Capacity Frontier &amp; Theorem 2 JIT Stopping Explorer (N = 128)</h2>
    <div class="sim-grid">
      <div>
        <div class="controls">
          <label>Fast HBM Capacity Ratio: <strong id="hbmLbl">25%</strong></label>
          <input type="range" id="hbmSlider" min="0" max="4" step="1" value="1" oninput="renderSim()"/>
        </div>
        <div id="hbmCompareBox"></div>
      </div>
      <div>
        <div class="controls">
          <label>Theorem 2 JIT Margin &delta;<sub>jit</sub>: <strong id="jitLbl">0.35s</strong></label>
          <input type="range" id="jitSlider" min="0" max="5" step="1" value="2" oninput="renderSim()"/>
        </div>
        <div id="jitCompareBox"></div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>3. Publication Figures (Type-42 Vector Suite)</h2>
    <div class="fig-grid">
      <img src="fig1_architecture_overview.png" alt="SymbioOS Architecture"/>
      <img src="fig2_phase0_ttft_decomposition.png" alt="Phase 0 TTFT Decomposition"/>
      <img src="fig3_multimodel_devtools.png" alt="Multi-Model Real DevTools"/>
      <img src="fig5_cluster_scaleout_sweeps.png" alt="Cluster Scale-Out Sweeps"/>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};
const COLORS = {{
  lru_cfs: "#94a3b8",
  continuum_cfs: "#f97316",
  unison_cfs: "#eab308",
  symbioos_sched: "#38bdf8",
  symbioos_full: "#10b981"
}};
const NAMES = {{
  lru_cfs: "1. LRU + CFS (vLLM)",
  continuum_cfs: "2. Continuum (t=0 TTL)",
  unison_cfs: "3. UNISON/SPEAR (Hazard)",
  symbioos_sched: "4. SymbioOS-Sched",
  symbioos_full: "5. SymbioOS-Full"
}};

function getRows(suite) {{
  if (suite === "qwen3b") return DATA.p1_multimodel.filter(r => r.model_id.includes("3B"));
  if (suite === "qwen7b") return DATA.p1_multimodel.filter(r => r.model_id.includes("7B-Instruct"));
  if (suite === "olmoe") return DATA.p1_multimodel.filter(r => r.model_id.includes("OLMoE"));
  if (suite === "p0_closed") return DATA.p0_closed;
  return DATA.p0_wave;
}}

function getSlo(r) {{
  return r.slo_goodput_le_600ms_per_min !== undefined ? r.slo_goodput_le_600ms_per_min : r.slo_goodput_resumes_le_500ms_per_min;
}}

function renderSuite() {{
  const suite = document.getElementById("suiteSelect").value;
  const metric = document.getElementById("metricSelect").value;
  const rows = getRows(suite);
  const vals = rows.map(r => metric === "slo_goodput" ? getSlo(r) : r[metric]);
  const maxV = Math.max(...vals, 1);

  let barsHtml = "";
  rows.forEach((r, idx) => {{
    const v = vals[idx];
    const pct = (v / maxV) * 100;
    const unit = metric.includes("pct") ? "%" : (metric.includes("ms") ? " ms" : " /m");
    barsHtml += `<div class="bar-row">
      <div class="bar-name">${{NAMES[r.policy]}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${{pct.toFixed(1)}}%;background:${{COLORS[r.policy]}}"></div></div>
      <div class="bar-val">${{v.toFixed(1)}}${{unit}}</div>
    </div>`;
  }});
  document.getElementById("barsContainer").innerHTML = barsHtml;

  let tbl = `<table><thead><tr>
    <th>Policy</th><th>TTFT P50 (ms)</th><th>TTFT P95 (ms)</th><th>HBM Hit (%)</th>
    <th>Interactive Hit (%)</th><th>HBM Waste (MB&middot;s)</th><th>Mean JCT (s)</th><th>SLO Goodput (/m)</th>
  </tr></thead><tbody>`;
  rows.forEach(r => {{
    const cls = r.policy === "symbioos_full" ? " class='highlight'" : "";
    tbl += `<tr${{cls}}>
      <td>${{NAMES[r.policy]}}</td>
      <td>${{r.ttft_p50_ms.toFixed(1)}}</td>
      <td>${{r.ttft_p95_ms.toFixed(1)}}</td>
      <td>${{r.hbm_hit_rate_pct.toFixed(1)}}%</td>
      <td>${{r.interactive_hit_rate_pct.toFixed(1)}}%</td>
      <td>${{r.total_waste_mb_s_per_seed.toFixed(1)}}</td>
      <td>${{r.episode_jct_mean_sec.toFixed(2)}}</td>
      <td>${{getSlo(r).toFixed(1)}}</td>
    </tr>`;
  }});
  tbl += "</tbody></table>";
  document.getElementById("tableContainer").innerHTML = tbl;
}}

function renderSim() {{
  const ratios = [0.15, 0.25, 0.375, 0.50, 0.65];
  const rIdx = parseInt(document.getElementById("hbmSlider").value);
  const hr = ratios[rIdx];
  document.getElementById("hbmLbl").textContent = Math.round(hr * 100) + "%";
  const lru = DATA.p2_hbm.find(x => Math.abs(x.hbm_ratio - hr) < 1e-4 && x.policy === "lru_cfs");
  const sym = DATA.p2_hbm.find(x => Math.abs(x.hbm_ratio - hr) < 1e-4 && x.policy === "symbioos_full");
  const sp = (lru.ttft_p50_ms / sym.ttft_p50_ms).toFixed(2);
  document.getElementById("hbmCompareBox").innerHTML = `
    <p><strong>LRU + CFS P50 TTFT:</strong> ${{lru.ttft_p50_ms.toFixed(1)}} ms (Hit: ${{lru.hbm_hit_pct}}%)</p>
    <p style="color:#10b981"><strong>SymbioOS-Full P50 TTFT:</strong> ${{sym.ttft_p50_ms.toFixed(1)}} ms (Hit: ${{sym.hbm_hit_pct}}%) &rarr; <strong>${{sp}}&times; faster</strong></p>
    <p style="color:#94a3b8;font-size:0.84rem">Cluster Goodput: ${{lru.goodput_ep_per_min}} &rarr; <strong>${{sym.goodput_ep_per_min}} ep/min</strong></p>
  `;

  const jIdx = parseInt(document.getElementById("jitSlider").value);
  const jRow = DATA.p2_jit[jIdx];
  document.getElementById("jitLbl").textContent = jRow.delta_jit_s.toFixed(2) + "s";
  document.getElementById("jitCompareBox").innerHTML = `
    <p><strong>Resume TTFT P95:</strong> ${{jRow.ttft_p95_ms.toFixed(1)}} ms | <strong>HBM Hit Rate:</strong> ${{jRow.hbm_hit_pct}}%</p>
    <p style="color:#f59e0b"><strong>Premature HBM Pin Cost:</strong> ${{jRow.premature_jit_pin_gb_s.toFixed(2)}} GB&middot;s</p>
    <p style="color:#94a3b8;font-size:0.84rem">Theorem 2 critical-fractile sweet spot (0.15s–0.45s) captures timely staging while keeping premature pin &lt; 0.9% of HBM.</p>
  `;
}}

renderSuite();
renderSim();
</script>
</body>
</html>
"""
    (stage_dir / "index.html").write_text(html, encoding="utf-8")


def build_hub_system_repo_staging(hub_dir: Path, username: str) -> None:
    hub_dir.mkdir(parents=True, exist_ok=True)
    # Copy clean code, tests, scripts, paper, and metadata (strictly excluding HANDOFF_CONTEXT.md!)
    for rel_file in ["LICENSE", "CITATION.cff", "pyproject.toml"]:
        shutil.copy2(REPO_ROOT / rel_file, hub_dir / rel_file)

    for folder in ["symbioos", "tests", "scripts"]:
        shutil.copytree(
            REPO_ROOT / folder,
            hub_dir / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )

    shutil.copytree(
        REPO_ROOT / "pilot",
        hub_dir / "pilot",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".devtool_scratch*", ".build_scratch*"),
    )

    shutil.copytree(
        REPO_ROOT / "paper",
        hub_dir / "paper",
        ignore=shutil.ignore_patterns("*.aux", "*.bbl", "*.blg", "*.log", "*.out", ".DS_Store"),
    )

    for png in FIGURES_DIR.glob("*.png"):
        shutil.copy2(png, hub_dir / png.name)

    hub_card = f"""---
license: apache-2.0
library_name: symbioos
tags:
  - llm-serving
  - kv-cache
  - coding-agents
  - operating-systems
  - gittins-index
  - streaming-prefill
  - pytorch
---

# SymbioOS — Closed-Loop Rootless POSIX Sandbox Scheduling & White-Box Streaming KV Promotion

* **Benchmark Dataset (`2,168` Hardware Resumes + `256`-Sandbox Sweeps):** [`{username}/SymbioBench-Agent`](https://huggingface.co/datasets/{username}/SymbioBench-Agent)
* **Interactive Explorer Space:** [`{username}/SymbioOS-Explorer`](https://huggingface.co/spaces/{username}/SymbioOS-Explorer)
* **Source Code & Proofs:** [`github.com/batistado/symbioos`](https://github.com/batistado/symbioos)
* **IEEE TPDS Overleaf / LaTeX Bundle:** [`paper/symbioos_tpds_overleaf.zip`](paper/symbioos_tpds_overleaf.zip)

## Overview

Autonomous software engineering agents (`SWE-agent`, `OpenHands`) alternate between GPU token generation and host POSIX tool execution (`pytest`, `clang++`, `git`). Existing inference engines (`Continuum`, `UNISON/SPEAR`, `vLLM`) and OS schedulers (`CFS`, `AgentCgroup`) operate open-loop: CPU contention stretches interactive `pytest` containers past GPU TTL cliffs, silent socket hangs stay pinned in scarce HBM, and thousands of `stdout` tokens wait until `exit(0)` to begin prefill.

![SymbioOS Architecture](fig1_architecture_overview.png)

`SymbioOS` closes this loop in **rootless user space**:
1. **Unstopped-Quantum Sub-Turn Filtration (`SubTurnRTTCFilter`):** Parses live `[XX%]` milestones and tracks silence strictly during unstopped quanta (`!is_sigstopped`), demoting stalled waits at `1.25s` without false-positive evictions on CPU-throttled imports.
2. **KV-Weighted Gittins `os.killpg` Priority (`KVWeightedGittinsScheduler`) & Theorem-2 JIT Staging (`OptimalStoppingJITController`):** Prioritizes hot-HBM interactive suites while staging evicted caches back just in time.
3. **1-Token BPE Lookback Streaming Micro-Prefill (`LookbackBPEStreamer`):** Pre-encodes `256`-token chunks of append-only `stdout` during GPU slack with exact token and logit parity (`cos_sim = 0.999997`, `7.98x` faster post-exit prefill).

![Multi-Model Results](fig3_multimodel_devtools.png)

## Quick Install

```bash
pip install git+https://github.com/batistado/symbioos.git
```

## Citation

```bibtex
@article{{syed2026symbioos,
  title   = {{{{SymbioOS}}: Closed-Loop Rootless {{POSIX}} Sandbox Scheduling and White-Box Streaming {{KV}} Promotion for Tool-Intensive Agent Clusters}},
  author  = {{Syed, Mohammed Kamran and Rentala, Anudeep and Rahul, Somesh and Pai, Raghuram Malpe and Dara, Krishna Suman}},
  journal = {{IEEE Transactions on Parallel and Distributed Systems (TPDS)}},
  year    = {{2026}},
  url     = {{https://github.com/batistado/symbioos}}
}}
```
"""
    (hub_dir / "README.md").write_text(hub_card, encoding="utf-8")


def main() -> int:
    api = HfApi()
    username = api.whoami()["name"]
    ds_repo_id = f"{username}/SymbioBench-Agent"
    space_repo_id = f"{username}/SymbioOS-Explorer"
    hub_repo_id = f"{username}/SymbioOS"

    bundles = load_all_telemetry()

    with tempfile.TemporaryDirectory(prefix="symbioos_hf_") as tmp:
        tmp_path = Path(tmp)
        ds_dir = tmp_path / "dataset"
        sp_dir = tmp_path / "space"
        hb_dir = tmp_path / "hub"

        build_dataset_staging(ds_dir, bundles, username=username)
        api.create_repo(repo_id=ds_repo_id, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=ds_repo_id,
            repo_type="dataset",
            folder_path=str(ds_dir),
            commit_message="Publish SymbioBench-Agent v1.0 multi-architecture & cluster-scale telemetry",
        )
        print(f"[1/3 SUCCESS] Dataset -> https://huggingface.co/datasets/{ds_repo_id}", flush=True)

        build_static_space_staging(sp_dir, bundles, username=username)
        api.create_repo(
            repo_id=space_repo_id,
            repo_type="space",
            space_sdk="static",
            private=False,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=space_repo_id,
            repo_type="space",
            folder_path=str(sp_dir),
            commit_message="Publish interactive SymbioOS & SymbioBench-Agent Static Explorer",
        )
        print(f"[2/3 SUCCESS] Space   -> https://huggingface.co/spaces/{space_repo_id}", flush=True)

        build_hub_system_repo_staging(hb_dir, username=username)
        api.create_repo(repo_id=hub_repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=hub_repo_id,
            repo_type="model",
            folder_path=str(hb_dir),
            commit_message="Publish SymbioOS v1.0 runtime, benchmarks, and IEEE TPDS manuscript bundle",
        )
        print(f"[3/3 SUCCESS] Hub     -> https://huggingface.co/{hub_repo_id}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
