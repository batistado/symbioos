#!/usr/bin/env python3
"""Generate Publication-Quality IEEE TPDS Figures for SymbioOS.

Reads verified telemetry from:
  - results/symbioos_pilot.json               (Phase 0 Closed-Pool, 600 resumes)
  - results/symbioos_pilot_open_wave.json     (Phase 0 Burst Wave, 480 resumes)
  - results/phase1_multimodel_devtools.json   (Phase 1 Real pytest/clang++/git across 3 LLMs)
  - results/phase2_scaleout_sweeps.json       (Phase 2 N=32..256 Tandem-Queue Sweeps)

Outputs Type-42 TrueType embedded vector PDFs (IEEE PDF eXpress compliant) and
300-DPI PNG previews into `paper/figures/`:
  1. fig1_architecture_overview.{pdf,png}
  2. fig2_phase0_ttft_decomposition.{pdf,png}
  3. fig3_multimodel_devtools.{pdf,png}
  4. fig4_chunk_and_jit_ablation.{pdf,png}
  5. fig5_cluster_scaleout_sweeps.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "paper" / "figures"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

PALETTE = {
    "lru_cfs": "#7F8C8D",
    "continuum_cfs": "#D55E00",
    "unison_cfs": "#E69F00",
    "symbioos_sched": "#56B4E9",
    "symbioos_full": "#0072B2",
    "green": "#009E73",
    "purple": "#CC79A7",
    "navy": "#1B365D",
}

SHORT_LABELS = {
    "lru_cfs": "LRU",
    "continuum_cfs": "Cont.",
    "unison_cfs": "UNISON",
    "symbioos_sched": "Sym-S",
    "symbioos_full": "Sym-Full",
}

LEGEND_LABELS = {
    "lru_cfs": "LRU + CFS (vLLM Default)",
    "continuum_cfs": "Continuum (t=0 TTL + CFS)",
    "unison_cfs": "UNISON/SPEAR (t=0 Hazard + CFS)",
    "symbioos_sched": "SymbioOS-Sched (OS+KV Loop)",
    "symbioos_full": "SymbioOS-Full (+Stream stdout)",
}

POLICIES = ["lru_cfs", "continuum_cfs", "unison_cfs", "symbioos_sched", "symbioos_full"]


def save_fig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated {stem}.{{pdf,png}}")


def draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str,
             fc: str, ec: str = "#1B365D", tc: str = "#111827", fontsize: float = 7.4,
             weight: str = "normal", lw: float = 1.1) -> None:
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(box)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=tc, fontweight=weight, linespacing=1.22)


def make_fig1_architecture() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.15), gridspec_kw={"width_ratios": [1.0, 1.16], "wspace": 0.08})
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Panel (a): Open-Loop Collision Timeline
    ax1.text(0.5, 0.97, "(a) Open-Loop OS + GPU Serving Collision", ha="center", va="top",
             fontsize=9.8, fontweight="bold", color="#1B365D")

    draw_box(ax1, 0.02, 0.65, 0.96, 0.23, "", fc="#FEF2F2", ec="#DC2626", lw=1.0)
    ax1.text(0.04, 0.835, "Sandbox A: pytest (1.9s solo CPU, Hot KV in HBM)", fontsize=7.6, fontweight="bold", color="#991B1B")
    draw_box(ax1, 0.04, 0.68, 0.45, 0.11, "50% CFS CPU\n(3.8s > 3.7s TTL)", fc="#FCA5A5", ec="#B91C1C", fontsize=6.9, weight="bold")
    draw_box(ax1, 0.51, 0.68, 0.22, 0.11, "Cold SSD\nReload", fc="#D55E00", ec="#7C2D12", tc="white", fontsize=6.9, weight="bold")
    draw_box(ax1, 0.75, 0.68, 0.21, 0.11, "Full EOF\nPrefill", fc="#E69F00", ec="#78350F", tc="#111827", fontsize=6.9, weight="bold")

    ax1.annotate("Blind TTL Cliff Evicts Hot KV @ 3.7s", xy=(0.49, 0.68), xytext=(0.46, 0.535),
                 arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=1.2),
                 fontsize=7.2, fontweight="bold", color="#B91C1C", ha="center")

    draw_box(ax1, 0.02, 0.27, 0.96, 0.21, "", fc="#F1F5F9", ec="#64748B", lw=1.0)
    ax1.text(0.04, 0.435, "Sandbox B: clang++ -O3 (KV Already Evicted to SSD)", fontsize=7.6, fontweight="bold", color="#334155")
    draw_box(ax1, 0.04, 0.295, 0.58, 0.105, "50% CFS CPU Contends With Sandbox A", fc="#CBD5E1", ec="#475569", fontsize=6.9)
    draw_box(ax1, 0.64, 0.295, 0.32, 0.105, "GPU Slack Unused", fc="#E2E8F0", ec="#64748B", fontsize=6.9)

    draw_box(ax1, 0.02, 0.04, 0.96, 0.17,
             "Stalled I/O Sandbox C (cmd='pytest'): Pinned in HBM for full TTL\n"
             "2.7 GB*s dead HBM waste; 3.4x-4.8x worse Resume TTFT",
             fc="#FFFBEB", ec="#D97706", tc="#92400E", fontsize=6.6, weight="bold")

    # Panel (b): SymbioOS Closed-Loop Architecture
    ax2.text(0.5, 0.97, "(b) SymbioOS Rootless Closed-Loop Control Plane", ha="center", va="top",
             fontsize=9.8, fontweight="bold", color="#1B365D")

    draw_box(ax2, 0.02, 0.61, 0.44, 0.27,
             "POSIX Sandbox Pool (pgid)\npytest | clang++ | git\nstart_new_session=True\nNon-blocking stdout pipes",
             fc="#EFF6FF", ec="#0072B2", fontsize=7.2, weight="bold")

    draw_box(ax2, 0.54, 0.61, 0.44, 0.27,
             "Sub-Turn RTTC Filter (F_t)\n[XX%] milestones + byte rate\nUnstopped silence >= 1.25s\nPosterior E[R_i(t) | F_t]",
             fc="#ECFDF5", ec="#009E73", fontsize=7.2, weight="bold")

    draw_box(ax2, 0.02, 0.27, 0.44, 0.24,
             "KV-Weighted Gittins OS\nos.killpg(SIGSTOP/CONT)\nw_i = |KV_i|*C_rel / R_i(t)\nRescues Hot HBM Suites",
             fc="#E0F2FE", ec="#0284C7", fontsize=7.2, weight="bold")

    draw_box(ax2, 0.54, 0.27, 0.44, 0.24,
             "Tiered HBM <-> SSD DMA\n1.25s Early Stall Demote\nThm 2 JIT: R_i <= T_DMA + d\nUncached F_NOCACHE I/O",
             fc="#F0FDF4", ec="#15803D", fontsize=7.2, weight="bold")

    draw_box(ax2, 0.02, 0.04, 0.96, 0.16,
             "Idle-Slack 256-Tok Streaming stdout Micro-Prefill (1-Tok BPE Guard)\n"
             "7.98x faster EOF prefill; 2x faster Station-1 HBM queue drain",
             fc="#0072B2", ec="#1B365D", tc="white", fontsize=6.6, weight="bold")

    arrows = [
        ((0.46, 0.745), (0.54, 0.745), "#0072B2"),
        ((0.76, 0.61), (0.76, 0.51), "#009E73"),
        ((0.54, 0.64), (0.40, 0.51), "#0284C7"),
        ((0.24, 0.51), (0.24, 0.61), "#0284C7"),
        ((0.54, 0.39), (0.46, 0.39), "#15803D"),
        ((0.24, 0.27), (0.24, 0.20), "#0072B2"),
        ((0.76, 0.27), (0.76, 0.20), "#15803D"),
    ]
    for p1, p2, col in arrows:
        ax2.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=11, lw=1.25, color=col))

    save_fig(fig, "fig1_architecture_overview")


def make_fig2_phase0() -> None:
    closed = json.loads((RESULTS / "symbioos_pilot.json").read_text())["summary"]
    wave = json.loads((RESULTS / "symbioos_pilot_open_wave.json").read_text())["summary"]
    c_map = {s["policy"]: s for s in closed}
    w_map = {s["policy"]: s for s in wave}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.4, 2.65), gridspec_kw={"wspace": 0.34})
    x = np.arange(len(POLICIES))
    xlabels = [SHORT_LABELS[p] for p in POLICIES]

    # (a) Stacked Mean TTFT Breakdown (Closed-Pool)
    cold = [c_map[p]["ttft_breakdown_mean_ms"]["cold_reload_wait_ms"] for p in POLICIES]
    qwait = [c_map[p]["ttft_breakdown_mean_ms"]["gpu_queue_wait_ms"] for p in POLICIES]
    eof_pf = [c_map[p]["ttft_breakdown_mean_ms"]["eof_prefill_and_decode_ms"] for p in POLICIES]

    ax1.bar(x, cold, width=0.56, label="Cold NVMe Reload", color="#D55E00")
    ax1.bar(x, qwait, width=0.56, bottom=cold, label="GPU Queue Wait", color="#E69F00")
    ax1.bar(x, eof_pf, width=0.56, bottom=np.array(cold) + np.array(qwait), label="Post-EOF Prefill+1st", color="#0072B2")
    for idx, tot in enumerate(np.array(cold) + np.array(qwait) + np.array(eof_pf)):
        ax1.text(idx, tot + 75, f"{tot:.0f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels)
    ax1.set_ylabel("Mean Resume TTFT (ms)")
    ax1.set_ylim(0, 4150)
    ax1.set_title("(a) Mean TTFT Breakdown (K=12)")
    ax1.legend(loc="upper right", frameon=True, fontsize=6.8)

    # (b) Burst-Wave Tail Latency Ladder (P50 / P95)
    w_bw = 0.36
    p50s = [w_map[p]["ttft_p50_ms"] / 1000.0 for p in POLICIES]
    p95s = [w_map[p]["ttft_p95_ms"] / 1000.0 for p in POLICIES]
    ax2.bar(x - w_bw / 2, p50s, width=w_bw, label="P50 TTFT", color="#0072B2")
    ax2.bar(x + w_bw / 2, p95s, width=w_bw, label="P95 TTFT", color="#CC79A7")
    for idx in range(len(POLICIES)):
        ax2.text(idx - w_bw / 2, p50s[idx] + 0.25, f"{p50s[idx]:.1f}", ha="center", fontsize=6.6)
        ax2.text(idx + w_bw / 2, p95s[idx] + 0.25, f"{p95s[idx]:.1f}", ha="center", fontsize=6.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels)
    ax2.set_ylabel("Resume TTFT (seconds)")
    ax2.set_ylim(0, 13.6)
    ax2.set_title("(b) Burst-Wave P50 & P95 (N=16)")
    ax2.legend(loc="upper left", frameon=True, fontsize=6.9)

    # (c) HBM Hit Rate vs HBM Waste Synergy (Burst Wave)
    hits = [w_map[p]["hbm_hit_rate_pct"] for p in POLICIES]
    ihits = [w_map[p]["interactive_hit_rate_pct"] for p in POLICIES]
    wastes = [w_map[p]["total_waste_mb_s_per_seed"] / 1000.0 for p in POLICIES]

    ax3.bar(x - w_bw / 2, hits, width=w_bw, label="All-Turn Hit %", color="#56B4E9")
    ax3.bar(x + w_bw / 2, ihits, width=w_bw, label="Interactive Hit %", color="#009E73")
    ax3.set_xticks(x)
    ax3.set_xticklabels(xlabels)
    ax3.set_ylabel("HBM KV Hit Rate (%)")
    ax3.set_ylim(0, 118)
    ax3.set_title("(c) HBM Hit % & Waste")

    ax3_r = ax3.twinx()
    ax3_r.spines["top"].set_visible(False)
    ax3_r.grid(False)
    ax3_r.plot(x, wastes, color="#D55E00", marker="o", lw=1.7, markersize=4.8, label="Waste (GB*s)")
    ax3_r.set_ylabel("HBM Waste (GB*s / seed)", color="#D55E00", fontsize=8.0)
    ax3_r.tick_params(axis="y", labelcolor="#D55E00", labelsize=7.5)
    ax3_r.set_ylim(0, 3.6)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, fontsize=6.5)

    save_fig(fig, "fig2_phase0_ttft_decomposition")


def make_fig3_multimodel() -> None:
    p1 = json.loads((RESULTS / "phase1_multimodel_devtools.json").read_text())["multimodel_summary"]
    models = [
        ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B\n(36 KB/tok)"),
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B\n(56 KB/tok)"),
        ("allenai/OLMoE-1B-7B-0924-Instruct", "OLMoE-1B-7B\n(128 KB/tok)"),
    ]
    lookup = {(s["model_id"], s["policy"]): s for s in p1}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.4, 2.70), gridspec_kw={"wspace": 0.30})
    mx = np.arange(len(models))
    bw = 0.145
    offsets = np.linspace(-2 * bw, 2 * bw, len(POLICIES))

    for pidx, pol in enumerate(POLICIES):
        col = PALETTE[pol]
        lbl = LEGEND_LABELS[pol]
        p50_vals = [lookup[(m[0], pol)]["ttft_p50_ms"] / 1000.0 for m in models]
        ihit_vals = [lookup[(m[0], pol)]["interactive_hit_rate_pct"] for m in models]
        slo_vals = [lookup[(m[0], pol)]["slo_goodput_le_600ms_per_min"] for m in models]

        ax1.bar(mx + offsets[pidx], p50_vals, width=bw, color=col, label=lbl)
        ax2.bar(mx + offsets[pidx], ihit_vals, width=bw, color=col, label=lbl)
        ax3.bar(mx + offsets[pidx], slo_vals, width=bw, color=col, label=lbl)

    ax1.set_xticks(mx)
    ax1.set_xticklabels([m[1] for m in models])
    ax1.set_ylabel("Resume TTFT P50 (seconds)")
    ax1.set_title("(a) Median TTFT Across Models")
    ax1.set_ylim(0, 4.9)

    ax2.set_xticks(mx)
    ax2.set_xticklabels([m[1] for m in models])
    ax2.set_ylabel("Interactive HBM Hit Rate (%)")
    ax2.set_title("(b) pytest/lint HBM Hit Rate")
    ax2.set_ylim(0, 108)

    ax3.set_xticks(mx)
    ax3.set_xticklabels([m[1] for m in models])
    ax3.set_ylabel("SLO Goodput (<=600ms / min)")
    ax3.set_title("(c) Strict-SLO Resume Goodput")
    ax3.set_ylim(0, 103)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True,
               bbox_to_anchor=(0.5, 1.10), fontsize=7.2)
    save_fig(fig, "fig3_multimodel_devtools")


def make_fig4_chunk_and_jit() -> None:
    chunks = json.loads((RESULTS / "phase1_multimodel_devtools.json").read_text())["chunk_ablation_summary"]
    jit = json.loads((RESULTS / "phase2_scaleout_sweeps.json").read_text())["sweep_d_jit_boundary"]

    order_map = {128: 0, 256: 1, 512: 2, 0: 3}
    chunks_sorted = sorted(chunks, key=lambda c: order_map[c["chunk_tokens"]])
    c_labels = ["128 tok", "256 tok\n(Default)", "512 tok", "inf\n(No Stream)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.60), gridspec_kw={"wspace": 0.30})
    cx = np.arange(len(chunks_sorted))
    bw = 0.25
    p50 = [c["ttft_p50_ms"] for c in chunks_sorted]
    p95 = [c["ttft_p95_ms"] for c in chunks_sorted]
    eof = [c["eof_prefill_mean_ms"] for c in chunks_sorted]

    ax1.bar(cx - bw, p50, width=bw, color="#0072B2", label="TTFT P50 (ms)")
    ax1.bar(cx, p95, width=bw, color="#56B4E9", label="TTFT P95 (ms)")
    ax1.bar(cx + bw, eof, width=bw, color="#009E73", label="Mean EOF Prefill (ms)")
    for i in range(len(cx)):
        ax1.text(i - bw, p50[i] + 22, f"{p50[i]:.0f}", ha="center", fontsize=6.8, fontweight="bold")
        ax1.text(i, p95[i] + 22, f"{p95[i]:.0f}", ha="center", fontsize=6.8)
    ax1.set_xticks(cx)
    ax1.set_xticklabels(c_labels)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_ylim(0, 1450)
    ax1.set_title("(a) Micro-Prefill Chunk Size (Qwen2.5-3B MPS)")
    ax1.legend(loc="upper left", ncol=2, frameon=True, fontsize=6.8)

    deltas = [r["delta_jit_s"] for r in jit]
    jp95 = [r["ttft_p95_ms"] for r in jit]
    jpin = [r["premature_jit_pin_gb_s"] for r in jit]

    ax2.plot(deltas, jp95, color="#0072B2", marker="o", lw=1.8, label="Resume TTFT P95 (ms)")
    ax2.axvspan(0.15, 0.45, color="#009E73", alpha=0.15, label="Thm 2 Sweet Spot (0.15-0.45s)")
    ax2.set_xlabel("JIT Promotion Margin delta_jit (seconds)")
    ax2.set_ylabel("Resume TTFT P95 (ms)", color="#0072B2")
    ax2.tick_params(axis="y", labelcolor="#0072B2")
    ax2.set_ylim(130, 395)
    ax2.set_title("(b) Theorem 2 Optimal-Stopping JIT Trade-Off")

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.grid(False)
    ax2_r.plot(deltas, jpin, color="#D55E00", marker="s", linestyle="--", lw=1.7, label="Premature Pin (GB*s)")
    ax2_r.set_ylabel("Premature HBM Pin (GB*s)", color="#D55E00")
    ax2_r.tick_params(axis="y", labelcolor="#D55E00")
    ax2_r.set_ylim(-5, 195)

    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2_r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, loc="upper right", frameon=True, fontsize=6.7)

    save_fig(fig, "fig4_chunk_and_jit_ablation")


def make_fig5_scaleout() -> None:
    p2 = json.loads((RESULTS / "phase2_scaleout_sweeps.json").read_text())
    sa = p2["sweep_a_concurrency"]
    sb = p2["sweep_b_hbm_ratio"]
    sc = p2["sweep_c_pareto_tail"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.4, 2.70), gridspec_kw={"wspace": 0.34})
    markers = {"lru_cfs": "o", "continuum_cfs": "^", "unison_cfs": "d", "symbioos_sched": "s", "symbioos_full": "*"}

    ns = [32, 64, 128, 256]
    for pol in POLICIES:
        vals = [next(r["ttft_p99_ms"] for r in sa if r["concurrency_n"] == n and r["policy"] == pol) for n in ns]
        ax1.plot(ns, vals, marker=markers[pol], color=PALETTE[pol], lw=1.7,
                 markersize=5.8 if pol == "symbioos_full" else 4.4, label=LEGEND_LABELS[pol])
    ax1.set_xticks(ns)
    ax1.set_xlabel("Concurrent Sandboxes (N)")
    ax1.set_ylabel("Resume TTFT P99 (ms)")
    ax1.set_title("(a) Scale-Out Tail Latency")

    hrs = [0.15, 0.25, 0.375, 0.50, 0.65]
    hr_pct = [int(round(h * 100)) for h in hrs]
    for pol in ["lru_cfs", "continuum_cfs", "unison_cfs", "symbioos_full"]:
        vals = [next(r["ttft_p50_ms"] for r in sb if abs(r["hbm_ratio"] - h) < 1e-4 and r["policy"] == pol) for h in hrs]
        ax2.plot(hr_pct, vals, marker=markers[pol], color=PALETTE[pol], lw=1.7,
                 markersize=5.8 if pol == "symbioos_full" else 4.4)
    ax2.set_yscale("log")
    ax2.set_xticks(hr_pct)
    ax2.set_xticklabels([f"{p}%" for p in hr_pct])
    ax2.set_xlabel("HBM Capacity Ratio (%)")
    ax2.set_ylabel("Resume TTFT P50 (ms, log)")
    ax2.set_title("(b) HBM Frontier (5.8x @ 15%)")

    alphas = [1.15, 1.35, 1.65, 2.20]
    for pol in POLICIES:
        vals = [next(r["waste_gb_s"] for r in sc if abs(r["pareto_alpha"] - a) < 1e-4 and r["policy"] == pol) for a in alphas]
        ax3.plot(alphas, vals, marker=markers[pol], color=PALETTE[pol], lw=1.7,
                 markersize=5.8 if pol == "symbioos_full" else 4.4)
    ax3.set_xticks(alphas)
    ax3.set_xlabel("Pareto Tail Exponent alpha")
    ax3.set_ylabel("HBM Waste (GB*s)")
    ax3.set_title("(c) Heavy-Tail HBM Waste")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True,
               bbox_to_anchor=(0.5, 1.10), fontsize=7.2)
    save_fig(fig, "fig5_cluster_scaleout_sweeps")


def main() -> None:
    print("Generating SymbioOS IEEE TPDS vector PDF + 300-DPI PNG figures...")
    make_fig1_architecture()
    make_fig2_phase0()
    make_fig3_multimodel()
    make_fig4_chunk_and_jit()
    make_fig5_scaleout()
    print("All 5 figures generated successfully.")


if __name__ == "__main__":
    main()
