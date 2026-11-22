#!/usr/bin/env python3
"""Validate Phase 2 Discrete-Event Tandem Simulator against Phase 0 Hardware Telemetry."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pilot"))
from phase2_tandem_simulator import SimTurn, run_tandem_sim  # noqa: E402


def pearson_r(x: list[float], y: list[float]) -> float:
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den > 1e-12 else 0.0


def main() -> None:
    p0 = json.loads(Path("/Users/mksyed/Documents/research/symbioos/results/symbioos_pilot.json").read_text())
    hw_map = {s["policy"]: s for s in p0["summary"]}
    policies = ["lru_cfs", "continuum_cfs", "unison_cfs", "symbioos_sched", "symbioos_full"]

    # Run simulator under matched Phase 0 Closed-Pool parameters:
    # K=12 concurrent slots, 20 episodes/seed, 2 turns/ep, C=4 cores, M_HBM=290 MB,
    # B_dma=160 MB/s, 36 KB/tok, Qwen2.5-3B MPS prefill rate ~3,100 tok/s.
    import random
    import phase2_tandem_simulator as sim_mod

    def make_phase0_matched_episodes(seed: int, num_episodes: int = 20):
        rng = random.Random(seed)
        base_toks = [1024, 1280, 1536, 1792]
        eps = []
        for sid in range(num_episodes):
            base_mb = (base_toks[(sid + rng.randint(0, 3)) % 4] * 36.0) / 1024.0
            turns = []
            for _ in range(2):
                r = rng.random()
                if r < 0.19:
                    turns.append(SimTurn("lint", "lint", rng.uniform(0.32, 0.44), 0.0, 140))
                elif r < 0.66:
                    turns.append(SimTurn("pytest", "pytest", rng.uniform(1.80, 2.25), 0.0, 1045))
                elif r < 0.85:
                    turns.append(SimTurn("compile", "build", rng.uniform(4.70, 5.40), 0.0, 1100))
                else:
                    turns.append(SimTurn("stalled", "pytest", rng.uniform(0.25, 0.34), rng.uniform(4.80, 5.40), 240))
            eps.append((base_mb, turns))
        return eps

    # Monkey-patch matched hardware parameters cleanly for calibration check
    orig_gen = sim_mod.generate_cluster_episodes
    sim_results = {}
    for pol in policies:
        p50s, hits, jcts = [], [], []
        for sd in [42, 137, 2026]:
            sim_mod.generate_cluster_episodes = lambda s, n, pareto_alpha=1.45, turns_per_ep=2: make_phase0_matched_episodes(s, n)
            # Run with 3B MPS scale: concurrency_n=12, oversub=3.0 (C=4), hbm_ratio=290/(12*58)=0.417
            res = sim_mod.run_tandem_sim(
                policy=pol,
                seed=sd,
                concurrency_n=12,
                total_episodes=20,
                oversub_ratio=3.0,
                pareto_alpha=1.45,
                delta_jit_s=0.55,
                dt=0.05,
                hbm_cap_mb_override=290.0,
                dma_bw_override=160.0,
                prefill_tok_s_override=3100.0,
                kv_mb_per_tok_override=36.0 / 1024.0,
            )
            p50s.append(res["ttft_p50_ms"])
            hits.append(res["hbm_hit_pct"])
            jcts.append(res["jct_mean_s"])
        sim_results[pol] = {
            "ttft_p50_ms": round(statistics.mean(p50s), 1),
            "hbm_hit_pct": round(statistics.mean(hits), 1),
            "jct_mean_s": round(statistics.mean(jcts), 2),
        }
    sim_mod.generate_cluster_episodes = orig_gen

    hw_p50s = [hw_map[p]["ttft_p50_ms"] for p in policies]
    sim_p50s = [sim_results[p]["ttft_p50_ms"] for p in policies]
    hw_hits = [hw_map[p]["hbm_hit_rate_pct"] for p in policies]
    sim_hits = [sim_results[p]["hbm_hit_pct"] for p in policies]
    print("Policy          | HW P50(ms) | Sim P50(ms) | HW Hit% | Sim Hit%")
    for p in policies:
        print(f"{p:15s} | {hw_map[p]['ttft_p50_ms']:10.1f} | {sim_results[p]['ttft_p50_ms']:11.1f} | {hw_map[p]['hbm_hit_rate_pct']:6.1f}% | {sim_results[p]['hbm_hit_pct']:7.1f}%")
    print(f"Pearson r (TTFT P50) : {pearson_r(hw_p50s, sim_p50s):.3f}")
    print(f"Pearson r (HBM Hit %): {pearson_r(hw_hits, sim_hits):.3f}")


if __name__ == "__main__":
    main()
