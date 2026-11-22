#!/usr/bin/env python3
"""SymbioOS Phase 2 Cluster-Scale Coupled Tandem-Queue Discrete-Event Simulator.

Scales evaluation to N = 32..256 concurrent agent sandboxes on a calibrated
multi-GPU serving node (Qwen2.5-32B / Llama-3-70B GQA KV economics: 160 KB/tok,
320-960 MB KV/session) to validate Theorem 1 (KV-Weighted Gittins Optimality under
Decreasing Hazard Rate Pareto tails) and Theorem 2 (Optimal-Stopping JIT Promotion
& Tandem Stability Frontier) across four controlled sweeps:

  - Sweep A: Concurrency scale-out N in {32, 64, 128, 256}
  - Sweep B: HBM capacity ratio M_HBM / M_active in {0.15, 0.25, 0.375, 0.50, 0.65} (N=128)
  - Sweep C: Pareto heavy-tail alpha in {1.15, 1.35, 1.65, 2.20} & CPU contention N/C
  - Sweep D: Theorem 2 JIT stopping boundary margin delta_jit_s in {0.0, 0.15, 0.35, 0.60, 1.20, 2.40}
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import random
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SimTurn:
    kind: str         # lint | pytest | compile | stalled
    cmd_hint: str     # lint | pytest | build
    cpu_req_s: float
    stall_s: float
    stdout_tokens: int


@dataclass
class SimAgent:
    aid: int
    base_kv_mb: float
    kv_mb: float
    turns: List[SimTurn]
    cur_turn: int = 0
    state: str = "running_tool"  # running_tool | waiting_gpu | completed
    in_hbm: bool = False
    was_evicted: bool = False
    jit_in_flight: bool = False
    jit_ready_t: float = 0.0
    jit_landed: bool = False
    cold_load_in_flight: bool = False
    cold_ready_t: float = 0.0
    had_hit_at_exit: bool = False

    launch_t: float = 0.0
    exit_t: float = 0.0
    last_gpu_t: float = 0.0
    cpu_done_s: float = 0.0
    obs_milestone_frac: float = 0.0
    last_output_t: float = 0.0
    prefilled_toks: int = 0
    micro_chunks: int = 0

    ep_start_t: float = 0.0
    turn_hbm_mb_s: float = 0.0


def sample_bounded_pareto(rng: random.Random, alpha: float, xm: float, xmax: float) -> float:
    u = rng.uniform(1e-6, 1.0 - 1e-6)
    num = 1.0 - u * (1.0 - (xm / xmax) ** alpha)
    return xm / (num ** (1.0 / alpha))


def generate_cluster_episodes(
    seed: int,
    num_episodes: int,
    pareto_alpha: float = 1.45,
    turns_per_ep: int = 3,
) -> List[Tuple[float, List[SimTurn]]]:
    rng = random.Random(seed)
    episodes: List[Tuple[float, List[SimTurn]]] = []
    for _ in range(num_episodes):
        # 2k to 5k initial context @ 160 KB/tok => 320 MB to 800 MB KV
        base_mb = rng.choice([320.0, 448.0, 576.0, 704.0, 800.0])
        turns: List[SimTurn] = []
        for _t in range(turns_per_ep):
            r = rng.random()
            if r < 0.20:
                turns.append(SimTurn("lint", "lint", rng.uniform(0.35, 0.65), 0.0, rng.randint(180, 320)))
            elif r < 0.65:
                # Medium interactive pytest with Pareto jitter around 1.8s-3.4s
                cpu_s = rng.uniform(1.6, 2.8) * sample_bounded_pareto(rng, max(pareto_alpha + 0.4, 1.5), 0.85, 2.2)
                turns.append(SimTurn("pytest", "pytest", min(cpu_s, 4.5), 0.0, rng.randint(1050, 1650)))
            elif r < 0.85:
                # Heavy Pareto-tailed compilation/build
                cpu_s = sample_bounded_pareto(rng, pareto_alpha, 4.8, 22.0)
                turns.append(SimTurn("compile", "build", cpu_s, 0.0, rng.randint(900, 1500)))
            else:
                # Ambiguous remote fixture / network stall (cmd_hint == "pytest")
                turns.append(SimTurn("stalled", "pytest", rng.uniform(0.30, 0.50), rng.uniform(6.5, 11.0), rng.randint(400, 650)))
        episodes.append((base_mb, turns))
    return episodes


def run_tandem_sim(
    policy: str,
    seed: int,
    concurrency_n: int = 128,
    total_episodes: int = 256,
    hbm_ratio: float = 0.375,
    oversub_ratio: float = 3.2,
    pareto_alpha: float = 1.45,
    delta_jit_s: float = 0.35,
    dt: float = 0.05,
    hbm_cap_mb_override: Optional[float] = None,
    dma_bw_override: Optional[float] = None,
    prefill_tok_s_override: Optional[float] = None,
    kv_mb_per_tok_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Time-stepped discrete-event simulation (dt = 50 ms) of a memory-coupled
    two-station tandem cluster.
    """
    episodes = generate_cluster_episodes(seed, total_episodes, pareto_alpha=pareto_alpha, turns_per_ep=3)
    cpu_cores = max(2, int(round(concurrency_n / oversub_ratio)))

    # Scale multi-GPU throughput & DMA bandwidth proportionally to cluster size (64 agents per GPU node)
    node_scale = concurrency_n / 64.0
    mean_ep_mb = 620.0
    hbm_cap_mb = hbm_cap_mb_override if hbm_cap_mb_override is not None else (concurrency_n * mean_ep_mb * hbm_ratio)
    dma_bw_mb_s = dma_bw_override if dma_bw_override is not None else (3200.0 * node_scale)
    prefill_tok_per_s = prefill_tok_s_override if prefill_tok_s_override is not None else (36_000.0 * node_scale)
    kv_mb_per_tok = kv_mb_per_tok_override if kv_mb_per_tok_override is not None else 0.160

    next_ep_idx = 0
    active: List[SimAgent] = []
    ttfts_ms: List[float] = []
    interactive_hits: List[bool] = []
    all_hits: List[bool] = []
    jcts_s: List[float] = []

    dead_waste_mb_s = 0.0
    stall_waste_mb_s = 0.0
    premature_jit_pin_mb_s = 0.0
    dma_free_at = 0.0
    gpu_free_at = 0.0
    micro_chunks_total = 0

    def hbm_used() -> float:
        return sum(a.kv_mb for a in active if a.in_hbm)

    def evict_one(exclude_id: int, now: float) -> bool:
        nonlocal dead_waste_mb_s
        cands = [a for a in active if a.in_hbm and a.aid != exclude_id and a.state == "running_tool"]
        if not cands:
            return False
        if policy == "lru_cfs":
            v = min(cands, key=lambda a: a.last_gpu_t)
        elif policy == "continuum_cfs":
            def c_key(a: SimAgent) -> Tuple[int, float]:
                hint = a.turns[a.cur_turn].cmd_hint
                ttl = 1.4 if hint == "lint" else (5.2 if hint == "pytest" else 0.6)
                return (1 if (now - a.launch_t) <= ttl else 0, a.last_gpu_t)
            v = min(cands, key=c_key)
        elif policy == "unison_cfs":
            def u_key(a: SimAgent) -> float:
                hint = a.turns[a.cur_turn].cmd_hint
                el = now - a.launch_t
                if hint == "lint":
                    haz = 2.5 if el < 1.8 else 0.4
                elif hint == "pytest":
                    haz = 0.85 if el <= 4.5 else 0.85 / (1.0 + ((el - 4.5) / 1.4) ** 2)
                else:
                    haz = 0.14 if el < 9.0 else 0.32
                return a.kv_mb * haz
            v = min(cands, key=u_key)
        else:
            def s_key(a: SimAgent) -> float:
                rttc, stalled = est_rttc(a, now)
                if stalled:
                    return -1e9
                return (a.kv_mb * (a.kv_mb / dma_bw_mb_s)) / max(rttc, 0.10)
            v = min(cands, key=s_key)

        v.in_hbm = False
        v.was_evicted = True
        v.jit_in_flight = False
        dead_waste_mb_s += v.turn_hbm_mb_s
        v.turn_hbm_mb_s = 0.0
        return True

    def ensure_space(need_mb: float, protect_id: int, now: float) -> bool:
        while hbm_used() + need_mb > hbm_cap_mb:
            if not evict_one(protect_id, now):
                return False
        return True

    def est_rttc(a: SimAgent, now: float) -> Tuple[float, bool]:
        t = a.turns[a.cur_turn]
        silence = now - a.last_output_t
        if a.obs_milestone_frac < 0.05 and silence >= 1.15:
            return (max(1.2, 8.0 - (now - a.launch_t)), True)
        if a.obs_milestone_frac >= 0.10 and a.cpu_done_s > 0.05:
            rem_cpu = max(0.05, a.cpu_done_s * (1.0 - a.obs_milestone_frac) / a.obs_milestone_frac)
        else:
            prior = {"lint": 0.50, "pytest": 2.40, "build": 8.00}[t.cmd_hint]
            rem_cpu = max(0.10, prior - a.cpu_done_s)
        share = 1.0 if a.in_hbm else max(0.22, cpu_cores / max(len(active), 1))
        return (rem_cpu / share, False)

    t_now = 0.0
    completed_eps = 0

    while completed_eps < total_episodes:
        # 1. Admit from backlog up to concurrency_n
        while len(active) < concurrency_n and next_ep_idx < total_episodes:
            base_mb, turns = episodes[next_ep_idx]
            if not ensure_space(base_mb, -1, t_now) and hbm_used() + base_mb > hbm_cap_mb:
                break
            ag = SimAgent(
                aid=next_ep_idx,
                base_kv_mb=base_mb,
                kv_mb=base_mb,
                turns=turns,
                in_hbm=True,
                launch_t=t_now,
                last_gpu_t=t_now,
                last_output_t=t_now,
                ep_start_t=t_now,
            )
            active.append(ag)
            next_ep_idx += 1

        # 2. Complete any finished JIT or Cold DMA transfers
        for a in active:
            if a.jit_in_flight and t_now >= a.jit_ready_t:
                a.jit_in_flight = False
                if a.state == "running_tool" and ensure_space(a.kv_mb, a.aid, t_now):
                    a.in_hbm = True
                    a.jit_landed = True
                    a.last_gpu_t = t_now
                elif a.state == "waiting_gpu" and ensure_space(a.kv_mb, a.aid, t_now):
                    a.in_hbm = True
                    a.last_gpu_t = t_now
            if a.cold_load_in_flight and t_now >= a.cold_ready_t:
                if ensure_space(a.kv_mb, a.aid, t_now):
                    a.cold_load_in_flight = False
                    a.in_hbm = True
                    a.last_gpu_t = t_now

        # 3. Tiering control: Continuum TTL expiry vs SymbioOS early demote & Theorem-2 JIT promote
        if policy == "continuum_cfs" and (hbm_used() / max(hbm_cap_mb, 1.0)) > 0.55:
            for a in active:
                if a.in_hbm and a.state == "running_tool":
                    hint = a.turns[a.cur_turn].cmd_hint
                    ttl = 1.4 if hint == "lint" else (5.2 if hint == "pytest" else 0.6)
                    if (t_now - a.launch_t) > ttl:
                        a.in_hbm = False
                        a.was_evicted = True
                        dead_waste_mb_s += a.turn_hbm_mb_s
                        a.turn_hbm_mb_s = 0.0

        elif policy in ("symbioos_sched", "symbioos_full"):
            util = hbm_used() / max(hbm_cap_mb, 1.0)
            for a in active:
                if a.in_hbm and a.state == "running_tool":
                    rttc, stalled = est_rttc(a, t_now)
                    hint = a.turns[a.cur_turn].cmd_hint
                    if stalled or (hint == "build" and rttc > 3.6 and util > 0.65):
                        a.in_hbm = False
                        a.was_evicted = True
                        dead_waste_mb_s += a.turn_hbm_mb_s
                        a.turn_hbm_mb_s = 0.0

            for a in active:
                if (
                    not a.in_hbm and not a.jit_in_flight and not a.cold_load_in_flight
                    and a.state == "running_tool" and a.obs_milestone_frac >= 0.50
                ):
                    rttc, stalled = est_rttc(a, t_now)
                    t_dma = a.kv_mb / dma_bw_mb_s
                    if (not stalled) and rttc <= (t_dma + delta_jit_s):
                        start_dma = max(t_now, dma_free_at)
                        if start_dma - t_now <= 0.40:
                            dma_free_at = start_dma + t_dma
                            a.jit_in_flight = True
                            a.jit_ready_t = dma_free_at

        # 4. OS Sandbox CPU Scheduling across running_tool agents
        running = [a for a in active if a.state == "running_tool"]
        runnable: List[SimAgent] = []
        for a in running:
            trn = a.turns[a.cur_turn]
            in_io_stall = (
                trn.kind == "stalled"
                and a.cpu_done_s >= 0.10
                and (t_now - a.launch_t) < (0.10 + trn.stall_s)
            )
            if a.in_hbm:
                bs = a.kv_mb * dt
                a.turn_hbm_mb_s += bs
                if in_io_stall:
                    stall_waste_mb_s += bs
                if a.jit_landed:
                    premature_jit_pin_mb_s += bs
            if not in_io_stall:
                runnable.append(a)

        if policy in ("lru_cfs", "continuum_cfs", "unison_cfs"):
            ordered = sorted(runnable, key=lambda a: (a.cpu_done_s, a.launch_t))
        else:
            def g_score(a: SimAgent) -> float:
                rttc, _ = est_rttc(a, t_now)
                wait_s = max(0.0, (t_now - a.launch_t) - a.cpu_done_s)
                val = (2.6 if (a.in_hbm or a.jit_in_flight) else 0.12) * a.kv_mb * (a.kv_mb / dma_bw_mb_s)
                return (val / max(rttc, 0.10)) + 0.35 * wait_s
            ordered = sorted(runnable, key=g_score, reverse=True)

        # Grant CPU core slices
        n_run = len(ordered)
        if n_run > 0:
            for idx, a in enumerate(ordered):
                if idx < cpu_cores:
                    a.cpu_done_s += dt
                    trn = a.turns[a.cur_turn]
                    true_frac = min(1.0, a.cpu_done_s / max(trn.cpu_req_s, 1e-3))
                    # Observable discrete [10%], [20%], ... milestones
                    new_mile = math.floor(true_frac * 10.0) / 10.0
                    if new_mile > a.obs_milestone_frac:
                        a.obs_milestone_frac = new_mile
                        a.last_output_t = t_now
                    if a.cpu_done_s >= trn.cpu_req_s:
                        a.state = "waiting_gpu"
                        a.exit_t = t_now + dt
                        a.had_hit_at_exit = a.in_hbm
                        if not a.in_hbm and not a.jit_in_flight and not a.cold_load_in_flight:
                            t_dma = a.kv_mb / dma_bw_mb_s
                            start_d = max(a.exit_t, dma_free_at)
                            dma_free_at = start_d + t_dma
                            a.cold_load_in_flight = True
                            a.cold_ready_t = dma_free_at
                        elif a.jit_in_flight:
                            # Promote in-flight JIT into mandatory cold completion
                            a.cold_load_in_flight = True
                            a.cold_ready_t = a.jit_ready_t
                            a.jit_in_flight = False

        # 5. GPU Serving (Work-Conserving Resume Queue + Idle-Slack Streaming Micro-Prefill)
        ready_gpu = sorted(
            [a for a in active if a.state == "waiting_gpu" and a.in_hbm],
            key=lambda x: x.exit_t,
        )
        while ready_gpu and gpu_free_at <= t_now + dt:
            a = ready_gpu.pop(0)
            trn = a.turns[a.cur_turn]
            rem_toks = max(16, trn.stdout_tokens - a.prefilled_toks) + 16
            pf_dur_s = rem_toks / prefill_tok_per_s + 0.018  # +18ms first-token decode
            start_g = max(t_now, a.exit_t, a.last_gpu_t if not a.had_hit_at_exit else a.exit_t, gpu_free_at)
            gpu_free_at = start_g + pf_dur_s
            ttft_ms = (gpu_free_at - a.exit_t) * 1000.0

            ttfts_ms.append(ttft_ms)
            all_hits.append(a.had_hit_at_exit)
            if trn.kind in ("lint", "pytest"):
                interactive_hits.append(a.had_hit_at_exit)

            if a.cur_turn + 1 < len(a.turns):
                grow_mb = rem_toks * kv_mb_per_tok
                ensure_space(grow_mb, a.aid, gpu_free_at)
                a.kv_mb += grow_mb
                a.cur_turn += 1
                a.state = "running_tool"
                a.was_evicted = False
                a.jit_landed = False
                a.launch_t = gpu_free_at
                a.last_gpu_t = gpu_free_at
                a.last_output_t = gpu_free_at
                a.cpu_done_s = 0.0
                a.obs_milestone_frac = 0.0
                a.prefilled_toks = 0
                a.micro_chunks = 0
                a.turn_hbm_mb_s = 0.0
            else:
                jcts_s.append(gpu_free_at - a.ep_start_t)
                active.remove(a)
                completed_eps += 1

        # Idle-slack 256-token streaming micro-prefill (SymbioOS-Full)
        if policy == "symbioos_full" and not ready_gpu and gpu_free_at <= t_now + dt:
            cands_m = []
            for a in active:
                if a.state == "running_tool" and a.in_hbm:
                    rttc, stalled = est_rttc(a, t_now)
                    if stalled:
                        continue
                    trn = a.turns[a.cur_turn]
                    emitted_toks = int(a.obs_milestone_frac * trn.stdout_tokens)
                    if emitted_toks - a.prefilled_toks >= 256:
                        cands_m.append((rttc, a))
            if cands_m:
                cands_m.sort(key=lambda x: x[0])
                _, ta = cands_m[0]
                grow_mb = 256 * kv_mb_per_tok
                if ensure_space(grow_mb, ta.aid, t_now):
                    chunk_s = 256.0 / prefill_tok_per_s
                    gpu_free_at = max(t_now, gpu_free_at) + chunk_s
                    ta.prefilled_toks += 256
                    ta.micro_chunks += 1
                    ta.kv_mb += grow_mb
                    ta.last_gpu_t = gpu_free_at
                    micro_chunks_total += 1

        t_now += dt

    arr = np.array(ttfts_ms, dtype=np.float64)
    makespan_min = t_now / 60.0
    return {
        "policy": policy,
        "concurrency_n": concurrency_n,
        "hbm_ratio": hbm_ratio,
        "oversub_ratio": oversub_ratio,
        "pareto_alpha": pareto_alpha,
        "delta_jit_s": delta_jit_s,
        "ttft_mean_ms": round(float(arr.mean()), 2),
        "ttft_p50_ms": round(float(np.percentile(arr, 50)), 2),
        "ttft_p95_ms": round(float(np.percentile(arr, 95)), 2),
        "ttft_p99_ms": round(float(np.percentile(arr, 99)), 2),
        "hbm_hit_pct": round(100.0 * sum(all_hits) / max(len(all_hits), 1), 2),
        "interactive_hit_pct": round(100.0 * sum(interactive_hits) / max(len(interactive_hits), 1), 2),
        "waste_gb_s": round((dead_waste_mb_s + stall_waste_mb_s) / 1024.0, 2),
        "premature_jit_pin_gb_s": round(premature_jit_pin_mb_s / 1024.0, 2),
        "jct_mean_s": round(statistics.mean(jcts_s), 2),
        "goodput_ep_per_min": round(total_episodes / max(makespan_min, 1e-6), 2),
        "micro_chunks": micro_chunks_total,
    }


def main() -> None:
    t0 = time.perf_counter()
    policies = ["lru_cfs", "continuum_cfs", "unison_cfs", "symbioos_sched", "symbioos_full"]

    print("[Sweep A] Concurrency Scale-Out N in {32, 64, 128, 256}...", flush=True)
    sweep_a = []
    for n in [32, 64, 128, 256]:
        for pol in policies:
            sweep_a.append(run_tandem_sim(pol, seed=42, concurrency_n=n, total_episodes=n * 2))

    print("[Sweep B] HBM Capacity Frontier M_HBM / M_active in {0.15, 0.25, 0.375, 0.50, 0.65} (N=128)...", flush=True)
    sweep_b = []
    for hr in [0.15, 0.25, 0.375, 0.50, 0.65]:
        for pol in policies:
            sweep_b.append(run_tandem_sim(pol, seed=42, concurrency_n=128, total_episodes=256, hbm_ratio=hr))

    print("[Sweep C] Theorem 1 Pareto Heavy-Tail Exponent alpha in {1.15, 1.35, 1.65, 2.20} (N=128)...", flush=True)
    sweep_c = []
    for alpha in [1.15, 1.35, 1.65, 2.20]:
        for pol in policies:
            sweep_c.append(run_tandem_sim(pol, seed=42, concurrency_n=128, total_episodes=256, pareto_alpha=alpha))

    print("[Sweep D] Theorem 2 Optimal-Stopping JIT Margin delta_jit_s in {0.0, 0.15, 0.35, 0.70, 1.40, 2.80}...", flush=True)
    sweep_d = []
    for dj in [0.0, 0.15, 0.35, 0.70, 1.40, 2.80]:
        sweep_d.append(run_tandem_sim("symbioos_full", seed=42, concurrency_n=128, total_episodes=256, delta_jit_s=dj))

    out_json = "results/phase2_scaleout_sweeps.json"
    out_md = "results/PHASE2_SCALEOUT_SUMMARY.md"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sweep_a_concurrency": sweep_a,
                "sweep_b_hbm_ratio": sweep_b,
                "sweep_c_pareto_tail": sweep_c,
                "sweep_d_jit_boundary": sweep_d,
            },
            f,
            indent=2,
        )

    lines = [
        "# SymbioOS Phase 2 Cluster Scale-Out (`N = 32..256`) & Theorem Verification Sweeps",
        "",
        "## 1. Sweep A — Cluster Concurrency Scale-Out (`N = 128` Highlight)",
        "",
        "| Policy (`N = 128`) | TTFT P50 (ms) | TTFT P95 (ms) | TTFT P99 (ms) | HBM Hit (%) | Interactive Hit (%) | HBM Waste (GB·s) | Mean JCT (s) | Goodput (ep/min) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in [x for x in sweep_a if x["concurrency_n"] == 128]:
        lines.append(
            f"| `{r['policy']}` | {r['ttft_p50_ms']:.1f} | {r['ttft_p95_ms']:.1f} | {r['ttft_p99_ms']:.1f} | "
            f"{r['hbm_hit_pct']:.1f}% | {r['interactive_hit_pct']:.1f}% | {r['waste_gb_s']:.1f} | "
            f"{r['jct_mean_s']:.2f} | {r['goodput_ep_per_min']:.1f} |"
        )

    lines.extend([
        "",
        "## 2. Sweep B — HBM Capacity Frontier (`N = 128`, `LRU+CFS` vs `SymbioOS-Full`)",
        "",
        "| HBM Ratio | `LRU+CFS` P50 (ms) | `SymbioOS-Full` P50 (ms) | `LRU+CFS` Hit (%) | `SymbioOS-Full` Hit (%) | `LRU+CFS` Goodput | `SymbioOS-Full` Goodput |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for hr in [0.15, 0.25, 0.375, 0.50, 0.65]:
        lru = next(x for x in sweep_b if x["hbm_ratio"] == hr and x["policy"] == "lru_cfs")
        sym = next(x for x in sweep_b if x["hbm_ratio"] == hr and x["policy"] == "symbioos_full")
        lines.append(
            f"| `{int(hr*100)}%` | {lru['ttft_p50_ms']:.1f} | **{sym['ttft_p50_ms']:.1f}** | "
            f"{lru['hbm_hit_pct']:.1f}% | **{sym['hbm_hit_pct']:.1f}%** | "
            f"{lru['goodput_ep_per_min']:.1f} | **{sym['goodput_ep_per_min']:.1f}** |"
        )

    lines.extend([
        "",
        "## 3. Sweep D — Theorem 2 JIT Optimal-Stopping Boundary (`N = 128`)",
        "",
        "| JIT Margin `delta_jit` (s) | TTFT P50 (ms) | TTFT P95 (ms) | HBM Hit (%) | Premature JIT Pin (GB·s) | Total HBM Waste (GB·s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for r in sweep_d:
        lines.append(
            f"| `{r['delta_jit_s']:.2f}s` | {r['ttft_p50_ms']:.1f} | {r['ttft_p95_ms']:.1f} | "
            f"{r['hbm_hit_pct']:.1f}% | {r['premature_jit_pin_gb_s']:.2f} | {r['waste_gb_s']:.1f} |"
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Completed all 4 cluster sweeps in {time.perf_counter()-t0:.2f}s -> {out_md}", flush=True)


if __name__ == "__main__":
    main()
