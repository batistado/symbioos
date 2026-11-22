#!/usr/bin/env python3
"""SymbioOS Phase 1 Local Workstation Benchmark:
Real Developer-Tool Process Groups (pytest 9.2 + clang++ -O3 + git) Across
Three Distinct LLM Architectures (Qwen2.5-3B, Qwen2.5-7B, OLMoE-1B-7B) +
Streaming stdout Micro-Prefill Chunk-Size Ablation.

All child sandboxes launch in isolated POSIX sessions (`start_new_session=True`)
and are scheduled as complete process trees via `os.killpg(pgid, SIGSTOP/SIGCONT)`.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import fcntl
import gc
import json
import os
import queue
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROGRESS_RE = re.compile(r"\[\s*(\d{1,3})%\]")
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
BUCKETS = (64, 128, 192, 256, 512)


@dataclass
class TurnSpec:
    turn_idx: int
    kind: str          # quick_lint | pytest_suite | heavy_compile | stalled_io
    cmd_hint: str      # lint | pytest | build (stalled_io has cmd_hint="pytest"!)
    pytest_iters: int
    clang_tus: int
    stall_sec: float


@dataclass
class EpisodeSpec:
    session_id: int
    arrival_sec: float
    base_prompt_tokens: int
    turns: List[TurnSpec]


def generate_devtool_workload(seed: int, num_episodes: int = 16, turns_per_session: int = 2) -> List[EpisodeSpec]:
    """Generate deterministic multi-turn real-devtool workload."""
    rng = random.Random(seed)
    base_lengths = [1024, 1280, 1536, 1792]

    total_turns = num_episodes * turns_per_session
    n_quick = max(1, round(0.19 * total_turns))     # 6 of 32
    n_pytest = max(1, round(0.47 * total_turns))    # 15 of 32
    n_compile = max(1, round(0.19 * total_turns))   # 6 of 32
    n_stall = max(1, total_turns - n_quick - n_pytest - n_compile)  # 5 of 32

    pool = (
        ["quick_lint"] * n_quick
        + ["pytest_suite"] * n_pytest
        + ["heavy_compile"] * n_compile
        + ["stalled_io"] * n_stall
    )
    rng.shuffle(pool)

    episodes: List[EpisodeSpec] = []
    idx = 0
    for sid in range(num_episodes):
        t_arr = round(sid * 0.14, 3)
        base_tok = base_lengths[(sid + rng.randint(0, 3)) % len(base_lengths)]
        turns: List[TurnSpec] = []
        for tidx in range(turns_per_session):
            kind = pool[idx]
            idx += 1
            if kind == "quick_lint":
                turns.append(TurnSpec(tidx, kind, "lint", pytest_iters=40_000, clang_tus=1, stall_sec=0.0))
            elif kind == "pytest_suite":
                p_iters = rng.randint(200_000, 250_000)  # ~1.65-2.05s real pytest -v
                turns.append(TurnSpec(tidx, kind, "pytest", pytest_iters=p_iters, clang_tus=1, stall_sec=0.0))
            elif kind == "heavy_compile":
                tus = rng.choice([8, 9])                 # ~3.7-4.3s real clang++ -O3 -c
                turns.append(TurnSpec(tidx, kind, "build", pytest_iters=40_000, clang_tus=tus, stall_sec=0.0))
            else:
                st_s = round(rng.uniform(4.40, 4.90), 3)
                turns.append(TurnSpec(tidx, kind, "pytest", pytest_iters=40_000, clang_tus=1, stall_sec=st_s))
        episodes.append(EpisodeSpec(sid, t_arr, base_tok, turns))
    return episodes


def cache_kv_bytes(cache: DynamicCache) -> int:
    total = 0
    for layer in cache.layers:
        if layer.keys is not None and layer.values is not None:
            total += layer.keys.numel() * layer.keys.element_size()
            total += layer.values.numel() * layer.values.element_size()
    return total


def clone_mps_cache(src: DynamicCache) -> DynamicCache:
    dst = DynamicCache()
    for idx, layer in enumerate(src.layers):
        dst.update(layer.keys.clone(), layer.values.clone(), idx)
    return dst


def cache_to_cpu_tuples(cache: DynamicCache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(layer.keys.detach().cpu(), layer.values.detach().cpu()) for layer in cache.layers]


def cpu_tuples_to_mps_cache(tuples: List[Tuple[torch.Tensor, torch.Tensor]], device: str = "mps") -> DynamicCache:
    dst = DynamicCache()
    for idx, (k_cpu, v_cpu) in enumerate(tuples):
        dst.update(k_cpu.to(device), v_cpu.to(device), idx)
    return dst


def forward_chunked_mps(
    model: Any,
    token_ids: List[int],
    cache: DynamicCache,
    max_step: int = 256,
    device: str = "mps",
) -> torch.Tensor:
    assert len(token_ids) > 0
    pos = 0
    n = len(token_ids)
    last_logits: Optional[torch.Tensor] = None
    with torch.no_grad():
        while pos < n:
            rem = n - pos
            if rem >= max_step:
                step_ids = token_ids[pos : pos + max_step]
                inp = torch.tensor([step_ids], dtype=torch.long, device=device)
                out = model(input_ids=inp, past_key_values=cache, use_cache=True)
                last_logits = out.logits[:, -1, :]
                pos += max_step
            elif rem in (1, 2):
                step_ids = token_ids[pos : pos + rem]
                inp = torch.tensor([step_ids], dtype=torch.long, device=device)
                out = model(input_ids=inp, past_key_values=cache, use_cache=True)
                last_logits = out.logits[:, -1, :]
                pos += rem
            else:
                bucket = next(b for b in BUCKETS if b >= rem)
                pad_len = bucket - rem
                step_ids = token_ids[pos : pos + rem] + ([1] * pad_len)
                inp = torch.tensor([step_ids], dtype=torch.long, device=device)
                out = model(input_ids=inp, past_key_values=cache, use_cache=True)
                last_logits = out.logits[:, rem - 1, :].clone()
                if pad_len > 0:
                    cache.crop(-pad_len)
                pos += rem
    assert last_logits is not None
    return last_logits


def warmup_mps_buckets(model: Any, device: str = "mps") -> None:
    c = DynamicCache()
    with torch.no_grad():
        for shape_len in (512, 256, 256, 192, 128, 64, 2, 1):
            inp = torch.randint(10, 1000, (1, shape_len), dtype=torch.long, device=device)
            _ = model(input_ids=inp, past_key_values=c, use_cache=True)
    torch.mps.synchronize()
    del c
    gc.collect()
    torch.mps.empty_cache()


def build_base_prompt_templates(
    model: Any,
    tokenizer: AutoTokenizer,
    lengths: List[int],
    device: str = "mps",
) -> Dict[int, DynamicCache]:
    header = (
        "System: Autonomous software engineering agent operating inside an isolated POSIX "
        "process-group sandbox. Analyze pytest and clang++ output and propose minimal fixes.\n"
        "Repository snapshot (symbioos/runtime/scheduler.py):\n"
    )
    filler = (
        "def compute_gittins_index(kv_bytes: int, rttc_sec: float, bw_bytes_sec: float) -> float:\n"
        "    reload_cost = kv_bytes / max(bw_bytes_sec, 1.0)\n"
        "    return reload_cost / max(rttc_sec, 0.05)\n\n"
    )
    raw_ids = tokenizer.encode(header + (filler * 140), add_special_tokens=False)
    suffix_ids = tokenizer.encode("\nExecuting sandbox verification tool:\nTool stdout:\n", add_special_tokens=False)

    templates: Dict[int, DynamicCache] = {}
    for target_len in lengths:
        seq_ids = raw_ids[: target_len - len(suffix_ids)] + suffix_ids
        assert len(seq_ids) == target_len
        c = DynamicCache()
        _ = forward_chunked_mps(model, seq_ids, c, max_step=256, device=device)
        torch.mps.synchronize()
        templates[target_len] = c
    return templates


@dataclass
class DMARequest:
    op: str
    req_id: str
    session_id: int
    turn_idx: int
    file_path: str
    kv_mb: float
    cpu_tuples: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    is_jit: bool = False


@dataclass
class DMACompletion:
    req_id: str
    session_id: int
    turn_idx: int
    kv_mb: float
    cpu_tuples: List[Tuple[torch.Tensor, torch.Tensor]]
    is_jit: bool


class NVMeDMAEngine:
    def __init__(self, nvme_bw_mb_s: float) -> None:
        self.nvme_bw_mb_s = nvme_bw_mb_s
        self.save_q: "queue.Queue[Optional[DMARequest]]" = queue.Queue()
        self.load_q: "queue.Queue[Optional[DMARequest]]" = queue.Queue()
        self.done_q: "queue.Queue[DMACompletion]" = queue.Queue()
        self._staging: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._on_disk: set[str] = set()
        self._lock = threading.Lock()
        self._save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self._load_thread = threading.Thread(target=self._load_loop, daemon=True)
        self._save_thread.start()
        self._load_thread.start()

    def register_save(
        self, req_id: str, session_id: int, turn_idx: int, file_path: str, kv_mb: float,
        cpu_tuples: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        with self._lock:
            self._staging[req_id] = cpu_tuples
        self.save_q.put(DMARequest("save", req_id, session_id, turn_idx, file_path, kv_mb, cpu_tuples))

    def request_load(
        self, req_id: str, session_id: int, turn_idx: int, file_path: str, kv_mb: float, is_jit: bool = False
    ) -> None:
        self.load_q.put(DMARequest("load", req_id, session_id, turn_idx, file_path, kv_mb, is_jit=is_jit))

    def _nocache(self, path: str) -> None:
        try:
            fd = os.open(path, os.O_RDONLY, 0o644)
            try:
                fcntl.fcntl(fd, F_NOCACHE, 1)
            finally:
                os.close(fd)
        except OSError:
            pass

    def _save_loop(self) -> None:
        while True:
            req = self.save_q.get()
            if req is None:
                break
            assert req.cpu_tuples is not None
            torch.save(req.cpu_tuples, req.file_path)
            self._nocache(req.file_path)
            with self._lock:
                self._on_disk.add(req.req_id)
                self._staging.pop(req.req_id, None)

    def _load_loop(self) -> None:
        while True:
            req = self.load_q.get()
            if req is None:
                break
            t0 = time.perf_counter()
            with self._lock:
                staged = self._staging.get(req.req_id)
                is_disk = req.req_id in self._on_disk
            if staged is not None and not is_disk:
                loaded = [(k.clone(), v.clone()) for k, v in staged]
            else:
                self._nocache(req.file_path)
                loaded = torch.load(req.file_path, weights_only=True)
            target_s = req.kv_mb / max(self.nvme_bw_mb_s, 1.0)
            dt = time.perf_counter() - t0
            if dt < target_s:
                time.sleep(target_s - dt)
            self.done_q.put(DMACompletion(req.req_id, req.session_id, req.turn_idx, req.kv_mb, loaded, req.is_jit))

    def shutdown(self) -> None:
        self.save_q.put(None)
        self.load_q.put(None)
        self._save_thread.join(timeout=3.0)
        self._load_thread.join(timeout=3.0)
        with self._lock:
            self._staging.clear()


@dataclass
class SessionRuntime:
    spec: EpisodeSpec
    state: str = "not_arrived"
    cur_turn_idx: int = 0
    in_hbm: bool = False
    cache: Optional[DynamicCache] = None
    kv_bytes: int = 0
    disk_req_id: str = ""
    disk_path: str = ""
    load_in_flight: bool = False
    load_is_jit: bool = False
    was_evicted_this_turn: bool = False
    jit_promoted_this_turn: bool = False
    last_gpu_touch_ts: float = 0.0
    had_hit_at_exit: Optional[bool] = None
    ready_in_hbm_ts: float = 0.0

    proc: Optional[subprocess.Popen] = None
    pgid: int = -1
    stdout_fd: int = -1
    stall_w_fd: int = -1
    stall_release_ts: float = 0.0
    stall_released: bool = False
    is_sigstopped: bool = False

    tool_launch_ts: float = 0.0
    tool_exit_ts: float = 0.0
    last_stdout_ts: float = 0.0
    unstopped_silence_s: float = 0.0
    stdout_buffer: str = ""
    stdout_bytes: int = 0
    progress_frac: float = 0.0
    cpu_granted_s: float = 0.0
    prefilled_stdout_tokens: int = 0
    micro_chunks_this_turn: int = 0

    rttc_est_s: float = 2.0
    is_stalled_est: bool = False
    episode_start_ts: float = 0.0
    episode_end_ts: float = 0.0
    turn_hbm_byte_s_accum: float = 0.0


def unison_hazard(cmd_hint: str, elapsed_s: float) -> float:
    if cmd_hint == "lint":
        return 2.8 if elapsed_s < 1.2 else 0.45
    if cmd_hint == "pytest":
        if elapsed_s <= 3.3:
            return 0.90
        return 0.90 / (1.0 + ((elapsed_s - 3.3) / 1.0) ** 2)
    return 0.15 if elapsed_s < 6.5 else 0.38


def continuum_ttl(cmd_hint: str) -> float:
    if cmd_hint == "lint":
        return 1.0
    if cmd_hint == "pytest":
        return 3.70  # ~2.0x solo real pytest runtime (~1.85s)
    return 0.45


def update_rttc_filter(s: SessionRuntime, now: float, num_cores: int, runnable_cnt: int) -> None:
    turn = s.spec.turns[s.cur_turn_idx]
    # Unstopped silence only accrues when container is NOT SIGSTOPped by OS scheduler!
    if s.progress_frac < 0.05 and s.unstopped_silence_s >= 1.25:
        s.is_stalled_est = True
        s.rttc_est_s = max(1.0, 4.70 - (now - s.tool_launch_ts))
        return
    s.is_stalled_est = False
    if s.progress_frac >= 0.99:
        rem_cpu = 0.04
    elif s.progress_frac >= 0.10 and s.cpu_granted_s > 0.03:
        rem_cpu = max(0.05, s.cpu_granted_s * (1.0 - s.progress_frac) / s.progress_frac)
    else:
        prior_cpu = {"lint": 0.25, "pytest": 1.85, "build": 4.00}[turn.cmd_hint]
        rem_cpu = max(0.08, prior_cpu - s.cpu_granted_s)
    share = 1.0 if s.in_hbm else min(1.0, max(0.25, num_cores / max(runnable_cnt, 1)))
    s.rttc_est_s = rem_cpu / share


def killpg_safe(pgid: int, sig: signal.Signals) -> None:
    if pgid <= 0:
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def run_devtool_policy(
    model_id: str,
    policy: str,
    seed: int,
    episodes: List[EpisodeSpec],
    model: Any,
    tokenizer: AutoTokenizer,
    base_templates: Dict[int, DynamicCache],
    kv_bytes_per_tok: int,
    max_concurrency: int,
    hbm_cap_mb: float,
    cpu_cores: int,
    nvme_bw_mb_s: float,
    micro_chunk_tokens: int,
    device: str = "mps",
) -> Dict[str, Any]:
    hbm_cap_bytes = int(hbm_cap_mb * 1024 * 1024)
    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_devtool_worker.py")
    scratch_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".devtool_scratch")
    os.makedirs(scratch_base, exist_ok=True)
    run_scratch = tempfile.mkdtemp(prefix=f"run_{policy}_s{seed}_", dir=scratch_base)

    turn_close_ids = tokenizer.encode("\nAssistant analysis:\n", add_special_tokens=False)
    next_open_ids = tokenizer.encode("\nNext sandbox tool stdout:\n", add_special_tokens=False)
    admit_kick_ids = tokenizer.encode("\nDispatching tool.\n", add_special_tokens=False)[:2]

    sessions = [SessionRuntime(spec=ep) for ep in episodes]
    state_lock = threading.Lock()
    stop_sched_event = threading.Event()

    metrics_accum = {
        "total_hbm_byte_s": 0.0,
        "dead_evicted_waste_byte_s": 0.0,
        "stalled_holding_byte_s": 0.0,
        "early_demotions": 0,
        "ttl_expirations": 0,
        "jit_promotions_started": 0,
        "jit_promotions_landed": 0,
        "reactive_evictions": 0,
    }

    evict_tmpdir = tempfile.TemporaryDirectory(prefix=f"kv_{policy}_s{seed}_")
    dma_engine = NVMeDMAEngine(nvme_bw_mb_s=nvme_bw_mb_s)
    turn_records: List[Dict[str, Any]] = []
    min_session_mb = (1024 * kv_bytes_per_tok) / (1024.0 * 1024.0)

    def current_hbm_bytes() -> int:
        return sum(s.kv_bytes for s in sessions if s.in_hbm)

    def evict_to_disk(victim: SessionRuntime, reason: str) -> None:
        if not victim.in_hbm or victim.cache is None:
            return
        kv_mb = victim.kv_bytes / (1024.0 * 1024.0)
        req_id = f"s{victim.spec.session_id}_t{victim.cur_turn_idx}_{int(time.perf_counter()*10000)}"
        fpath = os.path.join(evict_tmpdir.name, f"{req_id}.pt")
        cpu_tuples = cache_to_cpu_tuples(victim.cache)
        with state_lock:
            victim.in_hbm = False
            victim.cache = None
            victim.disk_req_id = req_id
            victim.disk_path = fpath
            victim.was_evicted_this_turn = True
            victim.load_in_flight = False
            victim.load_is_jit = False
            metrics_accum["dead_evicted_waste_byte_s"] += victim.turn_hbm_byte_s_accum
            victim.turn_hbm_byte_s_accum = 0.0
            if reason == "early_demote":
                metrics_accum["early_demotions"] += 1
            elif reason == "ttl_expire":
                metrics_accum["ttl_expirations"] += 1
            else:
                metrics_accum["reactive_evictions"] += 1
        dma_engine.register_save(req_id, victim.spec.session_id, victim.cur_turn_idx, fpath, kv_mb, cpu_tuples)

    def pick_victim(exclude_sid: int, now: float) -> Optional[SessionRuntime]:
        cands = [s for s in sessions if s.in_hbm and s.spec.session_id != exclude_sid and s.state == "running_tool"]
        if not cands:
            return None
        if policy == "lru_cfs":
            return min(cands, key=lambda s: s.last_gpu_touch_ts)
        if policy == "continuum_cfs":
            return min(
                cands,
                key=lambda s: (
                    1 if (now - s.tool_launch_ts) <= continuum_ttl(s.spec.turns[s.cur_turn_idx].cmd_hint) else 0,
                    s.last_gpu_touch_ts,
                ),
            )
        if policy == "unison_cfs":
            return min(
                cands,
                key=lambda s: (s.kv_bytes / (1024.0 * 1024.0))
                * unison_hazard(s.spec.turns[s.cur_turn_idx].cmd_hint, now - s.tool_launch_ts),
            )
        return min(
            cands,
            key=lambda s: -1e6
            if s.is_stalled_est
            else ((s.kv_bytes / (1024.0 * 1024.0)) ** 2 / nvme_bw_mb_s) / max(s.rttc_est_s, 0.10),
        )

    def ensure_hbm_space(needed_bytes: int, protect_sid: int, now: float) -> bool:
        while current_hbm_bytes() + needed_bytes > hbm_cap_bytes:
            v = pick_victim(protect_sid, now)
            if v is None:
                return False
            evict_to_disk(v, "reactive")
        return True

    pending_completions: List[DMACompletion] = []

    def drain_dma(now: float) -> None:
        while True:
            try:
                pending_completions.append(dma_engine.done_q.get_nowait())
            except queue.Empty:
                break
        still_pending: List[DMACompletion] = []
        for comp in pending_completions:
            s = sessions[comp.session_id]
            if s.cur_turn_idx != comp.turn_idx or s.in_hbm or s.state == "completed":
                with state_lock:
                    s.load_in_flight = False
                continue
            est_b = int(comp.kv_mb * 1024 * 1024)
            if comp.is_jit and s.state == "running_tool":
                while current_hbm_bytes() + est_b > hbm_cap_bytes:
                    v = pick_victim(s.spec.session_id, now)
                    if v is None:
                        break
                    v_mb = v.kv_bytes / (1024.0 * 1024.0)
                    v_sc = -1e6 if v.is_stalled_est else (v_mb * (v_mb / nvme_bw_mb_s)) / max(v.rttc_est_s, 0.10)
                    s_sc = (comp.kv_mb * (comp.kv_mb / nvme_bw_mb_s)) / max(s.rttc_est_s, 0.10)
                    if v_sc >= s_sc:
                        break
                    evict_to_disk(v, "reactive")
                if current_hbm_bytes() + est_b > hbm_cap_bytes:
                    still_pending.append(comp)
                    continue
            else:
                ok = ensure_hbm_space(est_b, s.spec.session_id, now)
                if not ok and current_hbm_bytes() + est_b > hbm_cap_bytes:
                    still_pending.append(comp)
                    continue

            mps_cache = cpu_tuples_to_mps_cache(comp.cpu_tuples, device=device)
            torch.mps.synchronize()
            t_mat = time.perf_counter()
            with state_lock:
                s.cache = mps_cache
                s.kv_bytes = cache_kv_bytes(mps_cache)
                s.in_hbm = True
                s.load_in_flight = False
                s.last_gpu_touch_ts = t_mat
                if s.state == "running_tool" and comp.is_jit:
                    s.jit_promoted_this_turn = True
                    metrics_accum["jit_promotions_landed"] += 1
                elif s.state == "waiting_gpu":
                    s.ready_in_hbm_ts = t_mat
        pending_completions[:] = still_pending

    def enqueue_cold_reloads() -> None:
        for s in sessions:
            if s.state == "waiting_gpu" and not s.in_hbm and not s.load_in_flight:
                kv_mb = max(min_session_mb, s.kv_bytes / (1024.0 * 1024.0))
                with state_lock:
                    s.load_in_flight = True
                    s.load_is_jit = False
                dma_engine.request_load(s.disk_req_id, s.spec.session_id, s.cur_turn_idx, s.disk_path, kv_mb, False)

    def spawn_devtool_pgroup(s: SessionRuntime, now: float) -> None:
        turn = s.spec.turns[s.cur_turn_idx]
        stall_r, stall_w = -1, -1
        pass_fds: Tuple[int, ...] = ()
        if turn.kind == "stalled_io":
            stall_r, stall_w = os.pipe()
            pass_fds = (stall_r,)

        cmd = [
            sys.executable, "-u", worker_script,
            "--kind", turn.kind,
            "--scratch-root", run_scratch,
            "--session-id", str(s.spec.session_id),
            "--turn-idx", str(turn.turn_idx),
            "--pytest-iters", str(turn.pytest_iters),
            "--clang-tus", str(turn.clang_tus),
            "--stall-fd", str(stall_r),
        ]
        # start_new_session=True creates an isolated POSIX session & process group (pgid == proc.pid)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            pass_fds=pass_fds,
            start_new_session=True,
        )
        if stall_r >= 0:
            os.close(stall_r)
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        os.set_blocking(fd, False)
        pgid = proc.pid
        killpg_safe(pgid, signal.SIGSTOP)

        with state_lock:
            s.proc = proc
            s.pgid = pgid
            s.stdout_fd = fd
            s.stall_w_fd = stall_w
            s.stall_release_ts = now + turn.stall_sec if turn.kind == "stalled_io" else 0.0
            s.stall_released = turn.kind != "stalled_io"
            s.is_sigstopped = True
            s.tool_launch_ts = now
            s.last_stdout_ts = now
            s.unstopped_silence_s = 0.0
            s.stdout_buffer = ""
            s.stdout_bytes = 0
            s.progress_frac = 0.0
            s.cpu_granted_s = 0.0
            s.prefilled_stdout_tokens = 0
            s.micro_chunks_this_turn = 0
            s.was_evicted_this_turn = not s.in_hbm
            s.jit_promoted_this_turn = False
            s.had_hit_at_exit = None
            s.ready_in_hbm_ts = 0.0
            s.turn_hbm_byte_s_accum = 0.0
            s.is_stalled_est = False
            s.rttc_est_s = {"lint": 0.25, "pytest": 1.85, "build": 4.00}[turn.cmd_hint]
            s.state = "running_tool"

    def os_pgroup_scheduler_loop() -> None:
        quantum_s = 0.080
        last_tick = time.perf_counter()
        while not stop_sched_event.is_set():
            time.sleep(quantum_s)
            now = time.perf_counter()
            dt = min(0.20, max(0.005, now - last_tick))
            last_tick = now

            with state_lock:
                for s in sessions:
                    if s.in_hbm and s.state in ("running_tool", "waiting_gpu"):
                        bs = s.kv_bytes * dt
                        metrics_accum["total_hbm_byte_s"] += bs
                        if s.state == "running_tool":
                            s.turn_hbm_byte_s_accum += bs
                            if not s.stall_released and (now - s.tool_launch_ts) > 0.15:
                                metrics_accum["stalled_holding_byte_s"] += bs

                for s in [x for x in sessions if x.state == "running_tool" and x.proc is not None]:
                    if not s.stall_released and now >= s.stall_release_ts and s.stall_w_fd >= 0:
                        try:
                            os.write(s.stall_w_fd, b"GO\n")
                            os.close(s.stall_w_fd)
                        except OSError:
                            pass
                        s.stall_w_fd = -1
                        s.stall_released = True

                    if s.stdout_fd >= 0:
                        chunks: List[bytes] = []
                        while True:
                            try:
                                raw = os.read(s.stdout_fd, 65536)
                                if not raw:
                                    break
                                chunks.append(raw)
                            except (BlockingIOError, OSError):
                                break
                        if chunks:
                            joined = b"".join(chunks)
                            txt = joined.decode("utf-8", errors="replace")
                            s.stdout_buffer += txt
                            s.stdout_bytes += len(joined)
                            s.last_stdout_ts = now
                            s.unstopped_silence_s = 0.0
                            matches = PROGRESS_RE.findall(txt)
                            if matches:
                                s.progress_frac = max(s.progress_frac, int(matches[-1]) / 100.0)
                        elif not s.is_sigstopped:
                            s.unstopped_silence_s += dt

                    assert s.proc is not None
                    if s.proc.poll() is not None:
                        if s.stdout_fd >= 0:
                            try:
                                while True:
                                    rem = os.read(s.stdout_fd, 65536)
                                    if not rem:
                                        break
                                    s.stdout_buffer += rem.decode("utf-8", errors="replace")
                                    s.stdout_bytes += len(rem)
                            except OSError:
                                pass
                        if s.stall_w_fd >= 0:
                            try:
                                os.close(s.stall_w_fd)
                            except OSError:
                                pass
                            s.stall_w_fd = -1
                        s.progress_frac = 1.0
                        s.tool_exit_ts = now
                        s.had_hit_at_exit = s.in_hbm
                        if s.in_hbm:
                            s.ready_in_hbm_ts = now
                        s.state = "waiting_gpu"

                still_running = [x for x in sessions if x.state == "running_tool" and x.proc is not None]
                runnable: List[SessionRuntime] = []
                sleeping_io: List[SessionRuntime] = []
                for s in still_running:
                    if not s.stall_released and (now - s.tool_launch_ts) >= 0.15:
                        sleeping_io.append(s)
                    else:
                        runnable.append(s)

                for s in sleeping_io:
                    if s.is_sigstopped:
                        killpg_safe(s.pgid, signal.SIGCONT)
                        s.is_sigstopped = False

                if policy in ("symbioos_sched", "symbioos_full"):
                    for s in still_running:
                        update_rttc_filter(s, now, cpu_cores, max(len(runnable), 1))

                if policy in ("lru_cfs", "continuum_cfs", "unison_cfs"):
                    ordered = sorted(runnable, key=lambda s: (s.cpu_granted_s, s.tool_launch_ts))
                else:
                    def gittins_score(s: SessionRuntime) -> float:
                        kv_mb = max(min_session_mb, s.kv_bytes / (1024.0 * 1024.0))
                        reload_s = kv_mb / nvme_bw_mb_s
                        wait_s = max(0.0, (now - s.tool_launch_ts) - s.cpu_granted_s)
                        turns_factor = 1.15 if (s.cur_turn_idx + 1 < len(s.spec.turns)) else 1.0
                        val = (2.6 if (s.in_hbm or (s.load_in_flight and s.load_is_jit)) else 0.12) * kv_mb * reload_s
                        return turns_factor * ((val / max(s.rttc_est_s, 0.10)) + 0.35 * wait_s)

                    ordered = sorted(runnable, key=gittins_score, reverse=True)

                chosen = set(s.spec.session_id for s in ordered[:cpu_cores])
                for s in runnable:
                    if s.spec.session_id in chosen:
                        if s.is_sigstopped:
                            killpg_safe(s.pgid, signal.SIGCONT)
                            s.is_sigstopped = False
                        s.cpu_granted_s += dt
                    else:
                        if not s.is_sigstopped:
                            killpg_safe(s.pgid, signal.SIGSTOP)
                            s.is_sigstopped = True

    sched_thread = threading.Thread(target=os_pgroup_scheduler_loop, daemon=True)
    t_bench_start = time.perf_counter()
    sched_thread.start()

    def run_tiering_tick(now: float) -> None:
        drain_dma(now)
        enqueue_cold_reloads()
        hbm_util = current_hbm_bytes() / max(hbm_cap_bytes, 1)
        if policy == "continuum_cfs" and hbm_util > 0.50:
            for s in sessions:
                if s.in_hbm and s.state == "running_tool":
                    if (now - s.tool_launch_ts) > continuum_ttl(s.spec.turns[s.cur_turn_idx].cmd_hint):
                        evict_to_disk(s, "ttl_expire")
        elif policy in ("symbioos_sched", "symbioos_full"):
            for s in sessions:
                if s.in_hbm and s.state == "running_tool":
                    cmd = s.spec.turns[s.cur_turn_idx].cmd_hint
                    if s.is_stalled_est or (cmd == "build" and s.rttc_est_s > 2.60 and hbm_util > 0.65):
                        evict_to_disk(s, "early_demote")
            for s in sessions:
                if (
                    not s.in_hbm and not s.load_in_flight and s.state == "running_tool"
                    and not s.is_stalled_est and s.progress_frac >= 0.60
                ):
                    kv_mb = max(min_session_mb, s.kv_bytes / (1024.0 * 1024.0))
                    t_dma = kv_mb / nvme_bw_mb_s
                    if s.rttc_est_s <= t_dma + 0.50:
                        free_b = hbm_cap_bytes - current_hbm_bytes()
                        dem_b = sum(
                            x.kv_bytes for x in sessions
                            if x.in_hbm and x.state == "running_tool" and (x.is_stalled_est or x.rttc_est_s > 2.20)
                        )
                        if free_b + dem_b >= s.kv_bytes:
                            with state_lock:
                                s.load_in_flight = True
                                s.load_is_jit = True
                                metrics_accum["jit_promotions_started"] += 1
                            dma_engine.request_load(s.disk_req_id, s.spec.session_id, s.cur_turn_idx, s.disk_path, kv_mb, True)

    try:
        while True:
            now = time.perf_counter()
            rel_t = now - t_bench_start
            run_tiering_tick(now)

            for s in sessions:
                active_cnt = sum(1 for x in sessions if x.state in ("running_tool", "waiting_gpu"))
                if active_cnt >= max_concurrency:
                    break
                if s.state == "not_arrived" and rel_t >= s.spec.arrival_sec:
                    tmpl = base_templates[s.spec.base_prompt_tokens]
                    needed = cache_kv_bytes(tmpl)
                    ok = ensure_hbm_space(needed, s.spec.session_id, now)
                    if not ok and current_hbm_bytes() + needed > hbm_cap_bytes:
                        continue
                    t_admit = time.perf_counter()
                    cloned = clone_mps_cache(tmpl)
                    _ = forward_chunked_mps(model, admit_kick_ids, cloned, max_step=256, device=device)
                    torch.mps.synchronize()
                    now_after = time.perf_counter()
                    with state_lock:
                        s.episode_start_ts = t_admit
                        s.cache = cloned
                        s.kv_bytes = cache_kv_bytes(cloned)
                        s.in_hbm = True
                        s.last_gpu_touch_ts = now_after
                    spawn_devtool_pgroup(s, now_after)

            run_tiering_tick(time.perf_counter())

            ready = sorted(
                [s for s in sessions if s.state == "waiting_gpu" and s.in_hbm and s.cache is not None],
                key=lambda x: x.tool_exit_ts,
            )
            if ready:
                s = ready[0]
                t_serve = time.perf_counter()
                hit_at_exit = bool(s.had_hit_at_exit)
                unint_hit = hit_at_exit and (not s.was_evicted_this_turn)
                jit_hit = hit_at_exit and s.jit_promoted_this_turn

                if hit_at_exit:
                    cold_wait_ms = 0.0
                    q_wait_ms = max(0.0, (t_serve - s.tool_exit_ts) * 1000.0)
                else:
                    t_mat = max(s.tool_exit_ts, s.ready_in_hbm_ts)
                    cold_wait_ms = max(0.0, (t_mat - s.tool_exit_ts) * 1000.0)
                    q_wait_ms = max(0.0, (t_serve - t_mat) * 1000.0)

                t_pf0 = time.perf_counter()
                all_ids = tokenizer.encode(s.stdout_buffer, add_special_tokens=False)
                rem_ids = all_ids[s.prefilled_stdout_tokens :] + turn_close_ids
                assert s.cache is not None

                pos = 0
                first_logits: Optional[torch.Tensor] = None
                step_sz = 512 if micro_chunk_tokens == 512 else 256
                while pos < len(rem_ids):
                    sl = rem_ids[pos : pos + step_sz]
                    first_logits = forward_chunked_mps(model, sl, s.cache, max_step=step_sz, device=device)
                    torch.mps.synchronize()
                    pos += len(sl)
                    if pos < len(rem_ids):
                        run_tiering_tick(time.perf_counter())

                assert first_logits is not None
                first_tok = int(first_logits.argmax(dim=-1).item())
                t_first = time.perf_counter()

                eof_pf_ms = (t_first - t_pf0) * 1000.0
                ttft_ms = (t_first - s.tool_exit_ts) * 1000.0
                tspec = s.spec.turns[s.cur_turn_idx]

                turn_records.append({
                    "model_id": model_id,
                    "policy": policy,
                    "micro_chunk_tokens": micro_chunk_tokens if policy == "symbioos_full" else 0,
                    "seed": seed,
                    "session_id": s.spec.session_id,
                    "turn_idx": s.cur_turn_idx,
                    "kind": tspec.kind,
                    "kv_mb_at_resume": round(s.kv_bytes / (1024.0 * 1024.0), 2),
                    "tool_wall_sec": round(s.tool_exit_ts - s.tool_launch_ts, 3),
                    "hbm_hit_at_exit": hit_at_exit,
                    "uninterrupted_hbm_hit": unint_hit,
                    "jit_rescued_hit": jit_hit,
                    "micro_chunks_prefilled": s.micro_chunks_this_turn,
                    "remaining_tokens_at_eof": len(rem_ids),
                    "ttft_ms": round(ttft_ms, 2),
                    "cold_reload_wait_ms": round(cold_wait_ms, 2),
                    "gpu_queue_wait_ms": round(q_wait_ms, 2),
                    "eof_prefill_and_decode_ms": round(eof_pf_ms, 2),
                })

                cur_tok = first_tok
                for _ in range(2):
                    lg = forward_chunked_mps(model, [cur_tok], s.cache, max_step=256, device=device)
                    cur_tok = int(lg.argmax(dim=-1).item())
                if s.cur_turn_idx + 1 < len(s.spec.turns):
                    _ = forward_chunked_mps(model, next_open_ids, s.cache, max_step=256, device=device)
                torch.mps.synchronize()
                now_done = time.perf_counter()

                if s.cur_turn_idx + 1 < len(s.spec.turns):
                    new_b = cache_kv_bytes(s.cache)
                    delta = max(0, new_b - s.kv_bytes)
                    if delta > 0:
                        ensure_hbm_space(delta, s.spec.session_id, now_done)
                    with state_lock:
                        s.cur_turn_idx += 1
                        s.kv_bytes = new_b
                        s.last_gpu_touch_ts = now_done
                    spawn_devtool_pgroup(s, now_done)
                else:
                    with state_lock:
                        s.state = "completed"
                        s.episode_end_ts = now_done
                        s.in_hbm = False
                        s.cache = None
                        s.kv_bytes = 0
                continue

            did_micro = False
            if policy == "symbioos_full" and micro_chunk_tokens > 0:
                cands: List[Tuple[float, SessionRuntime, List[int]]] = []
                with state_lock:
                    snap = [
                        (s, s.stdout_buffer, s.prefilled_stdout_tokens, s.rttc_est_s)
                        for s in sessions
                        if s.state == "running_tool" and s.in_hbm and s.cache is not None and not s.is_stalled_est
                    ]
                for s, buf_txt, pref_cnt, rttc in snap:
                    if len(buf_txt) < (pref_cnt + micro_chunk_tokens) * 2:
                        continue
                    tids = tokenizer.encode(buf_txt, add_special_tokens=False)
                    safe = tids[:-1] if len(tids) > 1 else []
                    if len(safe) - pref_cnt >= micro_chunk_tokens:
                        cands.append((rttc, s, safe[pref_cnt : pref_cnt + micro_chunk_tokens]))
                if cands:
                    cands.sort(key=lambda x: x[0])
                    _, ts, ch_ids = cands[0]
                    add_b = micro_chunk_tokens * kv_bytes_per_tok
                    ensure_hbm_space(add_b, ts.spec.session_id, time.perf_counter())
                    if ts.in_hbm and ts.cache is not None and ts.state == "running_tool":
                        _ = forward_chunked_mps(model, ch_ids, ts.cache, max_step=micro_chunk_tokens, device=device)
                        torch.mps.synchronize()
                        with state_lock:
                            if ts.cache is not None and ts.in_hbm:
                                ts.prefilled_stdout_tokens += len(ch_ids)
                                ts.micro_chunks_this_turn += 1
                                ts.kv_bytes = cache_kv_bytes(ts.cache)
                                ts.last_gpu_touch_ts = time.perf_counter()
                        did_micro = True

            if all(s.state == "completed" for s in sessions):
                break
            if not did_micro:
                time.sleep(0.006)

        makespan_s = time.perf_counter() - t_bench_start
    finally:
        stop_sched_event.set()
        sched_thread.join(timeout=2.0)
        for s in sessions:
            if s.stall_w_fd >= 0:
                try:
                    os.close(s.stall_w_fd)
                except OSError:
                    pass
            if s.pgid > 0:
                killpg_safe(s.pgid, signal.SIGCONT)
                killpg_safe(s.pgid, signal.SIGKILL)
            if s.proc is not None:
                try:
                    s.proc.wait(timeout=1.5)
                except Exception:
                    pass
            s.cache = None
        dma_engine.shutdown()
        evict_tmpdir.cleanup()
        shutil.rmtree(run_scratch, ignore_errors=True)
        gc.collect()
        torch.mps.empty_cache()

    jcts = [s.episode_end_ts - s.episode_start_ts for s in sessions]
    return {
        "model_id": model_id,
        "policy": policy,
        "micro_chunk_tokens": micro_chunk_tokens if policy == "symbioos_full" else 0,
        "seed": seed,
        "makespan_sec": round(makespan_s, 3),
        "episode_jcts_sec": [round(x, 3) for x in jcts],
        "dead_evicted_waste_mb_s": round(metrics_accum["dead_evicted_waste_byte_s"] / (1024.0 * 1024.0), 2),
        "stalled_holding_mb_s": round(metrics_accum["stalled_holding_byte_s"] / (1024.0 * 1024.0), 2),
        "early_demotions": metrics_accum["early_demotions"],
        "jit_promotions_landed": metrics_accum["jit_promotions_landed"],
        "turn_records": turn_records,
    }


def pct(vals: List[float], q: float) -> float:
    return float(np.percentile(np.array(vals, dtype=np.float64), q)) if vals else 0.0


def summarize_group(label_dict: Dict[str, Any], runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    turns = [r for run in runs for r in run["turn_records"]]
    inter = [r for r in turns if r["kind"] in ("quick_lint", "pytest_suite")]
    ttfts = [r["ttft_ms"] for r in turns]
    jcts = [j for run in runs for j in run["episode_jcts_sec"]]
    tot_min = sum(run["makespan_sec"] for run in runs) / 60.0
    slo_cnt = sum(1 for r in turns if r["ttft_ms"] <= 600.0)
    return {
        **label_dict,
        "total_resumes": len(turns),
        "ttft_mean_ms": round(statistics.mean(ttfts), 2),
        "ttft_p50_ms": round(pct(ttfts, 50), 2),
        "ttft_p95_ms": round(pct(ttfts, 95), 2),
        "ttft_p99_ms": round(pct(ttfts, 99), 2),
        "cold_reload_mean_ms": round(statistics.mean(r["cold_reload_wait_ms"] for r in turns), 2),
        "eof_prefill_mean_ms": round(statistics.mean(r["eof_prefill_and_decode_ms"] for r in turns), 2),
        "hbm_hit_rate_pct": round(100.0 * sum(1 for r in turns if r["hbm_hit_at_exit"]) / max(len(turns), 1), 2),
        "interactive_hit_rate_pct": round(100.0 * sum(1 for r in inter if r["hbm_hit_at_exit"]) / max(len(inter), 1), 2),
        "total_waste_mb_s_per_seed": round(
            statistics.mean(run["dead_evicted_waste_mb_s"] + run["stalled_holding_mb_s"] for run in runs), 2
        ),
        "episode_jct_mean_sec": round(statistics.mean(jcts), 3),
        "cluster_goodput_ep_per_min": round(len(jcts) / max(tot_min, 1e-6), 2),
        "slo_goodput_le_600ms_per_min": round(slo_cnt / max(tot_min, 1e-6), 2),
        "micro_chunks_prefilled": sum(r["micro_chunks_prefilled"] for r in turns),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,137")
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--max-concurrency", type=int, default=12)
    parser.add_argument("--cpu-cores", type=int, default=4)
    parser.add_argument("--out-json", default="results/phase1_multimodel_devtools.json")
    parser.add_argument("--out-md", default="results/PHASE1_MULTIMODEL_SUMMARY.md")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    workloads = {sd: generate_devtool_workload(sd, num_episodes=args.num_episodes) for sd in seeds}

    # Three distinct model architectures cached locally on workstation:
    # 1. Qwen2.5-3B-Instruct (Dense GQA 36 KB/tok)
    # 2. Qwen2.5-7B-Instruct (Dense GQA 56 KB/tok)
    # 3. allenai/OLMoE-1B-7B-0924-Instruct (Sparse MoE MHA 128 KB/tok)
    model_configs = [
        {"model_id": "Qwen/Qwen2.5-3B-Instruct", "arch": "Dense GQA-8:1 (36 KB/tok)", "hbm_cap_mb": 320.0, "nvme_bw_mb_s": 160.0},
        {"model_id": "Qwen/Qwen2.5-7B-Instruct", "arch": "Dense GQA-7:1 (56 KB/tok)", "hbm_cap_mb": 500.0, "nvme_bw_mb_s": 250.0},
        {"model_id": "allenai/OLMoE-1B-7B-0924-Instruct", "arch": "Sparse MoE MHA (128 KB/tok)", "hbm_cap_mb": 1140.0, "nvme_bw_mb_s": 560.0},
    ]
    policies = ["lru_cfs", "continuum_cfs", "unison_cfs", "symbioos_sched", "symbioos_full"]

    multimodel_summaries: List[Dict[str, Any]] = []
    chunk_ablation_summaries: List[Dict[str, Any]] = []
    all_raw_runs: List[Dict[str, Any]] = []

    for mcfg in model_configs:
        mid = mcfg["model_id"]
        print(f"\n=== Loading Model: {mid} ({mcfg['arch']}) ===", flush=True)
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, local_files_only=True).to("mps").eval()
        warmup_mps_buckets(model, device="mps")
        templates = build_base_prompt_templates(model, tokenizer, [1024, 1280, 1536, 1792], device="mps")
        kv_b_per_tok = cache_kv_bytes(templates[1024]) // 1024
        print(f"    Ready in {time.perf_counter()-t0:.1f}s | Measured KV = {kv_b_per_tok/1024:.1f} KB/tok", flush=True)

        for pol in policies:
            pol_runs = []
            for sd in seeds:
                tp = time.perf_counter()
                print(f"  [{mid.split('/')[-1]}] policy={pol:<15s} seed={sd} ...", end="", flush=True)
                res = run_devtool_policy(
                    model_id=mid,
                    policy=pol,
                    seed=sd,
                    episodes=workloads[sd],
                    model=model,
                    tokenizer=tokenizer,
                    base_templates=templates,
                    kv_bytes_per_tok=kv_b_per_tok,
                    max_concurrency=args.max_concurrency,
                    hbm_cap_mb=mcfg["hbm_cap_mb"],
                    cpu_cores=args.cpu_cores,
                    nvme_bw_mb_s=mcfg["nvme_bw_mb_s"],
                    micro_chunk_tokens=256,
                    device="mps",
                )
                pol_runs.append(res)
                all_raw_runs.append(res)
                s_one = summarize_group({}, [res])
                print(
                    f" {time.perf_counter()-tp:.1f}s | P50={s_one['ttft_p50_ms']:.1f}ms "
                    f"P95={s_one['ttft_p95_ms']:.1f}ms | Hit={s_one['hbm_hit_rate_pct']:.1f}% | "
                    f"JCT={s_one['episode_jct_mean_sec']:.2f}s",
                    flush=True,
                )
            multimodel_summaries.append(
                summarize_group({"model_id": mid, "arch": mcfg["arch"], "policy": pol}, pol_runs)
            )

        # On Qwen2.5-3B-Instruct, also run the Chunk-Size Sensitivity Ablation (128, 256, 512, inf)!
        if mid == "Qwen/Qwen2.5-3B-Instruct":
            print("\n  --- Running Streaming Micro-Prefill Chunk-Size Sensitivity (128 vs 256 vs 512 vs inf) ---", flush=True)
            # Reuse 256 (symbioos_full) and inf (symbioos_sched) already executed above!
            sched_runs = [r for r in all_raw_runs if r["model_id"] == mid and r["policy"] == "symbioos_sched"]
            full256_runs = [r for r in all_raw_runs if r["model_id"] == mid and r["policy"] == "symbioos_full" and r["micro_chunk_tokens"] == 256]
            chunk_ablation_summaries.append(summarize_group({"chunk_setting": "inf (No Stream / SymbioOS-Sched)", "chunk_tokens": 0}, sched_runs))
            for ch_sz in [128, 512]:
                ch_runs = []
                for sd in seeds:
                    tp = time.perf_counter()
                    print(f"  [ChunkAblation] chunk={ch_sz} seed={sd} ...", end="", flush=True)
                    res = run_devtool_policy(
                        model_id=mid,
                        policy="symbioos_full",
                        seed=sd,
                        episodes=workloads[sd],
                        model=model,
                        tokenizer=tokenizer,
                        base_templates=templates,
                        kv_bytes_per_tok=kv_b_per_tok,
                        max_concurrency=args.max_concurrency,
                        hbm_cap_mb=mcfg["hbm_cap_mb"],
                        cpu_cores=args.cpu_cores,
                        nvme_bw_mb_s=mcfg["nvme_bw_mb_s"],
                        micro_chunk_tokens=ch_sz,
                        device="mps",
                    )
                    ch_runs.append(res)
                    all_raw_runs.append(res)
                    s_one = summarize_group({}, [res])
                    print(
                        f" {time.perf_counter()-tp:.1f}s | P50={s_one['ttft_p50_ms']:.1f}ms "
                        f"EOF_pf={s_one['eof_prefill_mean_ms']:.1f}ms chunks={s_one['micro_chunks_prefilled']}",
                        flush=True,
                    )
                chunk_ablation_summaries.append(summarize_group({"chunk_setting": f"{ch_sz} tokens", "chunk_tokens": ch_sz}, ch_runs))
            chunk_ablation_summaries.insert(2, summarize_group({"chunk_setting": "256 tokens (Default)", "chunk_tokens": 256}, full256_runs))

        # Incremental checkpoint save after each model completes!
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "multimodel_summary": multimodel_summaries,
                    "chunk_ablation_summary": chunk_ablation_summaries,
                    "runs": all_raw_runs,
                },
                f,
                indent=2,
            )

        del templates, model, tokenizer
        gc.collect()
        torch.mps.empty_cache()

    # Write clean Markdown report
    md_lines = [
        "# SymbioOS Phase 1 Real-DevTool (`pytest` + `clang++` + `git`) & Multi-Architecture Results (Apple M5 Pro)",
        "",
        "## 1. Multi-Architecture 5-Policy Benchmark (`os.killpg` POSIX Process Groups)",
        "",
        "| Model (`fp16` MPS) | Policy | TTFT P50 (ms) | TTFT P95 (ms) | HBM Hit (%) | Interactive Hit (%) | HBM Waste (MB·s) | Mean JCT (s) | Goodput (ep/min) | SLO Goodput (`<=600ms`/min) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in multimodel_summaries:
        m_short = s["model_id"].split("/")[-1]
        md_lines.append(
            f"| `{m_short}` | `{s['policy']}` | {s['ttft_p50_ms']:.1f} | {s['ttft_p95_ms']:.1f} | "
            f"{s['hbm_hit_rate_pct']:.1f}% | {s['interactive_hit_rate_pct']:.1f}% | "
            f"{s['total_waste_mb_s_per_seed']:.1f} | {s['episode_jct_mean_sec']:.2f} | "
            f"{s['cluster_goodput_ep_per_min']:.2f} | {s['slo_goodput_le_600ms_per_min']:.1f} |"
        )

    md_lines.extend([
        "",
        "## 2. Streaming `stdout` Micro-Prefill Chunk-Size Ablation (`Qwen2.5-3B-Instruct`)",
        "",
        "| Chunk Setting | TTFT P50 (ms) | TTFT P95 (ms) | Mean EOF Prefill+Decode (ms) | HBM Hit (%) | Mean JCT (s) | Micro-Chunks Prefilled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for c in chunk_ablation_summaries:
        md_lines.append(
            f"| `{c['chunk_setting']}` | {c['ttft_p50_ms']:.1f} | {c['ttft_p95_ms']:.1f} | "
            f"{c['eof_prefill_mean_ms']:.1f} | {c['hbm_hit_rate_pct']:.1f}% | "
            f"{c['episode_jct_mean_sec']:.2f} | {c['micro_chunks_prefilled']} |"
        )

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nSaved Phase 1 JSON -> {args.out_json}\nSaved Phase 1 Markdown -> {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
