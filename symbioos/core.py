"""Core scheduling, sub-turn pipe filtration, JIT staging, and streaming prefill primitives."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import signal
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROGRESS_RE = re.compile(r"\[\s*(\d{1,3})%\]")


@dataclass
class SandboxTelemetry:
    """Observable sub-turn filtration state z_i(t) = (p_i, b_i, u_i, sigma_i)."""

    session_id: int
    cmd_hint: str = "pytest"
    pgid: int = -1
    kv_mb: float = 48.0
    in_hbm: bool = True
    jit_in_flight: bool = False
    is_sigstopped: bool = False
    turns_remaining: int = 1
    progress_frac: float = 0.0
    stdout_bytes: int = 0
    cpu_granted_s: float = 0.0
    unstopped_silence_s: float = 0.0
    elapsed_wall_s: float = 0.0
    rttc_est_s: float = 2.0
    is_stalled_est: bool = False


class SubTurnRTTCFilter:
    """White-box sub-turn residual-time and unstopped-quantum stall filter."""

    def __init__(
        self,
        class_priors_s: Optional[Dict[str, float]] = None,
        stall_silence_threshold_s: float = 1.25,
        min_milestone_frac: float = 0.10,
    ) -> None:
        self.class_priors_s = class_priors_s or {"lint": 0.30, "pytest": 1.95, "build": 4.50}
        self.stall_silence_threshold_s = stall_silence_threshold_s
        self.min_milestone_frac = min_milestone_frac

    def ingest_stdout_chunk(self, state: SandboxTelemetry, text: str, dt_s: float) -> None:
        if text:
            raw_len = len(text.encode("utf-8", errors="replace"))
            state.stdout_bytes += raw_len
            state.unstopped_silence_s = 0.0
            matches = PROGRESS_RE.findall(text)
            if matches:
                pct = min(100, max(0, int(matches[-1])))
                state.progress_frac = max(state.progress_frac, pct / 100.0)
        elif not state.is_sigstopped:
            state.unstopped_silence_s += max(0.0, dt_s)

    def update_posterior(self, state: SandboxTelemetry, cpu_cores: int, runnable_count: int) -> Tuple[float, bool]:
        if state.progress_frac < 0.05 and state.unstopped_silence_s >= self.stall_silence_threshold_s:
            state.is_stalled_est = True
            state.rttc_est_s = max(1.0, 5.0 - state.elapsed_wall_s)
            return state.rttc_est_s, True

        state.is_stalled_est = False
        if state.progress_frac >= 0.99:
            rem_cpu = 0.04
        elif state.progress_frac >= self.min_milestone_frac and state.cpu_granted_s > 0.02:
            rem_cpu = max(0.05, state.cpu_granted_s * (1.0 - state.progress_frac) / state.progress_frac)
        else:
            prior = self.class_priors_s.get(state.cmd_hint, 2.0)
            rem_cpu = max(0.08, prior - state.cpu_granted_s)

        share = 1.0 if state.in_hbm else min(1.0, max(0.25, cpu_cores / max(runnable_count, 1)))
        state.rttc_est_s = rem_cpu / share
        return state.rttc_est_s, False


class LookbackBPEStreamer:
    """One-token BPE lookback guard (Proposition 1) for exact streaming stdout micro-prefill."""

    def __init__(self, encode_fn: Callable[[str], List[int]], chunk_tokens: int = 256) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        self.encode_fn = encode_fn
        self.chunk_tokens = chunk_tokens
        self.buffer: str = ""
        self.prefilled_tokens: List[int] = []

    def append_stdout(self, text: str) -> List[List[int]]:
        """Append live stdout text and return zero or more exact BPE-safe micro-prefill chunks."""
        if not text:
            return []
        self.buffer += text
        cur_ids = self.encode_fn(self.buffer)
        safe_ids = cur_ids[:-1] if len(cur_ids) > 1 else []
        ready_chunks: List[List[int]] = []
        while len(safe_ids) - len(self.prefilled_tokens) >= self.chunk_tokens:
            start = len(self.prefilled_tokens)
            nxt = safe_ids[start : start + self.chunk_tokens]
            self.prefilled_tokens.extend(nxt)
            ready_chunks.append(nxt)
        return ready_chunks

    def finalize_eof(self) -> List[int]:
        """Return remaining unprefilled tail token IDs at process exit."""
        eof_ids = self.encode_fn(self.buffer)
        return eof_ids[len(self.prefilled_tokens) :]


class KVWeightedGittinsScheduler:
    """Theorem 1 KV-weighted residual-hazard process-group scheduler."""

    def __init__(
        self,
        cpu_cores: int = 4,
        nvme_bw_mb_s: float = 160.0,
        hbm_residency_weight: float = 2.6,
        evicted_weight: float = 0.12,
        aging_slope: float = 0.35,
    ) -> None:
        self.cpu_cores = max(1, cpu_cores)
        self.nvme_bw_mb_s = max(1.0, nvme_bw_mb_s)
        self.hbm_residency_weight = hbm_residency_weight
        self.evicted_weight = evicted_weight
        self.aging_slope = aging_slope

    def score(self, s: SandboxTelemetry) -> float:
        reload_s = s.kv_mb / self.nvme_bw_mb_s
        gamma = self.hbm_residency_weight if (s.in_hbm or s.jit_in_flight) else self.evicted_weight
        wait_s = max(0.0, s.elapsed_wall_s - s.cpu_granted_s)
        psi = 1.15 if s.turns_remaining > 1 else 1.0
        return psi * ((gamma * s.kv_mb * reload_s) / max(s.rttc_est_s, 0.10) + self.aging_slope * wait_s)

    def select_top_c(self, runnable: Sequence[SandboxTelemetry]) -> List[int]:
        ordered = sorted(runnable, key=self.score, reverse=True)
        return [s.session_id for s in ordered[: self.cpu_cores]]

    @staticmethod
    def apply_pgid_signals(runnable: Iterable[SandboxTelemetry], chosen_ids: set[int]) -> None:
        for s in runnable:
            if s.pgid <= 0:
                continue
            target_sig = signal.SIGCONT if s.session_id in chosen_ids else signal.SIGSTOP
            try:
                os.killpg(s.pgid, target_sig)
                s.is_sigstopped = target_sig == signal.SIGSTOP
            except (ProcessLookupError, PermissionError, OSError):
                pass


@dataclass
class JITDecision:
    demote_ids: List[int]
    promote_ids: List[int]


class OptimalStoppingJITController:
    """Theorem 2 early stall demotion and critical-fractile JIT KV promotion controller."""

    def __init__(
        self,
        hbm_cap_mb: float,
        nvme_bw_mb_s: float = 160.0,
        delta_jit_s: float = 0.35,
        min_progress_for_jit: float = 0.60,
    ) -> None:
        self.hbm_cap_mb = hbm_cap_mb
        self.nvme_bw_mb_s = max(1.0, nvme_bw_mb_s)
        self.delta_jit_s = delta_jit_s
        self.min_progress_for_jit = min_progress_for_jit

    def evaluate(self, sandboxes: Sequence[SandboxTelemetry]) -> JITDecision:
        used_mb = sum(s.kv_mb for s in sandboxes if s.in_hbm)
        util = used_mb / max(self.hbm_cap_mb, 1e-6)

        demote: List[int] = []
        for s in sandboxes:
            if s.in_hbm:
                if s.is_stalled_est or (s.cmd_hint == "build" and s.rttc_est_s > 2.60 and util > 0.65):
                    demote.append(s.session_id)

        demoted_mb = sum(s.kv_mb for s in sandboxes if s.session_id in set(demote))
        free_plus_demotable = (self.hbm_cap_mb - used_mb) + demoted_mb + sum(
            s.kv_mb for s in sandboxes
            if s.in_hbm and s.session_id not in set(demote) and s.rttc_est_s > 2.40
        )

        promote: List[int] = []
        budget = free_plus_demotable
        for s in sorted(sandboxes, key=lambda x: x.rttc_est_s):
            if (
                not s.in_hbm
                and not s.jit_in_flight
                and not s.is_stalled_est
                and s.progress_frac >= self.min_progress_for_jit
            ):
                t_dma = s.kv_mb / self.nvme_bw_mb_s
                if s.rttc_est_s <= t_dma + self.delta_jit_s and budget >= s.kv_mb:
                    promote.append(s.session_id)
                    budget -= s.kv_mb

        return JITDecision(demote_ids=demote, promote_ids=promote)
