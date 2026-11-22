"""Unit tests for SymbioOS core primitives (Proposition 1, Theorem 1, Theorem 2)."""

from symbioos import (
    KVWeightedGittinsScheduler,
    LookbackBPEStreamer,
    OptimalStoppingJITController,
    SandboxTelemetry,
    SubTurnRTTCFilter,
)


def _mock_bpe_encode(text: str) -> list[int]:
    """Toy subword merger that merges 'PASSED' into one token and splits partial prefixes."""
    tokens: list[int] = []
    for word in text.splitlines(keepends=True):
        for part in word.split(" "):
            if not part:
                continue
            tokens.append(hash(part) % 50_000)
    return tokens


def test_lookback_bpe_streamer_exact_prefix_parity():
    lines = [
        f"tests/test_mod_{i:02d}.py::test_inv PASSED [{i*10:3d}%]\n"
        for i in range(1, 11)
    ]
    streamer = LookbackBPEStreamer(_mock_bpe_encode, chunk_tokens=8)
    emitted: list[int] = []
    for line in lines:
        # Feed in arbitrary mid-token slices to stress-test 1-token lookback!
        mid = len(line) // 2
        for ch in streamer.append_stdout(line[:mid]):
            emitted.extend(ch)
        for ch in streamer.append_stdout(line[mid:]):
            emitted.extend(ch)
    tail = streamer.finalize_eof()
    full_eof = _mock_bpe_encode("".join(lines))
    assert emitted + tail == full_eof
    assert len(emitted) >= 24


def test_unstopped_silence_ignores_sigstopped_throttling():
    flt = SubTurnRTTCFilter(stall_silence_threshold_s=1.25)
    throttled = SandboxTelemetry(session_id=1, cmd_hint="pytest", is_sigstopped=True)
    for _ in range(25):
        flt.ingest_stdout_chunk(throttled, "", dt_s=0.08)
    _, stalled_a = flt.update_posterior(throttled, cpu_cores=4, runnable_count=8)
    assert not stalled_a
    assert throttled.unstopped_silence_s == 0.0

    # Unstopped socket block triggers stall detection cleanly at >= 1.25s
    blocked = SandboxTelemetry(session_id=2, cmd_hint="pytest", is_sigstopped=False)
    for _ in range(17):
        flt.ingest_stdout_chunk(blocked, "", dt_s=0.08)
    _, stalled_b = flt.update_posterior(blocked, cpu_cores=4, runnable_count=8)
    assert stalled_b
    assert blocked.is_stalled_est


def test_gittins_prioritizes_hot_hbm_near_completion():
    sched = KVWeightedGittinsScheduler(cpu_cores=1, nvme_bw_mb_s=160.0)
    hot_pytest = SandboxTelemetry(session_id=10, kv_mb=72.0, in_hbm=True, rttc_est_s=0.60)
    cold_compile = SandboxTelemetry(session_id=11, kv_mb=72.0, in_hbm=False, rttc_est_s=3.80)
    chosen = sched.select_top_c([cold_compile, hot_pytest])
    assert chosen == [10]


def test_theorem2_jit_controller_demotes_stall_and_promotes_imminent():
    ctrl = OptimalStoppingJITController(hbm_cap_mb=200.0, nvme_bw_mb_s=160.0, delta_jit_s=0.35)
    stalled = SandboxTelemetry(session_id=1, kv_mb=80.0, in_hbm=True, is_stalled_est=True, rttc_est_s=4.2)
    active_hbm = SandboxTelemetry(session_id=2, kv_mb=80.0, in_hbm=True, is_stalled_est=False, rttc_est_s=0.9)
    imminent_cold = SandboxTelemetry(
        session_id=3, kv_mb=64.0, in_hbm=False, progress_frac=0.80, rttc_est_s=0.55
    )
    dec = ctrl.evaluate([stalled, active_hbm, imminent_cold])
    assert dec.demote_ids == [1]
    assert dec.promote_ids == [3]
