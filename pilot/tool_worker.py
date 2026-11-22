#!/usr/bin/env python3
"""Real POSIX tool subprocess workload for SymbioOS Phase 0 Pilot.

Executes genuine CPU-bound computation (SHA-256 + AST parsing) so that OS
process scheduling via SIGSTOP/SIGCONT directly controls wall-clock completion
rate, or blocks on an explicit socket/pipe file descriptor for I/O stalls.
Emits realistic unbuffered stdout progress lines over its POSIX pipe.
"""

import argparse
import ast
import hashlib
import os
import sys


AST_SNIPPET = """
class SchedulerNode:
    def __init__(self, cid: int, kv_bytes: int, rttc: float):
        self.cid = cid
        self.kv_bytes = kv_bytes
        self.rttc = max(rttc, 1e-3)

    def gittins_weight(self, reload_penalty: float) -> float:
        return (self.kv_bytes * reload_penalty) / self.rttc
"""


def burn_cpu(total_iters: int, seed_tag: str) -> str:
    """Execute deterministic CPU work (SHA-256 + periodic Python AST parse)."""
    h = seed_tag.encode("utf-8")
    ast_interval = 25_000
    for i in range(1, total_iters + 1):
        h = hashlib.sha256(h).digest()
        if i % ast_interval == 0:
            tree = ast.parse(AST_SNIPPET)
            _ = len(tree.body)
    return h[:6].hex()


def run_quick_lint(iters_per_sec: int, session_id: int, turn_idx: int, cpu_sec: float) -> None:
    print(
        f"ruff check src/agent_mod_{session_id:02d}/turn_{turn_idx}.py --isolated",
        flush=True,
    )
    steps = 2
    step_iters = max(10_000, int((cpu_sec * iters_per_sec) / steps))
    for s in range(1, steps + 1):
        digest = burn_cpu(step_iters, f"lint-{session_id}-{turn_idx}-{s}")
        pct = int((s * 100) // steps)
        print(
            f"[{pct:3d}%] Checked 14 AST rules in module_{session_id:02d}_part_{s}.py "
            f"(0 errors, 1 style hint, sig={digest})",
            flush=True,
        )
    print("All checks passed! (28 files scanned, 0 violations)", flush=True)


def run_pytest_suite(iters_per_sec: int, session_id: int, turn_idx: int, cpu_sec: float) -> None:
    print(
        f"============================= test session starts =============================\n"
        f"platform darwin -- Python 3.11, pytest-8.3.2, session={session_id:02d}, turn={turn_idx}\n"
        f"collected 30 items in tests/runtime/test_kv_sandbox_{session_id:02d}.py",
        flush=True,
    )
    steps = 10
    step_iters = max(20_000, int((cpu_sec * iters_per_sec) / steps))
    for s in range(1, steps + 1):
        digest = burn_cpu(step_iters, f"pytest-{session_id}-{turn_idx}-{s}")
        pct = s * 10
        lines = []
        for sub in range(1, 3):
            item_no = (s - 1) * 2 + sub
            lines.append(
                f"tests/runtime/test_kv_sandbox_{session_id:02d}.py::"
                f"TestContainerEngine{s:02d}::test_case_{item_no:02d}_invariants PASSED [{pct:3d}%]"
            )
        lines.append(
            f"  [telemetry] suite_chunk={s:02d}/10 assertions=32 state_hash={digest} status=ok"
        )
        print("\n".join(lines), flush=True)
    print("============================== 20 passed in suite ==============================", flush=True)


def run_heavy_compile(iters_per_sec: int, session_id: int, turn_idx: int, cpu_sec: float) -> None:
    print(
        f"[  0%] Configuring native extension build target=symbio_ext_{session_id:02d}_t{turn_idx}",
        flush=True,
    )
    steps = 10
    step_iters = max(30_000, int((cpu_sec * iters_per_sec) / steps))
    for s in range(1, steps + 1):
        digest = burn_cpu(step_iters, f"compile-{session_id}-{turn_idx}-{s}")
        pct = s * 10
        print(
            f"[{pct:3d}%] Building CXX object CMakeFiles/ext_{session_id:02d}.dir/src/kernel_{s:02d}.cpp.o\n"
            f"  cc -O3 -std=c++20 -Wall -Wextra -fPIC -c kernel_{s:02d}.cpp -> obj_hash={digest}\n"
            f"  optimizing loop unroll pass {s}/10: register_spill=0 vector_width=8",
            flush=True,
        )
    print(f"[100%] Linking CXX shared module symbio_ext_{session_id:02d}.so — build succeeded.", flush=True)


def run_stalled_io(
    iters_per_sec: int,
    session_id: int,
    turn_idx: int,
    cpu_sec: float,
    stall_fd: int,
) -> None:
    # Ambiguous initial header: looks identical at t=0 to a pytest run fetching fixtures!
    print(
        f"============================= test session starts =============================\n"
        f"platform darwin -- Python 3.11, pytest-8.3.2, session={session_id:02d}, turn={turn_idx}\n"
        f"collecting fixtures from remote artifact cache lock://127.0.0.1:9418/s{session_id:02d} ...",
        flush=True,
    )
    # Initial tiny setup CPU work (~0.08s)
    _ = burn_cpu(max(10_000, int(0.08 * iters_per_sec)), f"stall-init-{session_id}-{turn_idx}")

    # Block on pipe FD (zero CPU used, zero stdout emitted while waiting on remote lock/socket!)
    if stall_fd >= 0:
        try:
            while True:
                data = os.read(stall_fd, 16)
                if not data or b"GO" in data:
                    break
        except OSError:
            pass

    # Post-timeout diagnostic burst + fast fallback verification (~0.22s CPU)
    rem_cpu = max(0.15, cpu_sec - 0.08)
    for s, pct in [(1, 85), (2, 100)]:
        digest = burn_cpu(
            max(10_000, int((rem_cpu * iters_per_sec) / 2)),
            f"stall-tail-{session_id}-{turn_idx}-{s}",
        )
        print(
            f"[{pct:3d}%] Fixture lock timeout recovered via local cache fallback "
            f"(session={session_id:02d}, step={s}/2, replay_sig={digest})\n"
            f"  tests/runtime/test_fallback_{session_id:02d}_{s}.py::test_offline_fixture PASSED [{pct:3d}%]\n"
            f"  warning: remote artifact endpoint stalled before fallback engaged.",
            flush=True,
        )
    print("=================== 2 passed, 1 warning (recovered after stall) ===================", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["quick_lint", "pytest_suite", "heavy_compile", "stalled_io"])
    parser.add_argument("--iters-per-sec", type=int, required=True)
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--turn-idx", type=int, required=True)
    parser.add_argument("--cpu-sec", type=float, required=True)
    parser.add_argument("--stall-fd", type=int, default=-1)
    args = parser.parse_args()

    if args.kind == "quick_lint":
        run_quick_lint(args.iters_per_sec, args.session_id, args.turn_idx, args.cpu_sec)
    elif args.kind == "pytest_suite":
        run_pytest_suite(args.iters_per_sec, args.session_id, args.turn_idx, args.cpu_sec)
    elif args.kind == "heavy_compile":
        run_heavy_compile(args.iters_per_sec, args.session_id, args.turn_idx, args.cpu_sec)
    elif args.kind == "stalled_io":
        run_stalled_io(args.iters_per_sec, args.session_id, args.turn_idx, args.cpu_sec, args.stall_fd)


if __name__ == "__main__":
    main()
