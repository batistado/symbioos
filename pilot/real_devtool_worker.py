#!/usr/bin/env python3
"""Phase 1 Real Developer-Tool Sandbox Driver for SymbioOS (POSIX Process Groups).

Spawns genuine developer CLI tools inside the caller's POSIX process group (pgid)
so os.killpg(pgid, SIGSTOP / SIGCONT) schedules the entire process tree (including
child pytest runners, git subprocesses, and /usr/bin/clang++ C++20 compiler stages):

  - quick_lint:    Real Python py_compile + AST static check + `git status --short`
  - pytest_suite:  Real `python -m pytest -v` over generated multi-file test suites
  - heavy_compile: Real `/usr/bin/clang++ -O3 -std=c++20 -c` multi-TU object builds
                   (strictly -c object compilation inside workspace scratch; zero /tmp or
                   unsigned binary execution)
  - stalled_io:    Ambiguous pytest remote-fixture stall (socket/pipe block) + fallback pytest
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile


GIT_BIN = "/usr/local/git/current/bin/git" if os.path.exists("/usr/local/git/current/bin/git") else "git"
CLANGXX_BIN = "/usr/bin/clang++"


def make_scratch_dir(base_root: str, prefix: str) -> str:
    os.makedirs(base_root, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base_root)


def run_quick_lint(scratch_root: str, session_id: int, turn_idx: int) -> None:
    wd = make_scratch_dir(scratch_root, f"lint_s{session_id:02d}_t{turn_idx}_")
    try:
        print(f"symbio-lint check workspace/mod_{session_id:02d}_t{turn_idx} --ast --pycompile", flush=True)
        for idx in range(1, 5):
            py_file = os.path.join(wd, f"service_handler_{idx}.py")
            with open(py_file, "w", encoding="utf-8") as f:
                f.write(
                    f"import hashlib\n"
                    f"class Handler{idx}:\n"
                    f"    def compute(self, payload: bytes) -> str:\n"
                    f"        h = payload\n"
                    f"        for _ in range(1024):\n"
                    f"            h = hashlib.sha256(h).digest()\n"
                    f"        return h.hex()\n"
                )
            py_compile.compile(py_file, doraise=True)
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=py_file)
            pct = idx * 25
            print(
                f"[{pct:3d}%] py_compile + AST verified service_handler_{idx}.py "
                f"(nodes={sum(1 for _ in ast.walk(tree))}, errors=0)",
                flush=True,
            )
        # Run a quick git status/version check for authentic VCS subprocess telemetry
        subprocess.run(
            [GIT_BIN, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        print("All static & bytecode checks passed (4 modules verified, 0 warnings).", flush=True)
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def run_pytest_suite(scratch_root: str, session_id: int, turn_idx: int, iters_per_test: int = 85_000) -> None:
    wd = make_scratch_dir(scratch_root, f"pytest_s{session_id:02d}_t{turn_idx}_")
    try:
        # Generate 5 test modules x 4 tests = 20 real pytest test cases!
        test_files = []
        for m in range(1, 6):
            tf = os.path.join(wd, f"test_kv_engine_{m:02d}.py")
            test_files.append(tf)
            funcs = ["import hashlib\n"]
            for k in range(1, 5):
                funcs.append(
                    f"def test_sandbox_s{session_id:02d}_m{m:02d}_case_{k:02d}_residency_invariant():\n"
                    f"    h = b'seed-{session_id}-{turn_idx}-{m}-{k}'\n"
                    f"    for _ in range({iters_per_test}):\n"
                    f"        h = hashlib.sha256(h).digest()\n"
                    f"    assert len(h) == 32\n"
                )
            with open(tf, "w", encoding="utf-8") as f:
                f.write("\n".join(funcs))

        print(f"[  0%] Launching pytest 9.2 runner on 20 cases (session={session_id:02d}, turn={turn_idx})", flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [
            sys.executable, "-u", "-m", "pytest",
            "-v", "--tb=short", "--no-header", "-o", "cache_dir=.p_cache",
            *[os.path.basename(tf) for tf in test_files],
        ]
        subprocess.run(cmd, cwd=wd, stdout=sys.stdout, stderr=subprocess.STDOUT, env=env, check=False)
    finally:
        shutil.rmtree(wd, ignore_errors=True)


CPP_TEMPLATE = """
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <vector>

namespace symbio_tu_{tu_idx} {{

template <int N, typename T>
struct StencilBlock {{
    std::array<T, N> weights;
    T evaluate(T x) const {{
        T acc = static_cast<T>(0);
        for (int i = 0; i < N; ++i) {{
            acc += weights[i] * std::sin(x * static_cast<T>(i + 1)) + std::cos(x - static_cast<T>(i));
        }}
        return acc;
    }}
}};

double compile_heavy_unit_{tu_idx}(int n_elems) {{
    std::vector<double> buf(n_elems);
    StencilBlock<64, double> s1{{}};
    StencilBlock<128, double> s2{{}};
    StencilBlock<256, double> s3{{}};
    for (int i = 0; i < n_elems; ++i) {{
        s1.weights[i % 64] = 0.01 * i;
        s2.weights[i % 128] = 0.02 * i;
        s3.weights[i % 256] = 0.03 * i;
        buf[i] = s1.evaluate(0.1 * i) + s2.evaluate(0.05 * i) + s3.evaluate(0.02 * i);
    }}
    std::sort(buf.begin(), buf.end());
    return std::accumulate(buf.begin(), buf.end(), 0.0);
}}

}} // namespace symbio_tu_{tu_idx}
"""


def run_heavy_compile(scratch_root: str, session_id: int, turn_idx: int, num_tus: int = 5) -> None:
    wd = make_scratch_dir(scratch_root, f"clang_s{session_id:02d}_t{turn_idx}_")
    try:
        print(
            f"[  0%] Configuring C++20 native extension target=symbio_kern_s{session_id:02d}_t{turn_idx} "
            f"(compiler={CLANGXX_BIN})",
            flush=True,
        )
        for tu in range(1, num_tus + 1):
            cpp_file = os.path.join(wd, f"tu_kernel_{tu:02d}.cpp")
            obj_file = os.path.join(wd, f"tu_kernel_{tu:02d}.o")
            with open(cpp_file, "w", encoding="utf-8") as f:
                f.write(CPP_TEMPLATE.format(tu_idx=tu))

            # Strictly `-c` object compilation (never links or executes unsigned Mach-O binaries!)
            env = os.environ.copy()
            env["TMPDIR"] = wd
            subprocess.run(
                [CLANGXX_BIN, "-O3", "-std=c++20", "-Wall", "-Wextra", "-fPIC", "-c", cpp_file, "-o", obj_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                check=True,
            )
            sz = os.path.getsize(obj_file) if os.path.exists(obj_file) else 0
            pct = int((tu * 100) // num_tus)
            print(
                f"[{pct:3d}%] Building CXX object CMakeFiles/symbio_s{session_id:02d}.dir/tu_kernel_{tu:02d}.cpp.o "
                f"(-O3 -std=c++20, obj_bytes={sz})\n"
                f"  [clang++-stat] template_instantiations=3x(64,128,256) vectorize=enabled status=ok",
                flush=True,
            )
        print(
            f"[100%] Completed object archive verification for target symbio_kern_s{session_id:02d} ({num_tus} TUs).",
            flush=True,
        )
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def run_stalled_io(scratch_root: str, session_id: int, turn_idx: int, stall_fd: int) -> None:
    wd = make_scratch_dir(scratch_root, f"stall_s{session_id:02d}_t{turn_idx}_")
    try:
        print(
            f"collecting ... tests/runtime/test_remote_fixture_s{session_id:02d}.py\n"
            f"connecting to artifact fixture lock endpoint 127.0.0.1:9418/session_{session_id:02d} ...",
            flush=True,
        )
        h = b"init"
        for _ in range(60_000):
            h = hashlib.sha256(h).digest()

        if stall_fd >= 0:
            try:
                while True:
                    data = os.read(stall_fd, 16)
                    if not data or b"GO" in data:
                        break
            except OSError:
                pass

        # Run a quick 2-test real pytest fallback recovery!
        tf = os.path.join(wd, "test_fallback_recovery.py")
        with open(tf, "w", encoding="utf-8") as f:
            f.write(
                "def test_cached_fallback_fixture_a():\n    assert 2 + 2 == 4\n\n"
                "def test_cached_fallback_fixture_b():\n    assert 'kv'.upper() == 'KV'\n"
            )
        print("[ 80%] Remote lock timed out; running offline fixture fallback via pytest...", flush=True)
        subprocess.run(
            [sys.executable, "-u", "-m", "pytest", "-v", "--tb=short", "--no-header", "-o", "cache_dir=" + os.path.join(wd, ".pc"), tf],
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["quick_lint", "pytest_suite", "heavy_compile", "stalled_io"])
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--turn-idx", type=int, required=True)
    parser.add_argument("--stall-fd", type=int, default=-1)
    parser.add_argument("--pytest-iters", type=int, default=85_000)
    parser.add_argument("--clang-tus", type=int, default=5)
    args = parser.parse_args()

    if args.kind == "quick_lint":
        run_quick_lint(args.scratch_root, args.session_id, args.turn_idx)
    elif args.kind == "pytest_suite":
        run_pytest_suite(args.scratch_root, args.session_id, args.turn_idx, args.pytest_iters)
    elif args.kind == "heavy_compile":
        run_heavy_compile(args.scratch_root, args.session_id, args.turn_idx, args.clang_tus)
    elif args.kind == "stalled_io":
        run_stalled_io(args.scratch_root, args.session_id, args.turn_idx, args.stall_fd)


if __name__ == "__main__":
    main()
