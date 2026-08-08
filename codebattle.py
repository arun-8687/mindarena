"""
The code-battle judge.

Both players get the same challenge; each submission is executed and scored, and
a winner is picked deterministically:

  1. Correctness   — more hidden tests passed wins.
  2. Scale         — if both are perfect, the one that survives the stress tier wins.
  3. Speed         — still tied, the faster total runtime wins.
  4. Draw          — otherwise.

Model-authored code is hostile input. It runs in a subprocess that:
  * has no access to the parent environment (so API keys are not readable),
  * runs in a throwaway working directory, never the repo,
  * carries CPU, address-space and file-size rlimits,
  * runs in its own session, so a timeout kills the whole process group rather
    than leaving forked grandchildren running.

None of this is a security boundary against a determined attacker — that needs a
container or a VM. It is enough to contain the realistic failure modes: runaway
loops, memory bombs, accidental file writes, and casual exfiltration of the keys
sitting in the server's environment.
"""
from __future__ import annotations

import json
import os
import random
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from challenge import (
    Challenge, Case, FUNCTION, STDIO,
    compare_answer, compare_stdout,
)
# Re-exported so existing imports of the local bank keep working.
from problems import PROBLEMS, get_problem, sample_problems  # noqa: F401

PYTHON = sys.executable or "python3"

# Status values for a single case.
PASS, FAIL, ERROR, TIMEOUT, SKIPPED = "pass", "fail", "error", "timeout", "skipped"


@dataclass
class CaseResult:
    label: str = ""
    status: str = PASS
    ms: float = 0.0
    expected: str = ""
    got: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == PASS


@dataclass
class BattleResult:
    """How one player's submission fared."""
    passed: int = 0
    total: int = 0
    cases: list[CaseResult] = field(default_factory=list)
    runtime_ms: float = 0.0
    compile_error: str = ""
    timed_out: bool = False
    stress_status: str = SKIPPED     # pass | fail | timeout | error | skipped
    stress_ms: float = 0.0
    stress_detail: str = ""
    source: str = ""
    stdout_noise: str = ""           # anything the solution printed while being judged

    @property
    def perfect(self) -> bool:
        return self.total > 0 and self.passed == self.total and not self.compile_error

    @property
    def first_failure(self) -> Optional[CaseResult]:
        return next((c for c in self.cases if not c.ok), None)

    def public_dict(self) -> dict:
        return {
            "passed": self.passed, "total": self.total,
            "runtime_ms": round(self.runtime_ms, 2),
            "compile_error": self.compile_error[:2000],
            "timed_out": self.timed_out,
            "stress_status": self.stress_status,
            "stress_ms": round(self.stress_ms, 2),
            "stress_detail": self.stress_detail[:400],
            "stdout_noise": self.stdout_noise[:400],
            "cases": [
                {"label": c.label, "status": c.status, "ms": round(c.ms, 2),
                 "expected": c.expected[:400], "got": c.got[:400], "detail": c.detail[:400]}
                for c in self.cases
            ],
        }


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------
_SAFE_ENV = {
    "PATH": "/usr/bin:/bin",
    "LC_ALL": "C.UTF-8",
    "LANG": "C.UTF-8",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",      # so repeated runs time consistently
}


def _limiter(cpu_seconds: int, memory_mb: int):
    """Applied in the child between fork and exec."""
    def apply():
        os.setsid()                                   # own process group
        soft = max(1, int(cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + 1))
        mem = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except (ValueError, OSError):
            pass                                      # not enforceable everywhere
        # No file writes at all. Solutions communicate through stdout, which is a
        # pipe and so unaffected; this stops a submission littering the host or
        # filling the disk. Bytecode writing is already off via -B.
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


@dataclass
class RunOutcome:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False
    ms: float = 0.0


def _run_sandboxed(args: list[str], cwd: str, timeout: float,
                   stdin_text: str = "", memory_mb: int = 1024) -> RunOutcome:
    """Run a command under the sandbox, killing the whole group on timeout."""
    started = time.perf_counter()
    try:
        proc = subprocess.Popen(
            args, cwd=cwd, env=dict(_SAFE_ENV, HOME=cwd, TMPDIR=cwd),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, preexec_fn=_limiter(int(timeout) + 1, memory_mb),
        )
    except OSError as e:
        return RunOutcome(stderr=f"could not start judge process: {e}", returncode=-1)

    try:
        out, err = proc.communicate(stdin_text, timeout=timeout)
        return RunOutcome(out, err, proc.returncode, False,
                          (time.perf_counter() - started) * 1000)
    except subprocess.TimeoutExpired:
        # Kill the session, not just the direct child: a solution that forked
        # would otherwise leave workers running after we gave up on it.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return RunOutcome(out, err, -9, True, (time.perf_counter() - started) * 1000)


# ---------------------------------------------------------------------------
# The in-sandbox runner for function challenges
# ---------------------------------------------------------------------------
_FUNCTION_RUNNER = r'''
import io, json, os, sys, copy, contextlib

# Isolated mode (-I) deliberately drops the script directory from sys.path, so
# put the sandbox directory back to make solution.py importable — and only that.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    payload = json.load(sys.stdin)
    noise = io.StringIO()
    try:
        with contextlib.redirect_stdout(noise):
            import solution
    except BaseException as e:
        json.dump({"ok": False, "phase": "import",
                   "error": f"{type(e).__name__}: {e}"}, sys.stderr)
        return 1

    fn = getattr(solution, payload["function_name"], None)
    if not callable(fn):
        json.dump({"ok": False, "phase": "import",
                   "error": "no function named " + payload["function_name"]}, sys.stderr)
        return 1

    sys.setrecursionlimit(payload.get("recursion_limit", 30000))
    in_place = payload["in_place"]
    results = []
    with contextlib.redirect_stdout(noise):
        for case in payload["cases"]:
            args = case["args"]
            import time as _t
            t0 = _t.perf_counter()
            try:
                if in_place:
                    first = copy.deepcopy(args[0])
                    fn(*([first] + list(args[1:])))
                    got, err = first, None
                else:
                    got, err = fn(*copy.deepcopy(args)), None
            except BaseException as e:
                got, err = None, f"{type(e).__name__}: {e}"
            ms = (_t.perf_counter() - t0) * 1000
            if err is None:
                try:
                    json.dumps(got)
                except (TypeError, ValueError):
                    got = repr(got)
            results.append({"got": got, "error": err, "ms": ms})

    json.dump({"ok": True, "results": results, "noise": noise.getvalue()[:2000]}, sys.stdout)
    return 0

sys.exit(main())
'''


def _write_sandbox(tmp: str, source: str, runner: str = "") -> None:
    with open(os.path.join(tmp, "solution.py"), "w", encoding="utf-8") as fh:
        fh.write(source)
    if runner:
        with open(os.path.join(tmp, "_runner.py"), "w", encoding="utf-8") as fh:
            fh.write(runner)


def _judge_function(ch: Challenge, source: str, cases: list[Case],
                    limit: float) -> tuple[list[CaseResult], float, str, bool, str]:
    """Run function-kind cases. Returns (results, ms, compile_error, timed_out, noise)."""
    tmp = tempfile.mkdtemp(prefix="mindarena-judge-")
    try:
        _write_sandbox(tmp, source, _FUNCTION_RUNNER)
        payload = json.dumps({
            "function_name": ch.function_name,
            "in_place": ch.in_place,
            "cases": [{"args": c.args or []} for c in cases],
        })
        run = _run_sandboxed([PYTHON, "-I", "-B", "_runner.py"], tmp, limit,
                             payload, ch.memory_mb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if run.timed_out:
        return ([CaseResult(c.label, TIMEOUT, detail="exceeded the time limit") for c in cases],
                run.ms, "", True, "")

    try:
        data = json.loads(run.stdout.strip() or "{}")
    except json.JSONDecodeError:
        data = {}
    if not data.get("ok"):
        err = ""
        try:
            err = (json.loads(run.stderr.strip() or "{}") or {}).get("error", "")
        except json.JSONDecodeError:
            pass
        err = err or run.stderr.strip()[:800] or "the solution produced no usable output"
        return [], run.ms, err, False, ""

    results = []
    for case, raw in zip(cases, data.get("results", [])):
        if raw.get("error"):
            results.append(CaseResult(case.label, ERROR, raw.get("ms", 0.0),
                                      _brief(case.expected), "", raw["error"]))
            continue
        got = raw.get("got")
        ok = compare_answer(ch.compare, got, case.expected)
        results.append(CaseResult(
            case.label, PASS if ok else FAIL, raw.get("ms", 0.0),
            _brief(case.expected), _brief(got)))
    return results, run.ms, "", False, data.get("noise", "")


def _judge_stdio(ch: Challenge, source: str, cases: list[Case],
                 limit: float) -> tuple[list[CaseResult], float, str, bool, str]:
    """Run stdio-kind cases, one process per case.

    Bails out early after repeated timeouts: a solution too slow for the first
    few cases is too slow for the rest, and grinding through all of them just
    makes the round take minutes.
    """
    tmp = tempfile.mkdtemp(prefix="mindarena-judge-")
    results: list[CaseResult] = []
    total_ms = 0.0
    compile_error = ""
    consecutive_timeouts = 0
    try:
        _write_sandbox(tmp, source)
        # A syntax error should be reported once, not once per test case.
        check = _run_sandboxed([PYTHON, "-I", "-B", "-c",
                                "import ast,sys;ast.parse(open('solution.py').read())"],
                               tmp, 10.0, "", ch.memory_mb)
        if check.returncode != 0:
            return [], 0.0, (check.stderr.strip()[-800:] or "solution.py does not parse"), False, ""

        for case in cases:
            if consecutive_timeouts >= 3:
                results.append(CaseResult(case.label, SKIPPED,
                                          detail="skipped after repeated timeouts"))
                continue
            run = _run_sandboxed([PYTHON, "-I", "-B", "solution.py"], tmp,
                                 ch.time_limit_s, case.stdin, ch.memory_mb)
            total_ms += run.ms
            if run.timed_out:
                consecutive_timeouts += 1
                results.append(CaseResult(case.label, TIMEOUT, run.ms,
                                          _brief(case.stdout), "",
                                          f"exceeded {ch.time_limit_s:g}s"))
                continue
            consecutive_timeouts = 0
            if run.returncode != 0:
                results.append(CaseResult(case.label, ERROR, run.ms, _brief(case.stdout), "",
                                          run.stderr.strip()[-400:] or f"exit code {run.returncode}"))
                continue
            ok = compare_stdout(run.stdout, case.stdout)
            results.append(CaseResult(case.label, PASS if ok else FAIL, run.ms,
                                      _brief(case.stdout), _brief(run.stdout)))
            if total_ms > limit * 1000:
                for rest in cases[len(results):]:
                    results.append(CaseResult(rest.label, SKIPPED,
                                              detail="skipped: overall budget spent"))
                break
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results, total_ms, compile_error, False, ""


def _brief(value: Any, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=repr)
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def build_stress_cases(ch: Challenge, seed: int) -> list[Case]:
    """Generate the scaled inputs and their expected answers.

    The reference solution runs here in the parent, never in the sandbox, so a
    submission cannot tamper with what it is being checked against. The seed is
    shared by both players so they face byte-identical inputs.
    """
    if not ch.stress_enabled:
        return []
    rng = random.Random(seed)
    cases = []
    for i, args in enumerate(ch.stress(rng)):
        import copy as _copy
        ref_args = _copy.deepcopy(args)
        if ch.in_place:
            ch.reference(*ref_args)
            expected = ref_args[0]
        else:
            expected = ch.reference(*ref_args)
        cases.append(Case(label=f"scaled input #{i + 1}", args=args, expected=expected))
    return cases


def judge(ch: Challenge, source: str, seed: int = 0,
          run_stress: bool = True) -> BattleResult:
    """Execute and score one submission."""
    result = BattleResult(total=len(ch.tests), source=source)
    if not source.strip():
        result.compile_error = "no solution submitted"
        return result

    runner = _judge_function if ch.kind == FUNCTION else _judge_stdio
    cases, ms, err, timed_out, noise = runner(ch, source, ch.tests, ch.time_limit_s * 4)
    result.cases = cases
    result.runtime_ms = ms
    result.compile_error = err
    result.timed_out = timed_out
    result.stdout_noise = noise
    result.passed = sum(1 for c in cases if c.ok)

    # The stress tier only matters between two otherwise-perfect solutions, and
    # failing it never costs correctness points — it is a tiebreak, not a gate.
    if run_stress and result.perfect and ch.stress_enabled:
        stress_cases = build_stress_cases(ch, seed)
        if stress_cases:
            s_cases, s_ms, s_err, s_timeout, _ = _judge_function(
                ch, source, stress_cases, ch.stress_limit_s)
            result.stress_ms = s_ms
            if s_timeout:
                result.stress_status = TIMEOUT
                result.stress_detail = f"did not finish scaled input within {ch.stress_limit_s:g}s"
            elif s_err:
                result.stress_status = ERROR
                result.stress_detail = s_err[:300]
            elif all(c.ok for c in s_cases):
                result.stress_status = PASS
            else:
                result.stress_status = FAIL
                bad = next((c for c in s_cases if not c.ok), None)
                result.stress_detail = (bad.detail or f"wrong answer on {bad.label}") if bad else "failed"
    return result


def decide_winner(r0: BattleResult, r1: BattleResult) -> int:
    """0 if player 0 wins, 1 if player 1 wins, -1 for a draw."""
    if r0.passed != r1.passed:
        return 0 if r0.passed > r1.passed else 1
    if r0.passed == 0:
        return -1                                  # neither solved anything
    rank = {PASS: 2, SKIPPED: 1, FAIL: 0, TIMEOUT: 0, ERROR: 0}
    s0, s1 = rank.get(r0.stress_status, 0), rank.get(r1.stress_status, 0)
    if s0 != s1:
        return 0 if s0 > s1 else 1
    t0 = r0.runtime_ms + r0.stress_ms
    t1 = r1.runtime_ms + r1.stress_ms
    # Ignore differences under 5%: below that we are timing scheduler noise.
    if t0 and t1 and abs(t0 - t1) / max(t0, t1) > 0.05:
        return 0 if t0 < t1 else 1
    return -1


def verdict_line(r: BattleResult) -> str:
    """One-line human summary, used in the battle log."""
    if r.compile_error:
        return f"{r.passed}/{r.total} — {r.compile_error.splitlines()[0][:80]}"
    if r.timed_out:
        return f"{r.passed}/{r.total} — timed out"
    bits = [f"{r.passed}/{r.total} tests", f"{r.runtime_ms:.0f}ms"]
    if r.stress_status != SKIPPED:
        bits.append(f"scaled: {r.stress_status}")
    return " · ".join(bits)
