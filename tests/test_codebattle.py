"""Checks for the code battle: the judge, the sandbox, and the problem banks.

The remote-bank checks need network and are skipped without it.

    python tests/test_codebattle.py
"""
from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import challenge as CH          # noqa: E402
import codebattle as CB         # noqa: E402
import problems as P            # noqa: E402
import problembank as PB        # noqa: E402

CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


TWO_SUM = P.get_problem("two-sum")

FAST_TWO_SUM = """
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
"""
SLOW_TWO_SUM = """
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
"""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@check
def every_test_is_counted_not_just_the_leading_run():
    """Regression: the judge stopped at the first failure, so 6/7 scored as 0."""
    src = """
def two_sum(nums, target):
    if nums == [2,7,11,15]:
        return [9,9]
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i]+nums[j] == target:
                return [i,j]
"""
    r = CB.judge(TWO_SUM, src, run_stress=False)
    assert r.passed == r.total - 1, f"expected all but one to pass, got {r.passed}/{r.total}"
    assert r.first_failure and r.first_failure.status == CB.FAIL


@check
def answers_the_statement_allows_are_accepted():
    """Two Sum says 'in either order'; comparing with == fails correct solutions."""
    src = """
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i]+nums[j] == target:
                return [j,i]
"""
    r = CB.judge(TWO_SUM, src, run_stress=False)
    assert r.passed == r.total, f"reversed-order answer scored {r.passed}/{r.total}"


@check
def comparison_modes_behave():
    assert CH.compare_answer(CH.UNORDERED, [1, 0], [0, 1])
    assert not CH.compare_answer(CH.UNORDERED, [1, 1], [0, 1])
    assert CH.compare_answer(CH.NESTED_UNORDERED, [["b", "a"], ["c"]], [["c"], ["a", "b"]])
    assert not CH.compare_answer(CH.NESTED_UNORDERED, [["a"]], [["a"], ["b"]])
    assert CH.compare_answer(CH.APPROX, 2.5000000001, 2.5)
    assert not CH.compare_answer(CH.APPROX, 2.6, 2.5)
    assert CH.compare_answer(CH.EXACT, [1, 2], [1, 2])
    assert not CH.compare_answer(CH.EXACT, [2, 1], [1, 2])
    # A mismatched type must not raise.
    assert not CH.compare_answer(CH.UNORDERED, None, [1])


@check
def stdout_comparison_is_whitespace_tolerant_but_not_sloppy():
    assert CH.compare_stdout("3\n1 2\n", "3\n1 2")
    assert CH.compare_stdout("YES", "Yes")
    assert CH.compare_stdout("1.0000001", "1.0")
    assert not CH.compare_stdout("1 2", "1 2 3")
    assert not CH.compare_stdout("4", "5")


@check
def scaled_inputs_separate_linear_from_quadratic():
    fast = CB.judge(TWO_SUM, FAST_TWO_SUM, seed=7)
    slow = CB.judge(TWO_SUM, SLOW_TWO_SUM, seed=7)
    assert fast.perfect and slow.perfect, "both should pass the small cases"
    assert fast.stress_status == CB.PASS, f"linear solution: {fast.stress_status}"
    assert slow.stress_status in (CB.TIMEOUT, CB.FAIL), f"quadratic solution: {slow.stress_status}"
    assert CB.decide_winner(slow, fast) == 1, "the linear solution should win"


@check
def winner_selection_prefers_correctness_then_scale_then_speed():
    def result(passed, total=7, stress=CB.SKIPPED, ms=100.0):
        r = CB.BattleResult(passed=passed, total=total, runtime_ms=ms)
        r.stress_status = stress
        r.cases = [CB.CaseResult(status=CB.PASS if i < passed else CB.FAIL)
                   for i in range(total)]
        return r
    assert CB.decide_winner(result(7), result(5)) == 0
    assert CB.decide_winner(result(3), result(6)) == 1
    assert CB.decide_winner(result(0), result(0)) == -1
    assert CB.decide_winner(result(7, stress=CB.PASS), result(7, stress=CB.TIMEOUT)) == 0
    assert CB.decide_winner(result(7, ms=50), result(7, ms=500)) == 0
    # Within timing noise, call it a draw rather than crowning a winner.
    assert CB.decide_winner(result(7, ms=100), result(7, ms=102)) == -1


@check
def a_submission_that_does_not_run_scores_zero_with_a_reason():
    r = CB.judge(TWO_SUM, "def two_sum(:\n", run_stress=False)
    assert r.passed == 0 and r.compile_error, "syntax error should be reported"
    r2 = CB.judge(TWO_SUM, "x = 1\n", run_stress=False)
    assert "two_sum" in r2.compile_error, r2.compile_error
    r3 = CB.judge(TWO_SUM, "", run_stress=False)
    assert r3.compile_error == "no solution submitted"


@check
def debug_prints_do_not_break_parsing():
    """Models leave prints in. That must not be read as part of the answer."""
    src = """
print("loaded")
def two_sum(nums, target):
    print("thinking about", target)
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
"""
    r = CB.judge(TWO_SUM, src, run_stress=False)
    assert r.passed == r.total, f"{r.passed}/{r.total} {r.compile_error}"
    assert "thinking about" in r.stdout_noise, r.stdout_noise


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------
@check
def sandboxed_code_cannot_read_the_servers_api_keys():
    import os
    os.environ["OLLAMA_API_KEY"] = "super-secret-value"
    src = """
import os
def two_sum(nums, target):
    return [os.environ.get('OLLAMA_API_KEY', 'NO-KEY')]
"""
    r = CB.judge(TWO_SUM, src, run_stress=False)
    got = r.cases[0].got if r.cases else ""
    assert "super-secret-value" not in got, f"the key leaked into the sandbox: {got}"
    assert "NO-KEY" in got, got


@check
def sandboxed_code_cannot_write_files():
    target = pathlib.Path("/tmp/mindarena-should-not-exist.txt")
    target.unlink(missing_ok=True)
    src = """
import pathlib
def two_sum(nums, target):
    try:
        pathlib.Path('/tmp/mindarena-should-not-exist.txt').write_text('escaped')
        return ['WROTE']
    except BaseException as e:
        return [type(e).__name__]
"""
    r = CB.judge(TWO_SUM, src, run_stress=False)
    assert "WROTE" not in (r.cases[0].got if r.cases else ""), "a write succeeded"
    assert target.read_text() == "" if target.exists() else True, "data was written"
    target.unlink(missing_ok=True)


@check
def runaway_code_is_killed_along_with_anything_it_forked():
    src = """
import os, time
def two_sum(nums, target):
    if os.fork() == 0:
        time.sleep(300)
        os._exit(0)
    while True:
        pass
"""
    started = time.time()
    r = CB.judge(TWO_SUM, src, run_stress=False)
    elapsed = time.time() - started
    assert r.timed_out, "the runaway solution was not stopped"
    assert elapsed < 60, f"took {elapsed:.0f}s to give up"
    time.sleep(1.0)
    needle = "_runner" + ".py"      # built at runtime so it cannot match this process
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    survivors = [l for l in ps.splitlines() if needle in l]
    assert not survivors, f"judge processes outlived the timeout: {survivors}"


@check
def memory_bombs_do_not_take_the_host_down():
    src = """
def two_sum(nums, target):
    blob = []
    while True:
        blob.append('x' * 10_000_000)
"""
    r = CB.judge(TWO_SUM, src, run_stress=False)
    assert r.passed == 0, "a memory bomb should not score"
    assert r.timed_out or r.compile_error or any(
        c.status in (CB.ERROR, CB.TIMEOUT) for c in r.cases), "expected it to be stopped"


# ---------------------------------------------------------------------------
# Problem banks
# ---------------------------------------------------------------------------
@check
def every_local_reference_satisfies_its_own_tests():
    bad = []
    for p in P.PROBLEMS:
        assert p.reference, f"{p.slug} has no reference implementation"
        assert p.stress, f"{p.slug} has no stress generator"
        for t in p.tests:
            args = copy.deepcopy(t.args)
            if p.in_place:
                p.reference(*args)
                got = args[0]
            else:
                got = p.reference(*args)
            if not CH.compare_answer(p.compare, got, t.expected):
                bad.append(f"{p.slug}[{t.label}]: expected {t.expected!r}, reference gave {got!r}")
    assert not bad, "\n      ".join(bad)


@check
def the_local_bank_is_not_all_easy():
    mix = {}
    for p in P.PROBLEMS:
        mix[p.difficulty] = mix.get(p.difficulty, 0) + 1
    assert mix.get("Medium", 0) >= 6, f"too few Medium problems: {mix}"
    assert mix.get("Hard", 0) >= 4, f"too few Hard problems: {mix}"
    assert len(P.PROBLEMS) >= 20, f"only {len(P.PROBLEMS)} problems"


@check
def selection_spreads_across_difficulties():
    import random
    bank = PB.ProblemBank()
    picked = bank.select(3, sources=("local",), rng=random.Random(4))
    assert len({p.difficulty for p in picked}) == 3, \
        f"a three-problem battle should span three tiers, got {[p.difficulty for p in picked]}"
    assert len({p.slug for p in bank.select(12, sources=('local',))}) == 12, "duplicates drawn"


@check
def stress_inputs_are_identical_for_both_players():
    a = CB.build_stress_cases(TWO_SUM, seed=99)
    b = CB.build_stress_cases(TWO_SUM, seed=99)
    assert a and a[0].args == b[0].args, "the same seed must produce the same input"
    c = CB.build_stress_cases(TWO_SUM, seed=100)
    assert a[0].args != c[0].args, "different seeds should differ"


@check
def ambiguous_and_interactive_problems_are_rejected():
    row = {
        "name": "1_A. Sample", "cf_rating": 1500,
        "description": "Do a thing. If there are multiple answers, print any of them.",
        "public_tests": {"input": ["1"], "output": ["1"]},
        "private_tests": {"input": ["2"] * 10, "output": ["2"] * 10},
    }
    import random
    assert PB._row_to_challenge(row, random.Random(0)) is None, \
        "a problem with multiple valid answers must not be admitted"
    row["description"] = "This is an interactive problem. Ask queries."
    assert PB._row_to_challenge(row, random.Random(0)) is None
    row["description"] = "Compute the unique answer and print it."
    assert PB._row_to_challenge(row, random.Random(0)) is not None
    row["private_tests"] = {"input": [], "output": []}
    assert PB._row_to_challenge(row, random.Random(0)) is None, "too few tests to judge on"


@check
def rating_maps_to_a_difficulty_tier():
    assert PB.difficulty_for_rating(800) == "Easy"
    assert PB.difficulty_for_rating(1500) == "Medium"
    assert PB.difficulty_for_rating(1800) == "Hard"
    assert PB.difficulty_for_rating(2600) == "Expert"


@check
def the_cache_survives_a_round_trip():
    ch = CH.Challenge(
        slug="probe", title="Probe", difficulty="Hard", rating=1900,
        description="statement", kind=CH.STDIO, source="code_contests",
        tests=[CH.Case(label="example 1", stdin="1\n", stdout="2\n", hidden=False)])
    restored = PB._from_cache(PB._to_cache([ch]))
    assert len(restored) == 1
    assert restored[0].slug == "probe" and restored[0].rating == 1900
    assert restored[0].tests[0].stdin == "1\n" and restored[0].examples


@check
def hidden_test_data_never_reaches_the_browser():
    ch = P.get_problem("two-sum")
    blob = json.dumps(ch.public_dict())
    hidden = [t for t in ch.tests if t.hidden]
    assert hidden, "the fixture should have hidden tests"
    leaked = [t for t in hidden if json.dumps(t.expected) in blob and json.dumps(t.args) in blob]
    assert not leaked, f"{len(leaked)} hidden cases were exposed to the UI"
    assert ch.public_dict()["test_count"] == len(ch.tests)


# ---------------------------------------------------------------------------
# Remote bank (needs network)
# ---------------------------------------------------------------------------
def _online() -> bool:
    try:
        PB._get_json({"dataset": PB.DATASET, "config": "default", "split": "test",
                      "where": '"cf_rating">=3000', "columns": "name",
                      "offset": 0, "length": 1}, 25)
        return True
    except Exception:
        return False


@check
def remote_problems_load_and_are_self_validated():
    if not _online():
        print("      (skipped: no network)")
        return
    got = PB.fetch_code_contests(target=2, min_rating=1300, max_rating=1900,
                                 validate=True, timeout=90)
    if not got:
        print("      (skipped: archive returned nothing)")
        return
    for c in got:
        assert c.kind == CH.STDIO and c.source == "code_contests"
        assert len(c.tests) >= PB.MIN_TESTS, f"{c.slug} has {len(c.tests)} tests"
        assert c.rating >= 1300 and c.difficulty in CH.DIFFICULTIES
        assert c.examples, f"{c.slug} has no worked example to show"
        assert c.description.strip()


def main() -> int:
    failed = 0
    for fn in CHECKS:
        name = fn.__name__.replace("_", " ")
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}\n      {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}\n      {type(e).__name__}: {e}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
