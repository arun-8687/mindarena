"""
LeetCode-style code battle for MindArena.

Both players solve the same problem; an automated judge runs a hidden test
suite against each solution and picks a winner deterministically:
  1. Correctness — more hidden tests passed wins.
  2. Tiebreaker — if both pass all, faster runtime wins.
  3. Fallback — if still tied, count it a draw.

Solutions are executed in a subprocess with a strict timeout so a runaway or
infinite loop can't hang the server.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional

PYTHON = "python3"


@dataclass
class TestCase:
    input: str
    expected: str


@dataclass
class Problem:
    slug: str
    title: str
    difficulty: str            # Easy | Medium | Hard
    description: str
    function_name: str         # the function the model must implement
    signature: str             # e.g. "def two_sum(nums: list, target: int) -> list:"
    starter_code: str          # code template given to the model
    harness: str               # how to build the test input args from raw input
    tests: list[TestCase]      # hidden test cases (input = JSON args list, expected = JSON value)
    in_place: bool = False     # if True, the function mutates its first arg & returns None
    time_limit_s: float = 5.0


@dataclass
class BattleResult:
    """Result of judging one player's submission."""
    passed: int = 0
    total: int = 0
    runtime_ms: float = 0.0
    compile_error: str = ""
    failed_case: Optional[dict] = None   # {input, expected, got}
    timed_out: bool = False
    source: str = ""


# ---------------------------------------------------------------------------
# Problem library
# ---------------------------------------------------------------------------

def _tc(input_json: str, expected_json: str) -> TestCase:
    return TestCase(input_json, expected_json)


PROBLEMS: list[Problem] = [
    Problem(
        slug="two-sum",
        title="Two Sum",
        difficulty="Easy",
        description=(
            "Given an array of integers `nums` and an integer `target`, return the "
            "indices of the two numbers that add up to target. Each input has exactly "
            "one solution, and you may not use the same element twice. Return the "
            "indices in any order as a list."
        ),
        function_name="two_sum",
        signature="def two_sum(nums: list, target: int) -> list:",
        starter_code=(
            "def two_sum(nums: list, target: int) -> list:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc("[2,7,11,15], 9", "[0,1]"),
            _tc("[3,2,4], 6", "[1,2]"),
            _tc("[3,3], 6", "[0,1]"),
            _tc("[1,5,8,3], 9", "[0,2]"),
            _tc("[0,4,3,0], 0", "[0,3]"),
            _tc("[-3,4,3,90], 0", "[0,2]"),
        ],
    ),
    Problem(
        slug="palindrome-number",
        title="Palindrome Number",
        difficulty="Easy",
        description=(
            "Given an integer `x`, return True if it is a palindrome (reads the same "
            "forward and backward), False otherwise. Negative numbers are not palindromes."
        ),
        function_name="is_palindrome",
        signature="def is_palindrome(x: int) -> bool:",
        starter_code=(
            "def is_palindrome(x: int) -> bool:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc("121", "true"),
            _tc("-121", "false"),
            _tc("10", "false"),
            _tc("0", "true"),
            _tc("12321", "true"),
            _tc("1001", "true"),
        ],
    ),
    Problem(
        slug="fizzbuzz",
        title="FizzBuzz",
        difficulty="Easy",
        description=(
            "Given an integer `n`, return a list of strings answer (1-indexed) where: "
            "answer[i] == \"FizzBuzz\" if i is divisible by 3 and 5, \"Fizz\" if divisible "
            "by 3, \"Buzz\" if divisible by 5, otherwise str(i). Return the list of size n."
        ),
        function_name="fizz_buzz",
        signature="def fizz_buzz(n: int) -> list:",
        starter_code=(
            "def fizz_buzz(n: int) -> list:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc("3", '["1","2","Fizz"]'),
            _tc("5", '["1","2","Fizz","4","Buzz"]'),
            _tc("15", '["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]'),
            _tc("1", '["1"]'),
        ],
    ),
    Problem(
        slug="reverse-string",
        title="Reverse String",
        difficulty="Easy",
        description=(
            "Write a function that reverses a string in-place. The string is given as a "
            "list of characters `s`. Do not allocate extra space for another array; "
            "modify the input list in-place and return nothing."
        ),
        function_name="reverse_string",
        signature="def reverse_string(s: list) -> None:",
        starter_code=(
            "def reverse_string(s: list) -> None:\n"
            "    # modify s in-place, return nothing\n"
        ),
        harness="",
        in_place=True,
        tests=[
            _tc('["h","e","l","l","o"]', '["o","l","l","e","h"]'),
            _tc('["H","a","n","n","a","h"]', '["h","a","n","n","a","H"]'),
            _tc('["a"]', '["a"]'),
            _tc('["a","b"]', '["b","a"]'),
        ],
    ),
    Problem(
        slug="valid-parentheses",
        title="Valid Parentheses",
        difficulty="Easy",
        description=(
            "Given a string `s` containing just the characters '(', ')', '{', '}', '[' "
            "and ']', determine if the input string is valid. A string is valid if open "
            "brackets must be closed by the same type and brackets close in the correct order."
        ),
        function_name="is_valid",
        signature="def is_valid(s: str) -> bool:",
        starter_code=(
            "def is_valid(s: str) -> bool:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc('"()"', "true"),
            _tc('"()[]{}"', "true"),
            _tc('"(]"', "false"),
            _tc('"([)]"', "false"),
            _tc('"{[]}"', "true"),
            _tc('""', "true"),
            _tc('"("', "false"),
        ],
    ),
    Problem(
        slug="merge-sorted-arrays",
        title="Merge Sorted Arrays",
        difficulty="Easy",
        description=(
            "Given two sorted integer arrays `nums1` and `nums2`, merge them into a "
            "single sorted array. Return the merged sorted array."
        ),
        function_name="merge",
        signature="def merge(nums1: list, nums2: list) -> list:",
        starter_code=(
            "def merge(nums1: list, nums2: list) -> list:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc("[1,2,3], [2,5,6]", "[1,2,2,3,5,6]"),
            _tc("[], []", "[]"),
            _tc("[0], [0]", "[0,0]"),
            _tc("[1], []", "[1]"),
            _tc("[-1,0,2], [1,3,4]", "[-1,0,1,2,3,4]"),
        ],
    ),
    Problem(
        slug="max-subarray",
        title="Maximum Subarray",
        difficulty="Medium",
        description=(
            "Given an integer array `nums`, find the contiguous subarray (containing at "
            "least one number) which has the largest sum and return its sum."
        ),
        function_name="max_subarray",
        signature="def max_subarray(nums: list) -> int:",
        starter_code=(
            "def max_subarray(nums: list) -> int:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc("[-2,1,-3,4,-1,2,1,-5,4]", "6"),
            _tc("[1]", "1"),
            _tc("[5,4,-1,7,8]", "23"),
            _tc("[-1]", "-1"),
            _tc("[-2,-3,-1,-5]", "-1"),
        ],
    ),
    Problem(
        slug="contains-duplicate",
        title="Contains Duplicate",
        difficulty="Easy",
        description=(
            "Given an integer array `nums`, return True if any value appears at least "
            "twice in the array, and return False if every element is distinct."
        ),
        function_name="contains_duplicate",
        signature="def contains_duplicate(nums: list) -> bool:",
        starter_code=(
            "def contains_duplicate(nums: list) -> bool:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc("[1,2,3,1]", "true"),
            _tc("[1,2,3,4]", "false"),
            _tc("[1,1,1,3,3,4,3,2,4,2]", "true"),
            _tc("[]", "false"),
        ],
    ),
    Problem(
        slug="climbing-stairs",
        title="Climbing Stairs",
        difficulty="Easy",
        description=(
            "You are climbing a staircase. It takes n steps to reach the top. Each time "
            "you can either climb 1 or 2 steps. In how many distinct ways can you climb "
            "to the top?"
        ),
        function_name="climb_stairs",
        signature="def climb_stairs(n: int) -> int:",
        starter_code=(
            "def climb_stairs(n: int) -> int:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc("2", "2"),
            _tc("3", "3"),
            _tc("4", "5"),
            _tc("5", "8"),
            _tc("10", "89"),
        ],
    ),
    Problem(
        slug="longest-common-prefix",
        title="Longest Common Prefix",
        difficulty="Easy",
        description=(
            "Write a function to find the longest common prefix string amongst an array "
            "of strings. If there is no common prefix, return an empty string."
        ),
        function_name="longest_common_prefix",
        signature="def longest_common_prefix(strs: list) -> str:",
        starter_code=(
            "def longest_common_prefix(strs: list) -> str:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc('["flower","flow","flight"]', '"fl"'),
            _tc('["dog","racecar","car"]', '""'),
            _tc('["a"]', '"a"'),
            _tc('[""]', '""'),
            _tc('["interstellar","internet","interval"]', '"inter"'),
        ],
    ),
    Problem(
        slug="best-time-buy-sell",
        title="Best Time to Buy and Sell Stock",
        difficulty="Easy",
        description=(
            "You are given an array `prices` where prices[i] is the price of a given "
            "stock on the ith day. You want to maximize your profit by choosing a single "
            "day to buy and a different day in the future to sell. Return the maximum "
            "profit you can achieve. If you cannot achieve any profit, return 0."
        ),
        function_name="max_profit",
        signature="def max_profit(prices: list) -> int:",
        starter_code=(
            "def max_profit(prices: list) -> int:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc("[7,1,5,3,6,4]", "5"),
            _tc("[7,6,4,3,1]", "0"),
            _tc("[2,4,1]", "2"),
            _tc("[3,2,6,5,0,3]", "4"),
        ],
    ),
    Problem(
        slug="single-number",
        title="Single Number",
        difficulty="Easy",
        description=(
            "Given a non-empty array of integers `nums`, every element appears twice "
            "except for one. Find that single one. You must implement a solution with a "
            "linear runtime complexity and use only constant extra space."
        ),
        function_name="single_number",
        signature="def single_number(nums: list) -> int:",
        starter_code=(
            "def single_number(nums: list) -> int:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc("[2,2,1]", "1"),
            _tc("[4,1,2,1,2]", "4"),
            _tc("[1]", "1"),
            _tc("[7,3,5,3,5,7,9]", "9"),
        ],
    ),
    Problem(
        slug="move-zeroes",
        title="Move Zeroes",
        difficulty="Easy",
        description=(
            "Given an integer array `nums`, move all 0's to the end of it while "
            "maintaining the relative order of the non-zero elements. Modify the array "
            "in-place and return nothing."
        ),
        function_name="move_zeroes",
        signature="def move_zeroes(nums: list) -> None:",
        starter_code=(
            "def move_zeroes(nums: list) -> None:\n"
            "    # modify nums in-place, return nothing\n"
        ),
        harness="",
        in_place=True,
        tests=[
            _tc("[0,1,0,3,12]", "[1,3,12,0,0]"),
            _tc("[0]", "[0]"),
            _tc("[1,2,3]", "[1,2,3]"),
            _tc("[0,0,1]", "[1,0,0]"),
        ],
    ),
    Problem(
        slug="majority-element",
        title="Majority Element",
        difficulty="Easy",
        description=(
            "Given an array `nums` of size n, return the majority element. The majority "
            "element is the element that appears more than n/2 times. You may assume "
            "that the majority element always exists in the array."
        ),
        function_name="majority_element",
        signature="def majority_element(nums: list) -> int:",
        starter_code=(
            "def majority_element(nums: list) -> int:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc("[3,2,3]", "3"),
            _tc("[2,2,1,1,1,2,2]", "2"),
            _tc("[1]", "1"),
            _tc("[5,5,5,1,5]", "5"),
        ],
    ),
    Problem(
        slug="valid-anagram",
        title="Valid Anagram",
        difficulty="Easy",
        description=(
            "Given two strings `s` and `t`, return True if t is an anagram of s, and "
            "False otherwise. An anagram is a word formed by rearranging the letters of "
            "another word."
        ),
        function_name="is_anagram",
        signature="def is_anagram(s: str, t: str) -> bool:",
        starter_code=(
            "def is_anagram(s: str, t: str) -> bool:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc('"anagram", "nagaram"', "true"),
            _tc('"rat", "car"', "false"),
            _tc('"a", "a"', "true"),
            _tc('"ab", "ba"', "true"),
            _tc('"", ""', "true"),
        ],
    ),
    Problem(
        slug="squares-of-sorted-array",
        title="Squares of a Sorted Array",
        difficulty="Easy",
        description=(
            "Given an integer array `nums` sorted in non-decreasing order, return an "
            "array of the squares of each number sorted in non-decreasing order."
        ),
        function_name="sorted_squares",
        signature="def sorted_squares(nums: list) -> list:",
        starter_code=(
            "def sorted_squares(nums: list) -> list:\n"
            "    # your code here\n"
        ),
        harness="",
        tests=[
            _tc("[-4,-1,0,3,10]", "[0,1,9,16,100]"),
            _tc("[-7,-3,2,3,11]", "[4,9,9,49,121]"),
            _tc("[0]", "[0]"),
            _tc("[-1]", "[1]"),
        ],
    ),
]


def get_problem(slug: str) -> Optional[Problem]:
    for p in PROBLEMS:
        if p.slug == slug:
            return p
    return None


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _build_runner(problem: Problem, source: str) -> str:
    """Build a standalone Python script that runs all tests against `source`."""
    # Indent the model's source so it sits inside the runner scope.
    indented = textwrap.indent(source.strip(), "    ")
    test_lines = []
    for t in problem.tests:
        # Store expected raw (it's already JSON: "true", "[0,1]", '"fl"', "121").
        # json.loads in the runner yields the right Python type (bool/list/str/int).
        expected_raw = t.expected
        test_lines.append(
            f"    ({t.input!r}, {expected_raw!r}),"
        )
    tests = "\n".join(test_lines)

    in_place = problem.in_place

    # Build the per-test call. For in-place problems, we deep-copy the first
    # arg, call the function, and compare the mutated copy against expected.
    # The semicolon chain must be on its own line (not indented after `got =`).
    if in_place:
        call_block = (
            f"        _arg0 = copy.deepcopy(args[0])\n"
            f"        {problem.function_name}(*([_arg0] + list(args[1:])))\n"
            f"        got = _arg0"
        )
    else:
        call_block = f"        got = {problem.function_name}(*args)"

    return f"""\
import json, sys, time, traceback, copy

def _run():
    # ---- candidate solution ----
{indented}

    # ---- tests ----
    tests = [
{tests}
    ]
    start = time.perf_counter()
    results = []
    for raw_input, expected_json in tests:
        expected = json.loads(expected_json)
        args = eval('[' + raw_input + ']')
{call_block}
        if got is not None and hasattr(got, 'tolist'):
            got = got.tolist()
        results.append({{"input": raw_input, "expected": expected, "got": got}})
    elapsed_ms = (time.perf_counter() - start) * 1000
    out = [{{"input": r["input"], "expected": r["expected"], "got": r["got"]}} for r in results]
    print(json.dumps({{"ok": True, "elapsed_ms": elapsed_ms, "results": out}}))

try:
    _run()
except Exception as e:
    print(json.dumps({{"ok": False, "error": repr(e), "traceback": traceback.format_exc()}}))
"""


def judge(problem: Problem, source: str, time_limit: Optional[float] = None) -> BattleResult:
    """Run the judge on a submission. Returns a BattleResult."""
    tl = time_limit or problem.time_limit_s
    script = _build_runner(problem, source)
    result = BattleResult(total=len(problem.tests), source=source)
    try:
        proc = subprocess.run(
            [PYTHON, "-c", script],
            capture_output=True, text=True, timeout=tl,
        )
    except subprocess.TimeoutExpired:
        result.timed_out = True
        return result

    # Parse stdout
    out = (proc.stdout or "").strip()
    # Find the last JSON line (model may print debug)
    import re
    lines = out.splitlines()
    payload = None
    for ln in reversed(lines):
        if ln.startswith("{"):
            try:
                payload = json.loads(ln)
                break
            except json.JSONDecodeError:
                continue
    if payload is None or not payload.get("ok"):
        err = payload.get("error") if payload else (proc.stderr or "").strip()[:500]
        result.compile_error = err or "no output"
        return result

    result.runtime_ms = payload["elapsed_ms"]
    passed = 0
    failed = None
    for r in payload["results"]:
        try:
            if json.loads(r["got"]) == r["expected"] if isinstance(r["got"], str) and r["got"].startswith(("[","{")) else r["got"] == r["expected"]:
                passed += 1
            else:
                failed = r
                break
        except Exception:
            if r["got"] == r["expected"]:
                passed += 1
            else:
                failed = r
                break
    result.passed = passed
    result.failed_case = failed
    return result


def decide_winner(r0: BattleResult, r1: BattleResult) -> int:
    """Return 0 if player0 wins, 1 if player1 wins, -1 for a draw.

    Correctness first; then faster runtime; then draw.
    """
    def _score(r: BattleResult) -> int:
        # compile error => effectively 0 passed
        if r.compile_error or r.timed_out:
            return 0
        return r.passed

    s0, s1 = _score(r0), _score(r1)
    if s0 != s1:
        return 0 if s0 > s1 else 1
    if s0 == 0:
        return -1  # both failed everything
    # tie on correctness -> runtime
    if r0.runtime_ms < r1.runtime_ms:
        return 0
    if r1.runtime_ms < r0.runtime_ms:
        return 1
    return -1  # true draw
