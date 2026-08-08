"""
The curated local problem bank.

This is the offline tier: it always works, needs no network, and every problem
here carries a reference implementation plus a stress generator so the speed
tiebreak has something real to measure. Harder, real-world problems are pulled
from remote archives at run time — see problembank.py.

Each problem carries three things the judge needs:

  tests      Worked examples plus hidden edge cases. Small, fast, and the only
             thing that decides correctness.
  reference  A known-correct implementation, run in the *parent* process (never
             in the sandbox) to compute expected answers for generated inputs.
  stress     A deterministic generator of large inputs. This is what makes the
             speed tiebreak mean something: on six-element lists an O(n^2)
             solution and an O(n) solution are indistinguishable.
"""
from __future__ import annotations

import json
import random
from typing import Optional

from challenge import (
    Challenge, Case, EXACT, UNORDERED, NESTED_UNORDERED, APPROX,
)

# Kept as module-level aliases so the bank below reads the same as before.
Problem = Challenge


def _tc(args_json: str, expected_json: str, label: str = "", shown: bool = False) -> Case:
    """Build a case from JSON text, so the bank stays readable.

    `args_json` is the argument list without brackets: "[2,7,11,15], 9".
    """
    return Case(label=label, args=json.loads(f"[{args_json}]"),
                expected=json.loads(expected_json), hidden=not shown)


# ---------------------------------------------------------------------------
# Reference implementations
#
# These run in the parent process to produce expected answers for generated
# inputs. They are written for obvious correctness, not for speed.
# ---------------------------------------------------------------------------
def _ref_two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []

def _ref_max_subarray(nums):
    best = cur = nums[0]
    for n in nums[1:]:
        cur = max(n, cur + n)
        best = max(best, cur)
    return best

def _ref_contains_duplicate(nums):
    return len(set(nums)) != len(nums)

def _ref_max_profit(prices):
    best, low = 0, float("inf")
    for p in prices:
        low = min(low, p)
        best = max(best, p - low)
    return best

def _ref_single_number(nums):
    out = 0
    for n in nums:
        out ^= n
    return out

def _ref_sorted_squares(nums):
    return sorted(n * n for n in nums)

def _ref_move_zeroes(nums):
    k = 0
    for i, n in enumerate(nums):
        if n != 0:
            nums[k], nums[i] = nums[i], nums[k]
            k += 1

def _ref_majority_element(nums):
    count, cand = 0, None
    for n in nums:
        if count == 0:
            cand = n
        count += 1 if n == cand else -1
    return cand

def _ref_product_except_self(nums):
    n = len(nums)
    out = [1] * n
    left = 1
    for i in range(n):
        out[i] = left
        left *= nums[i]
    right = 1
    for i in range(n - 1, -1, -1):
        out[i] *= right
        right *= nums[i]
    return out

def _ref_length_of_longest_substring(s):
    last, best, start = {}, 0, 0
    for i, ch in enumerate(s):
        if ch in last and last[ch] >= start:
            start = last[ch] + 1
        last[ch] = i
        best = max(best, i - start + 1)
    return best

def _ref_group_anagrams(strs):
    buckets = {}
    for w in strs:
        buckets.setdefault("".join(sorted(w)), []).append(w)
    return list(buckets.values())

def _ref_coin_change(coins, amount):
    INF = float("inf")
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return -1 if dp[amount] == INF else dp[amount]

def _ref_length_of_lis(nums):
    import bisect
    tails = []
    for n in nums:
        i = bisect.bisect_left(tails, n)
        if i == len(tails):
            tails.append(n)
        else:
            tails[i] = n
    return len(tails)

def _ref_three_sum(nums):
    nums = sorted(nums)
    out, n = [], len(nums)
    for i in range(n - 2):
        if i and nums[i] == nums[i - 1]:
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            total = nums[i] + nums[lo] + nums[hi]
            if total < 0:
                lo += 1
            elif total > 0:
                hi -= 1
            else:
                out.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                while lo < hi and nums[lo] == nums[lo - 1]:
                    lo += 1
                hi -= 1
    return out

def _ref_search_insert(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

def _ref_word_break(s, words):
    ws = set(words)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in ws:
                dp[i] = True
                break
    return dp[n]

def _ref_can_finish(num_courses, prereqs):
    from collections import deque
    adj = [[] for _ in range(num_courses)]
    indeg = [0] * num_courses
    for a, b in prereqs:
        adj[b].append(a)
        indeg[a] += 1
    q = deque(i for i in range(num_courses) if indeg[i] == 0)
    seen = 0
    while q:
        node = q.popleft()
        seen += 1
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    return seen == num_courses

def _ref_trap(height):
    if not height:
        return 0
    lo, hi = 0, len(height) - 1
    lmax, rmax, out = height[lo], height[hi], 0
    while lo < hi:
        if lmax <= rmax:
            lo += 1
            lmax = max(lmax, height[lo])
            out += lmax - height[lo]
        else:
            hi -= 1
            rmax = max(rmax, height[hi])
            out += rmax - height[hi]
    return out

def _ref_min_distance(a, b):
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[n]

def _ref_find_median(a, b):
    merged = sorted(a + b)
    n = len(merged)
    if n == 0:
        return 0.0
    if n % 2:
        return float(merged[n // 2])
    return (merged[n // 2 - 1] + merged[n // 2]) / 2.0

def _ref_min_window(s, t):
    from collections import Counter
    if not t or not s:
        return ""
    need = Counter(t)
    missing = len(t)
    best = (0, 0)
    lo = 0
    for hi, ch in enumerate(s, 1):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        if missing == 0:
            while need[s[lo]] < 0:
                need[s[lo]] += 1
                lo += 1
            if best == (0, 0) or hi - lo < best[1] - best[0]:
                best = (lo, hi)
            need[s[lo]] += 1
            missing += 1
            lo += 1
    return s[best[0]:best[1]]

def _ref_max_sliding_window(nums, k):
    from collections import deque
    dq, out = deque(), []
    for i, n in enumerate(nums):
        while dq and nums[dq[-1]] <= n:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out

def _ref_num_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 1 or seen[r][c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols and not seen[ny][nx] and grid[ny][nx] == 1:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
    return count

def _ref_top_k_frequent(nums, k):
    from collections import Counter
    return [n for n, _ in Counter(nums).most_common(k)]

def _ref_decode_string(s):
    stack = [("", 1)]
    num = 0
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch == "[":
            stack.append(("", num))
            num = 0
        elif ch == "]":
            text, rep = stack.pop()
            stack[-1] = (stack[-1][0] + text * rep, stack[-1][1])
        else:
            stack[-1] = (stack[-1][0] + ch, stack[-1][1])
    return stack[0][0]


# ---------------------------------------------------------------------------
# Stress generators — deterministic given the rng, so both players face the
# identical input and the comparison is fair.
# ---------------------------------------------------------------------------
def _ints(rng, n, lo=-10**6, hi=10**6):
    return [rng.randint(lo, hi) for _ in range(n)]

def _stress_two_sum(rng):
    n = 60000
    nums = _ints(rng, n, -10**7, 10**7)
    i, j = rng.sample(range(n), 2)
    return [[nums, nums[i] + nums[j]]]

def _stress_max_subarray(rng):
    return [[_ints(rng, 120000, -1000, 1000)]]

def _stress_contains_duplicate(rng):
    return [[rng.sample(range(10**7), 80000)]]        # all distinct: worst case

def _stress_max_profit(rng):
    return [[_ints(rng, 120000, 0, 10**5)]]

def _stress_single_number(rng):
    base = rng.sample(range(10**7), 40000)
    nums = base + base[:-1]
    rng.shuffle(nums)
    return [[nums]]

def _stress_sorted_squares(rng):
    return [[sorted(_ints(rng, 100000, -10**5, 10**5))]]

def _stress_move_zeroes(rng):
    nums = [rng.choice([0, rng.randint(1, 1000)]) for _ in range(100000)]
    return [[nums]]

def _stress_majority(rng):
    n = 100001
    maj = rng.randint(-1000, 1000)
    nums = [maj] * (n // 2 + 1) + _ints(rng, n // 2, -1000, 1000)
    rng.shuffle(nums)
    return [[nums]]

def _stress_product_except_self(rng):
    return [[_ints(rng, 60000, -20, 20)]]

def _stress_longest_substring(rng):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return [["".join(rng.choice(alphabet) for _ in range(120000))]]

def _stress_group_anagrams(rng):
    words = []
    for _ in range(30000):
        k = rng.randint(3, 7)
        w = "".join(rng.choice("abcdefg") for _ in range(k))
        words.append(w)
    return [[words]]

def _stress_coin_change(rng):
    coins = sorted(rng.sample(range(1, 60), 6))
    return [[coins, rng.randint(4000, 6000)]]

def _stress_lis(rng):
    return [[_ints(rng, 40000, -10**6, 10**6)]]

def _stress_three_sum(rng):
    return [[_ints(rng, 1200, -400, 400)]]

def _stress_search_insert(rng):
    nums = sorted(rng.sample(range(10**7), 200000))
    return [[nums, rng.randint(0, 10**7)]]

def _stress_word_break(rng):
    words = ["".join(rng.choice("ab") for _ in range(rng.randint(2, 5))) for _ in range(60)]
    s = "".join(rng.choice(words) for _ in range(220))
    return [[s, sorted(set(words))]]

def _stress_course_schedule(rng):
    n = 40000
    order = list(range(n))
    rng.shuffle(order)
    edges = []
    for _ in range(80000):
        i, j = sorted(rng.sample(range(n), 2))
        edges.append([order[j], order[i]])       # always forward: stays acyclic
    return [[n, edges]]

def _stress_trap(rng):
    return [[_ints(rng, 150000, 0, 5000)]]

def _stress_edit_distance(rng):
    a = "".join(rng.choice("abcde") for _ in range(700))
    b = "".join(rng.choice("abcde") for _ in range(700))
    return [[a, b]]

def _stress_median(rng):
    return [[sorted(_ints(rng, 60000)), sorted(_ints(rng, 60000))]]

def _stress_min_window(rng):
    s = "".join(rng.choice("abcdefgh") for _ in range(80000))
    t = "".join(rng.choice("abcdefgh") for _ in range(12))
    return [[s, t]]

def _stress_sliding_window_max(rng):
    return [[_ints(rng, 120000, -10**5, 10**5), 500]]

def _stress_num_islands(rng):
    rows, cols = 320, 320
    grid = [[1 if rng.random() < 0.45 else 0 for _ in range(cols)] for _ in range(rows)]
    return [[grid]]

def _stress_top_k(rng):
    return [[_ints(rng, 120000, 0, 4000), 12]]

def _stress_decode_string(rng):
    s = ""
    for _ in range(300):
        s += f"{rng.randint(2, 9)}[{''.join(rng.choice('abc') for _ in range(6))}]"
    return [[s]]


# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------
PROBLEMS: list[Problem] = [
    # ---------------------------------------------------------------- Easy --
    Problem(
        slug="two-sum", title="Two Sum", difficulty="Easy",
        description=(
            "Given an array of integers `nums` and an integer `target`, return the "
            "indices of the two numbers that add up to `target`. Each input has "
            "exactly one solution and you may not use the same element twice. "
            "Return the two indices as a list, in either order."
        ),
        function_name="two_sum",
        signature="def two_sum(nums: list, target: int) -> list:",
        starter_code="def two_sum(nums: list, target: int) -> list:\n    # your code here\n",
        compare=UNORDERED, complexity_hint="O(n) with a hash map",
        reference=_ref_two_sum, stress=_stress_two_sum,
        tests=[
            _tc("[2,7,11,15], 9", "[0,1]", "example", shown=True),
            _tc("[3,2,4], 6", "[1,2]", "example", shown=True),
            _tc("[3,3], 6", "[0,1]", "duplicate values"),
            _tc("[0,4,3,0], 0", "[0,3]", "zeroes"),
            _tc("[-3,4,3,90], 0", "[0,2]", "negatives"),
            _tc("[1,5,8,3], 9", "[0,2]", ""),
            _tc("[-1,-2,-3,-4], -7", "[2,3]", "all negative"),
        ],
    ),
    Problem(
        slug="contains-duplicate", title="Contains Duplicate", difficulty="Easy",
        description=(
            "Given an integer array `nums`, return True if any value appears at "
            "least twice, and False if every element is distinct."
        ),
        function_name="contains_duplicate",
        signature="def contains_duplicate(nums: list) -> bool:",
        starter_code="def contains_duplicate(nums: list) -> bool:\n    # your code here\n",
        complexity_hint="O(n) with a set",
        reference=_ref_contains_duplicate, stress=_stress_contains_duplicate,
        tests=[
            _tc("[1,2,3,1]", "true", "example", shown=True),
            _tc("[1,2,3,4]", "false", "example", shown=True),
            _tc("[1,1,1,3,3,4,3,2,4,2]", "true", ""),
            _tc("[]", "false", "empty"),
            _tc("[7]", "false", "single element"),
            _tc("[-1,-1]", "true", "negatives"),
        ],
    ),
    Problem(
        slug="best-time-buy-sell", title="Best Time to Buy and Sell Stock", difficulty="Easy",
        description=(
            "You are given an array `prices` where prices[i] is the price of a stock "
            "on day i. Choose one day to buy and a later day to sell to maximise "
            "profit. Return the maximum profit, or 0 if no profit is possible."
        ),
        function_name="max_profit",
        signature="def max_profit(prices: list) -> int:",
        starter_code="def max_profit(prices: list) -> int:\n    # your code here\n",
        complexity_hint="O(n) single pass tracking the running minimum",
        reference=_ref_max_profit, stress=_stress_max_profit,
        tests=[
            _tc("[7,1,5,3,6,4]", "5", "example", shown=True),
            _tc("[7,6,4,3,1]", "0", "monotonically falling"),
            _tc("[2,4,1]", "2", ""),
            _tc("[3,2,6,5,0,3]", "4", ""),
            _tc("[]", "0", "empty"),
            _tc("[5]", "0", "single day"),
            _tc("[2,2,2]", "0", "flat"),
        ],
    ),
    Problem(
        slug="single-number", title="Single Number", difficulty="Easy",
        description=(
            "Every element in `nums` appears exactly twice except for one, which "
            "appears once. Find and return that element. Your solution must run in "
            "linear time and use only constant extra space."
        ),
        function_name="single_number",
        signature="def single_number(nums: list) -> int:",
        starter_code="def single_number(nums: list) -> int:\n    # your code here\n",
        complexity_hint="XOR everything together",
        reference=_ref_single_number, stress=_stress_single_number,
        tests=[
            _tc("[2,2,1]", "1", "example", shown=True),
            _tc("[4,1,2,1,2]", "4", "example", shown=True),
            _tc("[1]", "1", "single element"),
            _tc("[7,3,5,3,5,7,9]", "9", ""),
            _tc("[-4,-4,-9]", "-9", "negatives"),
            _tc("[0,1,1]", "0", "answer is zero"),
        ],
    ),
    Problem(
        slug="squares-of-sorted-array", title="Squares of a Sorted Array", difficulty="Easy",
        description=(
            "Given an integer array `nums` sorted in non-decreasing order, return an "
            "array of the squares of each number, also sorted in non-decreasing order."
        ),
        function_name="sorted_squares",
        signature="def sorted_squares(nums: list) -> list:",
        starter_code="def sorted_squares(nums: list) -> list:\n    # your code here\n",
        complexity_hint="O(n) two-pointer from both ends beats O(n log n) sorting",
        reference=_ref_sorted_squares, stress=_stress_sorted_squares,
        tests=[
            _tc("[-4,-1,0,3,10]", "[0,1,9,16,100]", "example", shown=True),
            _tc("[-7,-3,2,3,11]", "[4,9,9,49,121]", "example", shown=True),
            _tc("[0]", "[0]", ""),
            _tc("[-1]", "[1]", "single negative"),
            _tc("[]", "[]", "empty"),
            _tc("[-5,-4,-3]", "[9,16,25]", "all negative"),
        ],
    ),
    Problem(
        slug="move-zeroes", title="Move Zeroes", difficulty="Easy",
        description=(
            "Given an integer array `nums`, move all 0s to the end while keeping the "
            "relative order of the non-zero elements. Modify the list in-place and "
            "return nothing."
        ),
        function_name="move_zeroes",
        signature="def move_zeroes(nums: list) -> None:",
        starter_code="def move_zeroes(nums: list) -> None:\n    # modify nums in-place, return nothing\n",
        in_place=True, complexity_hint="O(n) write pointer",
        reference=_ref_move_zeroes, stress=_stress_move_zeroes,
        tests=[
            _tc("[0,1,0,3,12]", "[1,3,12,0,0]", "example", shown=True),
            _tc("[0]", "[0]", ""),
            _tc("[1,2,3]", "[1,2,3]", "no zeroes"),
            _tc("[0,0,1]", "[1,0,0]", ""),
            _tc("[]", "[]", "empty"),
            _tc("[0,0,0]", "[0,0,0]", "all zeroes"),
        ],
    ),
    Problem(
        slug="majority-element", title="Majority Element", difficulty="Easy",
        description=(
            "Given an array `nums` of size n, return the element that appears more "
            "than n/2 times. You may assume such an element always exists."
        ),
        function_name="majority_element",
        signature="def majority_element(nums: list) -> int:",
        starter_code="def majority_element(nums: list) -> int:\n    # your code here\n",
        complexity_hint="Boyer-Moore voting, O(n) time and O(1) space",
        reference=_ref_majority_element, stress=_stress_majority,
        tests=[
            _tc("[3,2,3]", "3", "example", shown=True),
            _tc("[2,2,1,1,1,2,2]", "2", "example", shown=True),
            _tc("[1]", "1", ""),
            _tc("[5,5,5,1,5]", "5", ""),
            _tc("[-1,-1,2]", "-1", "negatives"),
        ],
    ),
    Problem(
        slug="search-insert-position", title="Search Insert Position", difficulty="Easy",
        description=(
            "Given a sorted array of distinct integers `nums` and a `target`, return "
            "the index of the target. If it is absent, return the index where it "
            "would be inserted to keep the array sorted. Must run in O(log n)."
        ),
        function_name="search_insert",
        signature="def search_insert(nums: list, target: int) -> int:",
        starter_code="def search_insert(nums: list, target: int) -> int:\n    # your code here\n",
        complexity_hint="Binary search",
        reference=_ref_search_insert, stress=_stress_search_insert,
        tests=[
            _tc("[1,3,5,6], 5", "2", "example", shown=True),
            _tc("[1,3,5,6], 2", "1", "example", shown=True),
            _tc("[1,3,5,6], 7", "4", "past the end"),
            _tc("[1,3,5,6], 0", "0", "before the start"),
            _tc("[], 4", "0", "empty"),
            _tc("[1], 1", "0", ""),
        ],
    ),

    # -------------------------------------------------------------- Medium --
    Problem(
        slug="max-subarray", title="Maximum Subarray", difficulty="Medium",
        description=(
            "Given an integer array `nums`, find the contiguous subarray containing "
            "at least one number which has the largest sum, and return that sum."
        ),
        function_name="max_subarray",
        signature="def max_subarray(nums: list) -> int:",
        starter_code="def max_subarray(nums: list) -> int:\n    # your code here\n",
        complexity_hint="Kadane's algorithm, O(n)",
        reference=_ref_max_subarray, stress=_stress_max_subarray,
        tests=[
            _tc("[-2,1,-3,4,-1,2,1,-5,4]", "6", "example", shown=True),
            _tc("[1]", "1", ""),
            _tc("[5,4,-1,7,8]", "23", "example", shown=True),
            _tc("[-1]", "-1", "single negative"),
            _tc("[-2,-3,-1,-5]", "-1", "all negative"),
            _tc("[0,0,0]", "0", "all zeroes"),
        ],
    ),
    Problem(
        slug="product-except-self", title="Product of Array Except Self", difficulty="Medium",
        description=(
            "Given an integer array `nums`, return an array `answer` where answer[i] "
            "is the product of all elements of nums except nums[i]. Solve it without "
            "using division, in O(n) time."
        ),
        function_name="product_except_self",
        signature="def product_except_self(nums: list) -> list:",
        starter_code="def product_except_self(nums: list) -> list:\n    # your code here\n",
        complexity_hint="Prefix products left-to-right then right-to-left",
        reference=_ref_product_except_self, stress=_stress_product_except_self,
        tests=[
            _tc("[1,2,3,4]", "[24,12,8,6]", "example", shown=True),
            _tc("[-1,1,0,-3,3]", "[0,0,9,0,0]", "contains a zero"),
            _tc("[2,3]", "[3,2]", "two elements"),
            _tc("[0,0]", "[0,0]", "two zeroes"),
            _tc("[1,1,1,1]", "[1,1,1,1]", ""),
        ],
    ),
    Problem(
        slug="longest-substring", title="Longest Substring Without Repeating Characters",
        difficulty="Medium",
        description=(
            "Given a string `s`, return the length of the longest substring that "
            "contains no repeated characters."
        ),
        function_name="length_of_longest_substring",
        signature="def length_of_longest_substring(s: str) -> int:",
        starter_code="def length_of_longest_substring(s: str) -> int:\n    # your code here\n",
        complexity_hint="Sliding window with last-seen positions, O(n)",
        reference=_ref_length_of_longest_substring, stress=_stress_longest_substring,
        tests=[
            _tc('"abcabcbb"', "3", "example", shown=True),
            _tc('"bbbbb"', "1", "example", shown=True),
            _tc('"pwwkew"', "3", "example", shown=True),
            _tc('""', "0", "empty"),
            _tc('"au"', "2", ""),
            _tc('"dvdf"', "3", "window must not jump backwards"),
            _tc('"abba"', "2", "classic off-by-one trap"),
        ],
    ),
    Problem(
        slug="group-anagrams", title="Group Anagrams", difficulty="Medium",
        description=(
            "Given an array of strings `strs`, group the anagrams together. Return a "
            "list of groups. The groups may be returned in any order, and the words "
            "within each group may be in any order."
        ),
        function_name="group_anagrams",
        signature="def group_anagrams(strs: list) -> list:",
        starter_code="def group_anagrams(strs: list) -> list:\n    # your code here\n",
        compare=NESTED_UNORDERED, complexity_hint="Bucket by sorted characters",
        reference=_ref_group_anagrams, stress=_stress_group_anagrams,
        tests=[
            _tc('["eat","tea","tan","ate","nat","bat"]',
                '[["eat","tea","ate"],["tan","nat"],["bat"]]', "example", shown=True),
            _tc('[""]', '[[""]]', "empty string"),
            _tc('["a"]', '[["a"]]', ""),
            _tc('["ab","ba","abc"]', '[["ab","ba"],["abc"]]', ""),
            _tc('[]', '[]', "empty input"),
        ],
    ),
    Problem(
        slug="coin-change", title="Coin Change", difficulty="Medium",
        description=(
            "Given a list of coin denominations `coins` and an integer `amount`, "
            "return the fewest number of coins needed to make up that amount. If it "
            "cannot be made from any combination, return -1. You have an infinite "
            "supply of each coin."
        ),
        function_name="coin_change",
        signature="def coin_change(coins: list, amount: int) -> int:",
        starter_code="def coin_change(coins: list, amount: int) -> int:\n    # your code here\n",
        complexity_hint="Bottom-up DP over amounts, O(amount * len(coins))",
        reference=_ref_coin_change, stress=_stress_coin_change,
        tests=[
            _tc("[1,2,5], 11", "3", "example", shown=True),
            _tc("[2], 3", "-1", "impossible"),
            _tc("[1], 0", "0", "zero amount"),
            _tc("[186,419,83,408], 6249", "20", "greedy fails here"),
            _tc("[2,5,10,1], 27", "4", ""),
            _tc("[7], 14", "2", ""),
        ],
    ),
    Problem(
        slug="longest-increasing-subsequence", title="Longest Increasing Subsequence",
        difficulty="Medium",
        description=(
            "Given an integer array `nums`, return the length of the longest strictly "
            "increasing subsequence. A subsequence need not be contiguous."
        ),
        function_name="length_of_lis",
        signature="def length_of_lis(nums: list) -> int:",
        starter_code="def length_of_lis(nums: list) -> int:\n    # your code here\n",
        complexity_hint="Patience sorting with bisect, O(n log n)",
        reference=_ref_length_of_lis, stress=_stress_lis,
        tests=[
            _tc("[10,9,2,5,3,7,101,18]", "4", "example", shown=True),
            _tc("[0,1,0,3,2,3]", "4", "example", shown=True),
            _tc("[7,7,7,7]", "1", "strictly increasing only"),
            _tc("[]", "0", "empty"),
            _tc("[5,4,3,2,1]", "1", "descending"),
            _tc("[1,2,3,4,5]", "5", "already sorted"),
        ],
    ),
    Problem(
        slug="three-sum", title="3Sum", difficulty="Medium",
        description=(
            "Given an integer array `nums`, return all unique triplets [a, b, c] such "
            "that a + b + c == 0. Each triplet must be sorted ascending, and no "
            "triplet may be repeated. The triplets may be returned in any order."
        ),
        function_name="three_sum",
        signature="def three_sum(nums: list) -> list:",
        starter_code="def three_sum(nums: list) -> list:\n    # your code here\n",
        compare=NESTED_UNORDERED, complexity_hint="Sort, then two pointers per anchor, O(n^2)",
        reference=_ref_three_sum, stress=_stress_three_sum,
        tests=[
            _tc("[-1,0,1,2,-1,-4]", "[[-1,-1,2],[-1,0,1]]", "example", shown=True),
            _tc("[0,1,1]", "[]", "no triplet"),
            _tc("[0,0,0]", "[[0,0,0]]", "all zeroes"),
            _tc("[0,0,0,0]", "[[0,0,0]]", "must deduplicate"),
            _tc("[]", "[]", "empty"),
            _tc("[-2,0,1,1,2]", "[[-2,0,2],[-2,1,1]]", ""),
        ],
    ),
    Problem(
        slug="word-break", title="Word Break", difficulty="Medium",
        description=(
            "Given a string `s` and a list of words `word_dict`, return True if `s` "
            "can be segmented into a space-separated sequence of one or more words "
            "from the dictionary. Words may be reused any number of times."
        ),
        function_name="word_break",
        signature="def word_break(s: str, word_dict: list) -> bool:",
        starter_code="def word_break(s: str, word_dict: list) -> bool:\n    # your code here\n",
        complexity_hint="DP over prefixes — naive recursion blows up exponentially",
        reference=_ref_word_break, stress=_stress_word_break,
        tests=[
            _tc('"leetcode", ["leet","code"]', "true", "example", shown=True),
            _tc('"applepenapple", ["apple","pen"]', "true", "reuses a word"),
            _tc('"catsandog", ["cats","dog","sand","and","cat"]', "false", "example", shown=True),
            _tc('"", ["a"]', "true", "empty string"),
            _tc('"aaaaaaab", ["a","aa","aaa"]', "false",
                "exponential without memoisation"),
            _tc('"cars", ["car","ca","rs"]', "true", ""),
        ],
    ),
    Problem(
        slug="course-schedule", title="Course Schedule", difficulty="Medium",
        description=(
            "There are `num_courses` courses labelled 0 to num_courses-1. "
            "`prerequisites[i] = [a, b]` means you must take course b before course a. "
            "Return True if it is possible to finish all courses, False otherwise."
        ),
        function_name="can_finish",
        signature="def can_finish(num_courses: int, prerequisites: list) -> bool:",
        starter_code="def can_finish(num_courses: int, prerequisites: list) -> bool:\n    # your code here\n",
        complexity_hint="Cycle detection — topological sort with in-degrees",
        reference=_ref_can_finish, stress=_stress_course_schedule,
        tests=[
            _tc("2, [[1,0]]", "true", "example", shown=True),
            _tc("2, [[1,0],[0,1]]", "false", "two-cycle"),
            _tc("1, []", "true", "no prerequisites"),
            _tc("4, [[1,0],[2,1],[3,2]]", "true", "chain"),
            _tc("3, [[0,1],[1,2],[2,0]]", "false", "three-cycle"),
            _tc("5, [[1,0],[2,0],[3,1],[4,2]]", "true", "tree"),
        ],
    ),
    Problem(
        slug="top-k-frequent", title="Top K Frequent Elements", difficulty="Medium",
        description=(
            "Given an integer array `nums` and an integer `k`, return the k most "
            "frequent elements. The answer may be returned in any order. The input "
            "guarantees the k-th most frequent element is unambiguous."
        ),
        function_name="top_k_frequent",
        signature="def top_k_frequent(nums: list, k: int) -> list:",
        starter_code="def top_k_frequent(nums: list, k: int) -> list:\n    # your code here\n",
        compare=UNORDERED, complexity_hint="Counter plus a heap or bucket sort",
        reference=_ref_top_k_frequent, stress=_stress_top_k,
        tests=[
            _tc("[1,1,1,2,2,3], 2", "[1,2]", "example", shown=True),
            _tc("[1], 1", "[1]", ""),
            _tc("[4,4,4,5,5,6], 3", "[4,5,6]", "k equals distinct count"),
            _tc("[-1,-1,2], 1", "[-1]", "negatives"),
        ],
    ),
    Problem(
        slug="decode-string", title="Decode String", difficulty="Medium",
        description=(
            "Given an encoded string `s`, return its decoded form. The encoding rule "
            "is k[encoded_string], meaning the substring inside the brackets repeats "
            "exactly k times. Brackets may be nested. k is always a positive integer."
        ),
        function_name="decode_string",
        signature="def decode_string(s: str) -> str:",
        starter_code="def decode_string(s: str) -> str:\n    # your code here\n",
        complexity_hint="Stack of (text, repeat) frames",
        reference=_ref_decode_string, stress=_stress_decode_string,
        tests=[
            _tc('"3[a]2[bc]"', '"aaabcbc"', "example", shown=True),
            _tc('"3[a2[c]]"', '"accaccacc"', "nested"),
            _tc('"2[abc]3[cd]ef"', '"abcabccdcdcdef"', "example", shown=True),
            _tc('"abc"', '"abc"', "no brackets"),
            _tc('"10[a]"', '"aaaaaaaaaa"', "multi-digit count"),
            _tc('""', '""', "empty"),
        ],
    ),

    # ---------------------------------------------------------------- Hard --
    Problem(
        slug="trapping-rain-water", title="Trapping Rain Water", difficulty="Hard",
        description=(
            "Given `height`, a list of non-negative integers representing an "
            "elevation map where the width of each bar is 1, compute how much water "
            "it can trap after raining."
        ),
        function_name="trap",
        signature="def trap(height: list) -> int:",
        starter_code="def trap(height: list) -> int:\n    # your code here\n",
        complexity_hint="Two pointers, O(n) time and O(1) space",
        reference=_ref_trap, stress=_stress_trap,
        tests=[
            _tc("[0,1,0,2,1,0,1,3,2,1,2,1]", "6", "example", shown=True),
            _tc("[4,2,0,3,2,5]", "9", "example", shown=True),
            _tc("[]", "0", "empty"),
            _tc("[1,2,3]", "0", "monotonic, traps nothing"),
            _tc("[3,2,1]", "0", "descending"),
            _tc("[5,0,5]", "5", "single basin"),
            _tc("[2,0,2,0,2]", "4", "two basins"),
        ],
    ),
    Problem(
        slug="edit-distance", title="Edit Distance", difficulty="Hard",
        description=(
            "Given two strings `word1` and `word2`, return the minimum number of "
            "single-character insertions, deletions or substitutions needed to turn "
            "word1 into word2."
        ),
        function_name="min_distance",
        signature="def min_distance(word1: str, word2: str) -> int:",
        starter_code="def min_distance(word1: str, word2: str) -> int:\n    # your code here\n",
        complexity_hint="Levenshtein DP, O(m*n) — a rolling row keeps memory linear",
        reference=_ref_min_distance, stress=_stress_edit_distance,
        tests=[
            _tc('"horse", "ros"', "3", "example", shown=True),
            _tc('"intention", "execution"', "5", "example", shown=True),
            _tc('"", ""', "0", "both empty"),
            _tc('"", "abc"', "3", "one empty"),
            _tc('"abc", "abc"', "0", "identical"),
            _tc('"a", "b"', "1", "single substitution"),
        ],
    ),
    Problem(
        slug="median-two-sorted", title="Median of Two Sorted Arrays", difficulty="Hard",
        description=(
            "Given two sorted integer arrays `nums1` and `nums2`, return the median "
            "of the combined sorted array as a float. If the combined length is even, "
            "the median is the mean of the two middle values."
        ),
        function_name="find_median_sorted_arrays",
        signature="def find_median_sorted_arrays(nums1: list, nums2: list) -> float:",
        starter_code="def find_median_sorted_arrays(nums1: list, nums2: list) -> float:\n    # your code here\n",
        compare=APPROX, complexity_hint="Binary search on the partition, O(log(m+n))",
        reference=_ref_find_median, stress=_stress_median,
        tests=[
            _tc("[1,3], [2]", "2.0", "example", shown=True),
            _tc("[1,2], [3,4]", "2.5", "even total"),
            _tc("[], [1]", "1.0", "one empty"),
            _tc("[], [2,3]", "2.5", ""),
            _tc("[0,0], [0,0]", "0.0", "all equal"),
            _tc("[1,2,3,4,5], [6,7,8]", "4.5", ""),
        ],
    ),
    Problem(
        slug="minimum-window-substring", title="Minimum Window Substring", difficulty="Hard",
        description=(
            "Given strings `s` and `t`, return the shortest substring of s that "
            "contains every character of t including duplicates. If there is no such "
            "substring, return the empty string. The answer is guaranteed unique."
        ),
        function_name="min_window",
        signature="def min_window(s: str, t: str) -> str:",
        starter_code="def min_window(s: str, t: str) -> str:\n    # your code here\n",
        complexity_hint="Sliding window with a need-counter, O(n)",
        reference=_ref_min_window, stress=_stress_min_window,
        tests=[
            _tc('"ADOBECODEBANC", "ABC"', '"BANC"', "example", shown=True),
            _tc('"a", "a"', '"a"', ""),
            _tc('"a", "aa"', '""', "not enough copies"),
            _tc('"", "a"', '""', "empty source"),
            _tc('"ab", ""', '""', "empty target"),
            _tc('"aa", "aa"', '"aa"', "duplicates required"),
        ],
    ),
    Problem(
        slug="sliding-window-maximum", title="Sliding Window Maximum", difficulty="Hard",
        description=(
            "Given an integer array `nums` and a window size `k`, return a list of "
            "the maximum value in each contiguous window of size k as the window "
            "slides from left to right."
        ),
        function_name="max_sliding_window",
        signature="def max_sliding_window(nums: list, k: int) -> list:",
        starter_code="def max_sliding_window(nums: list, k: int) -> list:\n    # your code here\n",
        complexity_hint="Monotonic deque of indices, O(n) — a per-window max is O(n*k)",
        reference=_ref_max_sliding_window, stress=_stress_sliding_window_max,
        tests=[
            _tc("[1,3,-1,-3,5,3,6,7], 3", "[3,3,5,5,6,7]", "example", shown=True),
            _tc("[1], 1", "[1]", ""),
            _tc("[9,8,7,6], 2", "[9,8,7]", "descending"),
            _tc("[1,2,3,4], 4", "[4]", "window is the whole array"),
            _tc("[-7,-8,-9], 2", "[-7,-8]", "negatives"),
        ],
    ),
    Problem(
        slug="number-of-islands", title="Number of Islands", difficulty="Hard",
        description=(
            "Given a 2D grid of 0s (water) and 1s (land), return the number of "
            "islands. An island is a group of 1s connected horizontally or "
            "vertically; diagonals do not connect. The grid edges are all water."
        ),
        function_name="num_islands",
        signature="def num_islands(grid: list) -> int:",
        starter_code="def num_islands(grid: list) -> int:\n    # your code here\n",
        complexity_hint="Flood fill — use an explicit stack, recursion will hit the limit",
        reference=_ref_num_islands, stress=_stress_num_islands,
        tests=[
            _tc("[[1,1,0],[0,1,0],[0,0,1]]", "2", "example", shown=True),
            _tc("[[1,1,1],[1,1,1]]", "1", "all one island"),
            _tc("[[0,0],[0,0]]", "0", "all water"),
            _tc("[]", "0", "empty grid"),
            _tc("[[1,0,1,0,1]]", "3", "single row"),
            _tc("[[1],[0],[1]]", "2", "single column"),
        ],
    ),
]


def get_problem(slug: str) -> Optional[Problem]:
    for p in PROBLEMS:
        if p.slug == slug:
            return p
    return None


for _p in PROBLEMS:
    _p.source = "local"
    _p.attribution = "MindArena curated bank"

BY_DIFFICULTY: dict[str, list[Problem]] = {"Easy": [], "Medium": [], "Hard": []}
for _p in PROBLEMS:
    BY_DIFFICULTY.setdefault(_p.difficulty, []).append(_p)


def sample_problems(count: int, rng: Optional[random.Random] = None) -> list[Problem]:
    """Pick `count` distinct problems spread across the difficulty tiers.

    Plain random.sample over a bank that is mostly Easy produces mostly-Easy
    battles, which is not much of a contest. This walks Easy -> Medium -> Hard
    so a three-round battle gets one of each, and longer battles keep the mix.
    """
    rng = rng or random.Random()
    pools = {d: rng.sample(ps, len(ps)) for d, ps in BY_DIFFICULTY.items() if ps}
    order = [d for d in ("Easy", "Medium", "Hard") if d in pools]
    picked: list[Problem] = []
    while len(picked) < count and any(pools[d] for d in order):
        for d in order:
            if pools[d] and len(picked) < count:
                picked.append(pools[d].pop())
    return picked[:count]
