"""
The shape of a code-battle challenge, shared by the judge and every problem source.

Two kinds of challenge exist, because the interesting problem sources disagree
about what a solution looks like:

  function  The model writes one function; the judge calls it with arguments and
            compares return values. This is the LeetCode-style shape used by the
            curated local bank.

  stdio     The model writes a whole program that reads stdin and writes stdout.
            This is the shape real competitive-programming archives use, so it is
            what the Codeforces-derived bank produces.

Comparison is deliberately explicit per challenge. Several problems allow answers
in any order, and checking those with `==` fails correct solutions — which is a
worse failure than a missing test, because it silently decides the match wrong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Kinds.
FUNCTION = "function"
STDIO = "stdio"

# Comparison modes for the `function` kind.
EXACT = "exact"                        # got == expected
UNORDERED = "unordered"                # flat list, order irrelevant
NESTED_UNORDERED = "nested_unordered"  # list of lists, order irrelevant at both levels
APPROX = "approx"                      # numeric, 1e-6 tolerance

DIFFICULTIES = ("Easy", "Medium", "Hard", "Expert")


@dataclass
class Case:
    """One test. `args`/`expected` for function challenges, `stdin`/`stdout` for stdio."""
    label: str = ""
    args: Optional[list] = None
    expected: Any = None
    stdin: str = ""
    stdout: str = ""
    hidden: bool = True     # False for the worked examples shown in the statement


@dataclass
class Challenge:
    slug: str
    title: str
    difficulty: str                     # Easy | Medium | Hard | Expert
    description: str
    tests: list[Case]
    kind: str = FUNCTION
    rating: int = 0                     # Codeforces-style rating; 0 = unrated
    source: str = "local"
    attribution: str = ""
    tags: list[str] = field(default_factory=list)
    # function kind only
    function_name: str = ""
    signature: str = ""
    starter_code: str = ""
    compare: str = EXACT
    in_place: bool = False
    reference: Optional[Callable] = None
    stress: Optional[Callable] = None   # (rng) -> list[list]; parent-side, trusted
    complexity_hint: str = ""
    # limits
    time_limit_s: float = 5.0
    stress_limit_s: float = 10.0
    memory_mb: int = 1024

    @property
    def stress_enabled(self) -> bool:
        return self.kind == FUNCTION and self.reference is not None and self.stress is not None

    @property
    def examples(self) -> list[Case]:
        return [c for c in self.tests if not c.hidden]

    def public_dict(self) -> dict:
        """What the UI is allowed to see. Hidden test data stays on the server."""
        return {
            "slug": self.slug, "title": self.title, "difficulty": self.difficulty,
            "rating": self.rating, "source": self.source, "attribution": self.attribution,
            "tags": self.tags, "kind": self.kind, "description": self.description,
            "signature": self.signature, "complexity_hint": self.complexity_hint,
            "test_count": len(self.tests),
            "examples": [
                {"stdin": c.stdin, "stdout": c.stdout,
                 "args": c.args, "expected": c.expected, "label": c.label}
                for c in self.examples[:3]
            ],
        }


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------
def _approx_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_approx_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    return a == b


def _sort_key(x: Any):
    """Total order over mixed JSON values, so sorting never raises on odd input."""
    return (type(x).__name__, x if isinstance(x, (int, float, str, bool)) else repr(x))


def compare_answer(mode: str, got: Any, expected: Any) -> bool:
    """Check a function-kind answer under the challenge's comparison mode."""
    try:
        if mode == UNORDERED:
            if not isinstance(got, list) or len(got) != len(expected):
                return False
            return sorted(got, key=_sort_key) == sorted(expected, key=_sort_key)
        if mode == NESTED_UNORDERED:
            if not isinstance(got, list) or len(got) != len(expected):
                return False
            norm = lambda xs: sorted((sorted(x, key=_sort_key) if isinstance(x, list) else x
                                      for x in xs), key=_sort_key)
            return norm(got) == norm(expected)
        if mode == APPROX:
            return _approx_equal(got, expected)
        return got == expected
    except (TypeError, ValueError):
        return False


def compare_stdout(got: str, expected: str, tol: float = 1e-6) -> bool:
    """Token-wise stdout comparison.

    Competitive-programming judges ignore trailing whitespace and line-ending
    differences, and accept small float error. Comparing raw strings would fail
    correct programs over a trailing newline.
    """
    gt, et = got.split(), expected.split()
    if len(gt) != len(et):
        return False
    for a, b in zip(gt, et):
        if a == b:
            continue
        try:
            if math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol):
                continue
        except (TypeError, ValueError):
            pass        # not numeric — fall through to the textual comparison
        # Case-insensitive, because judges accept YES/Yes/yes interchangeably.
        if a.lower() != b.lower():
            return False
    return True
