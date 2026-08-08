"""
Where challenges come from.

Two sources feed the arena:

  local          The curated bank in problems.py. Always available, no network,
                 function-shaped, with reference solutions and stress generators.

  code_contests  Real Codeforces problems pulled at run time from DeepMind's
                 CodeContests archive via the public Hugging Face datasets API.
                 These are the ones that actually stretch a model: they carry a
                 Codeforces difficulty rating (800 easiest, 3500 hardest), real
                 hidden test data, and statements written for human competitors.

Everything about the remote path is best-effort. If the network is unavailable,
the archive is down, or the response is unusable, the bank quietly falls back to
the local set and says so in the log — a code battle should never fail to start
because a dataset host is having a bad day.

Fetched problems are filtered before they are admitted:

  * statements that accept more than one valid output are dropped, because the
    judge compares against one expected answer and would mark correct programs
    wrong — a silent, match-deciding failure;
  * interactive problems are dropped, since there is no interactor here;
  * problems with too few tests are dropped;
  * optionally, a known-good Python solution shipped with the dataset is run
    against the selected tests, and the problem is only admitted if it passes.

Attribution: CodeContests is published by DeepMind under Apache-2.0; the
underlying statements belong to their original authors (mostly Codeforces).
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Optional

from challenge import Challenge, Case, STDIO

# Fixed, non-configurable host. Nothing here is derived from user input, so this
# is not an SSRF surface the way a user-supplied model endpoint is.
DATASETS_API = "https://datasets-server.huggingface.co/filter"
DATASET = "deepmind/code_contests"
CACHE_VERSION = 1

CACHE_DIR = Path(os.environ.get("MINDARENA_CACHE",
                                Path.home() / ".cache" / "mindarena"))
CACHE_FILE = CACHE_DIR / f"code_contests-v{CACHE_VERSION}.json"

# Statements whose answer is not unique. The judge has exactly one expected
# output per test, so admitting these would fail correct solutions.
AMBIGUOUS = re.compile(
    r"print any|output any|any of them|any such|if there are (?:several|multiple|many)"
    r"|multiple (?:possible )?(?:answers|solutions|ways)|any valid|in any order"
    r"|this is an interactive problem|interactor",
    re.I,
)
# Floating-point answers need a checker with a tolerance the statement specifies;
# token comparison with a fixed epsilon is not faithful enough.
FLOAT_ANSWER = re.compile(r"absolute or relative error|checker|10\^\{?-[0-9]", re.I)

MAX_TESTS = 12              # per problem, to keep a round to a sane wall time
MIN_TESTS = 8
MAX_DESCRIPTION = 6000

# Codeforces rating -> the arena's difficulty label.
RATING_BANDS = [
    ("Easy", 800, 1199),
    ("Medium", 1200, 1699),
    ("Hard", 1700, 2199),
    ("Expert", 2200, 3500),
]


def difficulty_for_rating(rating: int) -> str:
    for label, lo, hi in RATING_BANDS:
        if lo <= rating <= hi:
            return label
    return "Expert" if rating > 3500 else "Easy"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _get_json(params: dict, timeout: float) -> dict:
    url = f"{DATASETS_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "MindArena/1.0 (+code battle problem loader)",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


# ---------------------------------------------------------------------------
# Row -> Challenge
# ---------------------------------------------------------------------------
def _pick_tests(row: dict, rng: random.Random) -> list[Case]:
    """Take the worked examples, then as many hidden tests as the budget allows."""
    cases: list[Case] = []
    pub = row.get("public_tests") or {"input": [], "output": []}
    for i, (stdin, stdout) in enumerate(zip(pub["input"], pub["output"])):
        cases.append(Case(label=f"example {i + 1}", stdin=stdin, stdout=stdout, hidden=False))

    hidden: list[Case] = []
    for key in ("private_tests", "generated_tests"):
        block = row.get(key) or {"input": [], "output": []}
        for stdin, stdout in zip(block["input"], block["output"]):
            hidden.append(Case(label="hidden", stdin=stdin, stdout=stdout, hidden=True))
    rng.shuffle(hidden)
    for i, case in enumerate(hidden[: max(0, MAX_TESTS - len(cases))], start=1):
        case.label = f"hidden {i}"
        cases.append(case)
    return cases


def _row_to_challenge(row: dict, rng: random.Random) -> Optional[Challenge]:
    description = (row.get("description") or "").strip()
    if not description or len(description) > MAX_DESCRIPTION:
        return None
    if AMBIGUOUS.search(description) or FLOAT_ANSWER.search(description):
        return None

    cases = _pick_tests(row, rng)
    if len(cases) < MIN_TESTS:
        return None
    # A test whose expected output is empty cannot be checked meaningfully.
    if any(not c.stdout.strip() for c in cases):
        return None

    rating = int(row.get("cf_rating") or 0)
    name = row.get("name") or "Untitled"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]

    # Codeforces limits assume C++. Python needs materially more headroom, or
    # every problem becomes "did the model write C++-fast Python".
    cf_seconds = float((row.get("time_limit") or {}).get("seconds") or 2)
    limit = min(max(cf_seconds * 3.0, 4.0), 12.0)
    memory_mb = int((row.get("memory_limit_bytes") or 256_000_000) / (1024 * 1024))

    return Challenge(
        slug=slug,
        title=re.sub(r"^\d+_[A-Z0-9]+\.\s*", "", name),
        difficulty=difficulty_for_rating(rating),
        rating=rating,
        description=description,
        tests=cases,
        kind=STDIO,
        source="code_contests",
        attribution="Codeforces via DeepMind CodeContests (Apache-2.0)",
        tags=list(row.get("cf_tags") or []),
        time_limit_s=limit,
        memory_mb=min(max(memory_mb, 256), 2048),
    )


def _reference_solution(row: dict) -> str:
    """The shortest Python 3 solution the dataset ships for this problem."""
    block = row.get("solutions") or {}
    langs = block.get("language") or []
    sources = block.get("solution") or []
    # CodeContests language enum: 1=Python2, 2=C++, 3=Python3, 4=Java.
    py = [s for s, lang in zip(sources, langs) if lang == 3]
    return min(py, key=len) if py else ""


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch_code_contests(
    target: int = 12,
    min_rating: int = 1200,
    max_rating: int = 2600,
    split: str = "test",
    validate: bool = True,
    timeout: float = 60.0,
    log: Optional[Callable[[str], None]] = None,
) -> list[Challenge]:
    """Pull real contest problems, filter them, and optionally self-validate.

    Returns whatever it managed to get — an empty list is a normal outcome when
    the network is unavailable, and callers fall back to the local bank.
    """
    say = log or (lambda _m: None)
    rng = random.Random(0xC0DE)
    admitted: list[Challenge] = []
    seen_slugs: set[str] = set()
    batch = 4
    offset = 0
    # Ask for a few pages at most: each row is roughly a megabyte, so this is
    # the difference between a warm-up that takes seconds and one that takes
    # minutes. Half the rows survive filtering in practice.
    max_pages = max(4, (target * 3) // batch)

    columns = ",".join([
        "name", "description", "public_tests", "private_tests",
        "cf_rating", "cf_tags", "time_limit", "memory_limit_bytes",
    ] + (["solutions"] if validate else []))

    for _page in range(max_pages):
        if len(admitted) >= target:
            break
        try:
            data = _get_json({
                "dataset": DATASET, "config": "default", "split": split,
                "where": f'"cf_rating">={min_rating} AND "cf_rating"<={max_rating}',
                "columns": columns, "offset": offset, "length": batch,
            }, timeout)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as e:
            say(f"code_contests: fetch stopped ({type(e).__name__}: {e})")
            break

        rows = data.get("rows") or []
        if not rows:
            break
        offset += len(rows)

        for entry in rows:
            if len(admitted) >= target:
                break
            row = entry.get("row") or {}
            challenge = _row_to_challenge(row, rng)
            if challenge is None or challenge.slug in seen_slugs:
                continue
            if validate:
                ok, why = _validate(challenge, _reference_solution(row))
                if not ok:
                    say(f"code_contests: dropped {challenge.slug} ({why})")
                    continue
            seen_slugs.add(challenge.slug)
            admitted.append(challenge)
            say(f"code_contests: + {challenge.title} [{challenge.difficulty} {challenge.rating}]")

    return admitted


def _validate(challenge: Challenge, solution: str) -> tuple[bool, str]:
    """Run a known-good solution against the selected tests.

    This is what makes a scraped problem trustworthy: it proves the tests are
    self-consistent, that plain token comparison is a fair check, and that the
    time limit is actually reachable from Python.
    """
    if not solution:
        return False, "no Python reference solution in the dataset"
    import codebattle  # imported late: codebattle imports the local bank
    probe = Challenge(**{**challenge.__dict__, "tests": challenge.tests[:6]})
    result = codebattle.judge(probe, solution, run_stress=False)
    if result.compile_error:
        return False, f"reference failed to run: {result.compile_error[:80]}"
    if result.passed != result.total:
        bad = result.first_failure
        return False, f"reference scored {result.passed}/{result.total} ({bad.status if bad else '?'})"
    return True, ""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _to_cache(challenges: Iterable[Challenge]) -> list[dict]:
    return [{
        "slug": c.slug, "title": c.title, "difficulty": c.difficulty,
        "rating": c.rating, "description": c.description, "tags": c.tags,
        "source": c.source, "attribution": c.attribution, "kind": c.kind,
        "time_limit_s": c.time_limit_s, "memory_mb": c.memory_mb,
        "tests": [{"label": t.label, "stdin": t.stdin, "stdout": t.stdout,
                   "hidden": t.hidden} for t in c.tests],
    } for c in challenges]


def _from_cache(blobs: list[dict]) -> list[Challenge]:
    out = []
    for b in blobs:
        try:
            out.append(Challenge(
                slug=b["slug"], title=b["title"], difficulty=b["difficulty"],
                rating=b.get("rating", 0), description=b["description"],
                tags=b.get("tags", []), source=b.get("source", "code_contests"),
                attribution=b.get("attribution", ""), kind=b.get("kind", STDIO),
                time_limit_s=b.get("time_limit_s", 6.0),
                memory_mb=b.get("memory_mb", 512),
                tests=[Case(label=t.get("label", ""), stdin=t.get("stdin", ""),
                            stdout=t.get("stdout", ""), hidden=t.get("hidden", True))
                       for t in b["tests"]],
            ))
        except (KeyError, TypeError):
            continue
    return out


def load_cached() -> list[Challenge]:
    try:
        blob = json.loads(CACHE_FILE.read_text("utf-8"))
    except (OSError, ValueError):
        return []
    if blob.get("version") != CACHE_VERSION:
        return []
    return _from_cache(blob.get("problems") or [])


def save_cache(challenges: list[Challenge]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps({
            "version": CACHE_VERSION, "fetched_at": time.time(),
            "problems": _to_cache(challenges),
        }), "utf-8")
    except OSError:
        pass       # a cache we cannot write is not worth failing a battle over


# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------
class ProblemBank:
    """Holds every challenge the arena can draw from."""

    def __init__(self):
        from problems import PROBLEMS as LOCAL
        self.local: list[Challenge] = list(LOCAL)
        self.remote: list[Challenge] = load_cached()
        self.remote_error: str = ""
        self.last_refresh: float = 0.0

    @property
    def all(self) -> list[Challenge]:
        return self.local + self.remote

    def counts(self) -> dict:
        by_difficulty: dict[str, int] = {}
        for c in self.all:
            by_difficulty[c.difficulty] = by_difficulty.get(c.difficulty, 0) + 1
        return {
            "local": len(self.local), "remote": len(self.remote),
            "by_difficulty": by_difficulty, "remote_error": self.remote_error,
        }

    def refresh_remote(self, target: int = 12, min_rating: int = 1200,
                       max_rating: int = 2600, validate: bool = True,
                       log: Optional[Callable[[str], None]] = None) -> int:
        """Fetch and cache remote problems. Returns how many are now held."""
        got = fetch_code_contests(target=target, min_rating=min_rating,
                                  max_rating=max_rating, validate=validate, log=log)
        if got:
            existing = {c.slug: c for c in self.remote}
            for c in got:
                existing[c.slug] = c
            self.remote = list(existing.values())
            self.remote_error = ""
            save_cache(self.remote)
        elif not self.remote:
            self.remote_error = "could not reach the remote archive"
        return len(self.remote)

    def select(self, count: int, sources: Iterable[str] = ("local", "code_contests"),
               rng: Optional[random.Random] = None) -> list[Challenge]:
        """Pick `count` distinct challenges spread across the difficulty tiers.

        Plain sampling over a bank that is mostly Easy produces mostly-Easy
        battles, which is not much of a contest. This walks the tiers in order so
        a three-round battle gets one of each and longer battles keep the mix.
        """
        rng = rng or random.Random()
        wanted = set(sources)
        pool = [c for c in self.all if c.source in wanted] or list(self.all)

        tiers: dict[str, list[Challenge]] = {}
        for c in pool:
            tiers.setdefault(c.difficulty, []).append(c)
        for group in tiers.values():
            rng.shuffle(group)

        order = [d for d in ("Easy", "Medium", "Hard", "Expert") if tiers.get(d)]
        picked: list[Challenge] = []
        while len(picked) < count and any(tiers[d] for d in order):
            for d in order:
                if tiers[d] and len(picked) < count:
                    picked.append(tiers[d].pop())
        return picked[:count]


BANK = ProblemBank()
