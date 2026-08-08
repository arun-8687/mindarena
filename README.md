# MindArena

Two LLMs go head-to-head while you watch — either across a board game (chess or
Othello) or in a **code battle**, where both models solve the same programming
problems and a sandboxed judge scores their submissions against hidden tests.

The server talks to the model endpoint itself rather than from the browser,
which sidesteps the CORS wall that stops a static page from calling
`ollama.com/v1` directly.

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
OLLAMA_API_KEY=... python launcher.py
```

Then open http://127.0.0.1:8791.

`launcher.py` will also read a `.env` file if you'd rather not export keys by
hand; it checks `$MINDARENA_ENV`, then `./.env`, then
`~/.config/mindarena/.env`. Values already in the environment take precedence.

## Configuration

Everything below has a sensible default; set only what you need.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_API_KEY` | — | Key for the Ollama Cloud preset |
| `OPENCODE_GO_API_KEY` | — | Key for the OpenCode Zen preset |
| `GT_API_KEY` | — | Fallback key for the global endpoint |
| `GT_BASE_URL` | `https://ollama.com/v1` | Global OpenAI-compatible endpoint |
| `GT_HOST` | `127.0.0.1` | Bind address |
| `GT_PORT` | `8791` | Bind port |
| `GT_P0_MODEL` / `GT_P1_MODEL` | `kimi-k3` / `deepseek-v4-pro` | Starting models |
| `GT_P0_LABEL` / `GT_P1_LABEL` | model names | Starting display names |
| `GT_ALLOW_INTERNAL_ENDPOINTS` | off when exposed | Allow private/loopback model endpoints |
| `MINDARENA_CACHE` | `~/.cache/mindarena` | Where fetched contest problems are cached |

Match settings — models, personas, temperature, reasoning effort, number of
games, ply limit, retry budget — all live in the in-app Settings drawer and can
be changed between series without restarting.

### A note on exposure

The server has no authentication and holds your API keys, so it binds to
loopback by default. If you set `GT_HOST=0.0.0.0` to reach it from a phone or
over a tailnet, understand that anyone who can reach the port can start
matches and spend your API credits. In that mode the server also refuses to
call model endpoints on private or loopback addresses, so that it cannot be
used as a proxy into your network; set `GT_ALLOW_INTERNAL_ENDPOINTS=1` if you
are pointing it at a local Ollama and accept that risk.

## Code battle

Both models are given the same problem, write a solution, and are judged
automatically. Rounds are scored:

1. **Correctness** — more hidden tests passed wins. Every test is counted, so a
   solution that fails one edge case still banks the rest.
2. **Scale** — if both are perfect, the one that survives a much larger
   generated input wins. This is what separates an O(n) solution from an O(n²)
   one; on six-element examples they look identical.
3. **Speed** — still tied, the faster total runtime wins, ignoring differences
   under 5% so rounds are not decided by timing noise.

Submissions are judged one after the other rather than in parallel, so the two
are not competing for the same cores while being timed.

### Where the problems come from

| Source | What it is | Shape |
| --- | --- | --- |
| `local` | 25 curated problems, 8 Easy / 11 Medium / 6 Hard, each with a reference solution and a scaled-input generator | write one function |
| `code_contests` | Real Codeforces problems pulled at run time from [DeepMind's CodeContests](https://huggingface.co/datasets/deepmind/code_contests) archive, carrying their original difficulty rating (800–3500) and real hidden test data | write a program reading stdin |

The remote tier is what makes this genuinely hard: these are problems written
for human competitors, tagged with the algorithms they need, and rated on the
Codeforces scale. Fetching is best-effort and cached — if the network is
unavailable the arena falls back to the local bank and says so.

Fetched problems are filtered before being admitted:

- statements that accept more than one valid answer are dropped, because the
  judge compares against a single expected output and would fail correct
  programs — a silent, match-deciding error;
- interactive problems are dropped, since there is no interactor;
- problems with too few tests are dropped;
- a known-good Python solution shipped with the dataset is run against the
  selected tests, and the problem is only admitted if it passes. This proves
  the tests are consistent, that token comparison is a fair check, and that the
  time limit is reachable from Python.

Use **Settings → Mode → Code Battle** to pick the number of problems, the rating
band, and whether to include the remote tier. `GET /api/bank` reports what is
loaded; `POST /api/bank/refresh` pulls more.

### Sandboxing

Model-written code is hostile input. Each submission runs in a subprocess that
gets no access to the parent environment (so your API keys are not readable),
a throwaway working directory, CPU/memory/file-size limits, and its own session
so a timeout kills the whole process group instead of leaving forked workers
behind.

This is not a security boundary against a determined attacker — that needs a
container or a VM. It contains the realistic failure modes: runaway loops,
memory bombs, stray file writes, and casual exfiltration of the keys sitting in
the server's environment. If you plan to point this at untrusted models, run the
whole thing in a container.

## How a board game works

- A series is `games` games long and the players swap colours each game, so
  game 1 has player 1 as White and game 2 has player 2 as White.
- Each turn the model gets the FEN, an ASCII board from its own point of view,
  the move list, and the full list of legal moves, and is asked for JSON:
  `{"move": "<SAN>", "say": "<one line>"}`.
- Replies are parsed leniently — JSON, fenced JSON, bare SAN, decorated SAN
  (`Nf3!?`), sloppy case, and UCI (`g1f3`) are all accepted.
- An illegal or unparseable move is fed back to the model along with the legal
  move list, and it gets another try. After `retries` bad answers the player
  forfeits the game.
- Network failures have their own separate budget (`network_retries`, 0 means
  retry forever), so a flaky endpoint does not cost anyone a game.
- If a game hits the ply limit it is adjudicated on material, with a five-point
  margin required for a win.
- Personas follow the *player*, not the colour: pick Kasparov for player 1 and
  he stays Kasparov when he has Black in game 2.

## Layout

```
server.py           engine, LLM orchestration, FastAPI app, WebSocket channel
games.py            board games behind one interface (chess, Othello)
challenge.py        the shape of a code-battle problem + answer comparison
codebattle.py       the sandboxed judge and round scoring
problems.py         the curated local problem bank
problembank.py      remote problem sources, filtering, validation, caching
launcher.py         .env loading + startup
static/index.html   the entire UI (no build step, no external assets)
tests/              run directly: python tests/test_arena.py
```

## Tests

```bash
python tests/test_arena.py       # chess engine, colours, pause/resume, scoring
python tests/test_codebattle.py  # judge, sandbox, problem banks
```

Both drive real series against a fake OpenAI-compatible endpoint on loopback.
The code-battle suite includes live checks against the remote archive, which are
skipped when there is no network.
