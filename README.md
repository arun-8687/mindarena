# MindArena

Two LLMs play a chess series against each other while you watch the board, the
per-move trash talk, and the token/latency/cost tally for each side.

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

## How a match works

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
server.py          engine, LLM orchestration, FastAPI app, WebSocket channel
launcher.py        .env loading + startup
static/index.html  the entire UI (no build step, no external assets)
```
