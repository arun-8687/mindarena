"""
MindArena — server-side chess engine + LLM orchestrator.

The backend calls Ollama Cloud (or any OpenAI-compatible endpoint) directly,
bypassing the browser CORS wall that blocks a static page from reaching
ollama.com/v1.

Exposes:
  GET  /                -> monitor page
  GET  /api/state       -> current snapshot
  GET  /api/providers   -> provider presets (never includes secrets)
  GET  /api/models      -> models from the globally configured endpoint
  GET  /api/models_for  -> models for a provider preset or an explicit base URL
  WS   /ws              -> live game stream + control channel
"""
from __future__ import annotations

import asyncio
import datetime
import ipaddress
import json
import os
import re
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import chess
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from games import create_engine, GAMES, GameEngine
from fastapi.responses import JSONResponse
from starlette.responses import Response


# ---------------------------------------------------------------------------
# Config — loaded from env / .env by launcher.py
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

ENDPOINT = os.environ.get("GT_BASE_URL", "https://ollama.com/v1")
API_KEY = os.environ.get("OLLAMA_API_KEY", os.environ.get("GT_API_KEY", ""))
DEFAULT_P0_MODEL = os.environ.get("GT_P0_MODEL", "kimi-k3")
DEFAULT_P1_MODEL = os.environ.get("GT_P1_MODEL", "deepseek-v4-pro")
DEFAULT_P0_LABEL = os.environ.get("GT_P0_LABEL", "Kimi K3")
DEFAULT_P1_LABEL = os.environ.get("GT_P1_LABEL", "DeepSeek V4 Pro")

# Default to loopback: this server has no authentication and holds API keys.
# Set GT_HOST=0.0.0.0 deliberately if you want it reachable on your LAN/tailnet.
HOST = os.environ.get("GT_HOST", "127.0.0.1")
PORT = int(os.environ.get("GT_PORT", "8791"))

# Pointing a player at a local Ollama (http://localhost:11434/v1) is a normal
# thing to do, so internal addresses are allowed while we are bound to
# loopback. Once the server is exposed, anyone who can reach it can also choose
# the endpoints it calls, so internal targets are refused unless you opt back in.
_LOOPBACK_BINDS = {"127.0.0.1", "localhost", "::1"}
ALLOW_INTERNAL_ENDPOINTS = (
    os.environ.get("GT_ALLOW_INTERNAL_ENDPOINTS", "").strip().lower() in ("1", "true", "yes")
    or HOST in _LOOPBACK_BINDS
)

# Provider presets — known endpoints + the env var holding their key.
# `models` is only a starter list; the live /v1/models listing is merged on top.
PROVIDERS: dict[str, dict] = {
    "ollama": {
        "base_url": "https://ollama.com/v1",
        "key_env": "OLLAMA_API_KEY",
        "label": "Ollama Cloud",
        "models": [
            "deepseek-v4-pro", "deepseek-v4-flash:0731", "deepseek-v4-flash:preview",
            "glm-5.2", "glm-5.1", "kimi-k3", "kimi-k2.7-code", "kimi-k2.6",
            "minimax-m3", "minimax-m2.7", "qwen3.5:397b", "gpt-oss:120b", "gpt-oss:20b",
            "gemma4:31b", "mistral-large-3:675b", "nemotron-3-ultra", "nemotron-3-super",
            "nemotron-3-nano:30b",
        ],
    },
    "opencode_go": {
        "base_url": "https://opencode.ai/zen/go/v1",
        "key_env": "OPENCODE_GO_API_KEY",
        "label": "OpenCode Zen GO",
        "models": [
            "gpt-5.6-luna", "grok-4.5", "deepseek-v4-pro", "deepseek-v4-flash",
            "glm-5", "glm-5.1", "glm-5.2", "kimi-k3", "kimi-k2.7-code", "kimi-k2.6",
            "kimi-k2.5", "minimax-m3", "minimax-m2.7", "minimax-m2.5",
            "qwen3.8-max", "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus", "qwen3.5-plus",
            "mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro", "mimo-v2-omni",
            "hy3", "hy3-preview",
        ],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "label": "OpenRouter (free models)",
        "models": [],  # populated at startup by _refresh_provider_models()
    },
    "opencode_zen": {
        "base_url": "https://opencode.ai/zen/v1",
        "key_env": "OPENCODE_GO_API_KEY",  # same key works for Go and Zen
        "label": "OpenCode Zen (free models)",
        "models": [],
        "free_filter": lambda m: m.endswith("-free") or "free" in m,
    },
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class PlayerCfg:
    label: str = ""
    model: str = ""
    provider: str = ""           # preset key; blank = custom endpoint
    effort: str = "default"      # default | max | high | medium | low | none
    temperature: float = 0.2
    base_url: str = ""           # per-player endpoint override; "" = use global
    api_key: str = ""            # per-player key override; "" = provider env / global
    rating: int = 0              # display-only Elo badge; 0 = hide
    # Persona. "{color}" is substituted with the colour this player has in the
    # current game, so a persona follows the player when colours alternate.
    system: str = ""

@dataclass
class Settings:
    base_url: str = ENDPOINT
    api_key: str = API_KEY
    players: list[PlayerCfg] = field(default_factory=lambda: [
        PlayerCfg(DEFAULT_P0_LABEL, DEFAULT_P0_MODEL),
        PlayerCfg(DEFAULT_P1_LABEL, DEFAULT_P1_MODEL),
    ])
    games: int = 4
    max_plies: int = 200
    retries: int = 5           # bad/illegal answers tolerated before forfeit
    network_retries: int = 0   # transport failures tolerated; 0 = retry forever
    max_tokens: int = 16000
    commentary: bool = True    # ask for (and show) the per-move one-liner
    include_previous: bool = True
    speed_ms: int = 0          # delay between moves
    game_type: str = "chess"   # chess | othello
    prompt_template: str = (
        "You are {player}, playing a game of chess as {color} against {opponent}.\n"
        "This is game {gameNumber} of {totalGames}.\n"
        "Earlier games: {previousGames}\n\n"
        "FEN: {fen}\n\n{board}\n\n"
        "Move number: {moveNumber}\n"
        "Last move: {lastMove}\n"
        "Check status: {inCheck}\n\n"
        "Moves so far: {moves}\n\n"
        "LEGAL MOVES ({legalMoveCount}): {legalMoves}\n\n"
        "Choose your move."
    )

DEFAULT_SYSTEM = (
    "You are playing a game of chess as {color}.\n"
    'Respond with JSON: {"move": "<SAN>", "say": "<one short sentence of trash talk or reasoning, max 12 words>"}'
)
DEFAULT_SYSTEM_NO_SAY = (
    "You are playing a game of chess as {color}.\n"
    'Respond with JSON: {"move": "<SAN>"}'
)

@dataclass
class PlayerStats:
    label: str = ""
    model: str = ""
    color: str = ""
    effort: str = "default"
    rating: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    moves: int = 0
    tokens: int = 0
    reasoning: int = 0
    illegal: int = 0
    cost: float = 0.0
    last_ms: int = 0
    avg_ms: int = 0
    _move_times: list = field(default_factory=list)

@dataclass
class GameState:
    status: str = "IDLE"        # IDLE | LIVE | PAUSED | DONE | ERROR
    game_no: int = 0
    total_games: int = 4
    series_score: str = "0–0"
    board_fen: str = chess.STARTING_FEN
    othello_grid: str = ""
    last_move: str = ""
    last_from: str = ""
    last_to: str = ""
    check_square: str = ""
    last_say: str = ""
    last_player_idx: int = -1
    plies: int = 0
    move_list: list[str] = field(default_factory=list)
    battle_log: list[dict] = field(default_factory=list)
    players: list[PlayerStats] = field(default_factory=lambda: [PlayerStats(), PlayerStats()])
    game_results: list[dict] = field(default_factory=list)
    total_cost: float = 0.0
    error: str = ""


class TransientLLMError(RuntimeError):
    """Endpoint hiccup (timeout, connection reset, 5xx, 429). Worth retrying."""


class FatalLLMError(RuntimeError):
    """Endpoint said no in a way retrying will not fix (401, 404, bad model)."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class MindArenaEngine:
    def __init__(self):
        self.settings = Settings()
        self.state = GameState(total_games=self.settings.games)
        self.ws_clients: set[WebSocket] = set()
        self._task: Optional[asyncio.Task] = None
        self._pause = asyncio.Event()
        self._stop = asyncio.Event()
        self._pause.set()  # not paused initially
        self._stop.set()   # not running initially
        self._log_seq = 0
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0))
        self._reset_series()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # -- state lifecycle ----------------------------------------------------
    def _reset_series(self):
        """Wipe the board, the counters and the series record. Keeps settings."""
        self.state = GameState(total_games=self.settings.games)
        self._sync_player_meta()

    def _sync_player_meta(self):
        """Push labels/models/ratings from settings onto the live stats rows.

        Deliberately does NOT touch counters — editing settings mid-series must
        not erase the score, the move list or the board.
        """
        while len(self.state.players) < len(self.settings.players):
            self.state.players.append(PlayerStats())
        del self.state.players[len(self.settings.players):]
        for i, p in enumerate(self.settings.players):
            ps = self.state.players[i]
            ps.label = p.label
            ps.model = p.model
            ps.effort = p.effort
            ps.rating = p.rating
            if not ps.color:
                ps.color = "white" if i == 0 else "black"

    # -- websocket plumbing -------------------------------------------------
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.ws_clients.add(ws)
        await ws.send_json(self.full_snapshot())

    def disconnect(self, ws: WebSocket):
        self.ws_clients.discard(ws)

    def full_snapshot(self) -> dict:
        st = self.state
        return {
            "type": "snapshot",
            "status": st.status,
            "game_no": st.game_no,
            "total_games": st.total_games,
            "series_score": st.series_score,
            "fen": st.board_fen,
            "othello_grid": st.othello_grid,
            "last_move": st.last_move,
            "last_from": st.last_from,
            "last_to": st.last_to,
            "check_square": st.check_square,
            "last_say": st.last_say,
            "last_player_idx": st.last_player_idx,
            "plies": st.plies,
            "moves": st.move_list,
            "battle_log": st.battle_log[-200:],
            "players": [_ps_dict(p) for p in st.players],
            "game_results": st.game_results,
            "total_cost": st.total_cost,
            "error": st.error,
            "game_type": self.settings.game_type,
            "settings": _settings_dict(self.settings),
        }

    def _stats_delta(self) -> dict:
        return {
            "type": "stats",
            "status": self.state.status,
            "game_no": self.state.game_no,
            "total_games": self.state.total_games,
            "series_score": self.state.series_score,
            "plies": self.state.plies,
            "total_cost": self.state.total_cost,
            "players": [_ps_dict(p) for p in self.state.players],
        }

    async def _broadcast(self, msg: dict):
        dead = []
        for ws in list(self.ws_clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.discard(ws)

    async def _log(self, level: str, msg: str):
        """Store in state.battle_log AND broadcast live."""
        self._log_seq += 1
        entry = {
            "id": self._log_seq,
            "t": datetime.datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "msg": msg,
        }
        self.state.battle_log.append(entry)
        if len(self.state.battle_log) > 400:
            self.state.battle_log = self.state.battle_log[-400:]
        await self._broadcast({"type": "log", **entry})

    async def _broadcast_state(self):
        await self._broadcast(self.full_snapshot())

    async def _broadcast_stats(self):
        await self._broadcast(self._stats_delta())

    async def _set_status(self, status: str):
        self.state.status = status
        await self._broadcast_stats()

    async def _sleep(self, seconds: float):
        """Sleep, but wake immediately if the series is being torn down."""
        if seconds <= 0:
            return
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    # -- control ------------------------------------------------------------
    async def start(self, new_settings: Optional[dict] = None):
        if self.running:
            await self._log("warn", "A series is already running — reset it first.")
            return
        self._task = None
        if new_settings:
            self._apply_settings(new_settings)
        self._reset_series()
        self._stop.clear()
        self._pause.set()
        self.state.status = "LIVE"
        self.state.error = ""
        self._task = asyncio.create_task(self._run_series())
        await self._broadcast_state()

    async def pause(self):
        self._pause.clear()
        if self.state.status == "LIVE":
            await self._set_status("PAUSED")

    async def resume(self):
        self._pause.set()
        if self.state.status == "PAUSED":
            await self._set_status("LIVE")

    async def reset(self):
        self._stop.set()
        self._pause.set()
        task, self._task = self._task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._reset_series()
        self.state.status = "IDLE"
        await self._broadcast_state()

    def _apply_settings(self, d: dict):
        s = self.settings
        s.base_url = d.get("base_url", s.base_url)
        if d.get("api_key"):
            s.api_key = d["api_key"]
        s.games = _clamp_int(d.get("games"), s.games, 1, 100)
        s.max_plies = _clamp_int(d.get("max_plies"), s.max_plies, 2, 1000)
        s.retries = _clamp_int(d.get("retries"), s.retries, 1, 50)
        s.network_retries = _clamp_int(d.get("network_retries"), s.network_retries, 0, 1000)
        s.max_tokens = _clamp_int(d.get("max_tokens"), s.max_tokens, 64, 200000)
        s.speed_ms = _clamp_int(d.get("speed_ms"), s.speed_ms, 0, 60000)
        if "commentary" in d:
            s.commentary = bool(d["commentary"])
        if "include_previous" in d:
            s.include_previous = bool(d["include_previous"])
        s.game_type = d.get("game_type", s.game_type)
        if s.game_type not in GAMES:
            s.game_type = "chess"
        if d.get("prompt_template"):
            s.prompt_template = d["prompt_template"]

        for i, pd in enumerate(d.get("players", [])[:len(s.players)]):
            p = s.players[i]
            if pd.get("label"):
                p.label = pd["label"]
            if pd.get("model"):
                p.model = pd["model"]
            if pd.get("effort"):
                p.effort = pd["effort"]
            if "provider" in pd:
                p.provider = pd["provider"] or ""
            if "system" in pd:
                p.system = pd["system"] or ""
            if pd.get("base_url"):
                p.base_url = pd["base_url"]
            if pd.get("api_key"):
                p.api_key = pd["api_key"]
            if pd.get("rating") is not None:
                p.rating = _clamp_int(pd.get("rating"), p.rating, 0, 4000)
            t = _as_float(pd.get("temperature"))
            if t is not None:
                p.temperature = min(max(t, 0.0), 2.0)

        # Legacy wire format: system prompts used to be keyed by colour. They
        # are really per-player personas, so map them onto the player slots.
        if d.get("system_white"):
            s.players[0].system = d["system_white"]
        if d.get("system_black") and len(s.players) > 1:
            s.players[1].system = d["system_black"]

        if not self.running:
            self.state.total_games = s.games
        self._sync_player_meta()

    # -- LLM call -----------------------------------------------------------
    def _resolve_key(self, p: PlayerCfg) -> str:
        if p.api_key:
            return p.api_key
        if p.provider and p.provider in PROVIDERS:
            key = os.environ.get(PROVIDERS[p.provider]["key_env"], "")
            if key:
                return key
        return self.settings.api_key

    async def _llm_move(self, player_idx: int, color: str, engine: GameEngine,
                        game_no: int, move_no: int, prev_games: str,
                        feedback: str = "") -> tuple[str, str, dict]:
        """Return (parsed_move_str, say, usage). Raises Transient/FatalLLMError."""
        s = self.settings
        p = s.players[player_idx]
        template = engine.prompt_template()
        vals = engine.prompt_values(
            p.label, s.players[1 - player_idx].label, color,
            game_no, s.games, move_no, self.state.last_move,
            self.state.move_list, prev_games)
        prompt = _render(template, vals)
        base_system = p.system or engine.system_prompt(s.commentary, color)
        system = _render(base_system, {"color": color})

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]
        if feedback:
            messages.append({"role": "user", "content": feedback})

        body = {
            "model": p.model,
            "messages": messages,
            "temperature": p.temperature,
            "max_tokens": s.max_tokens,
        }
        # Reasoning-effort hint — an extra field most OpenAI-compatible APIs ignore.
        if p.effort and p.effort != "default":
            body["reasoning_effort"] = p.effort

        url = _endpoint(p.base_url or s.base_url, "/chat/completions")
        headers = {"Authorization": f"Bearer {self._resolve_key(p)}",
                   "Content-Type": "application/json"}

        t0 = time.monotonic()
        try:
            r = await self._http.post(url, json=body, headers=headers)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise TransientLLMError(f"{type(e).__name__}: {e}") from e
        dt = int((time.monotonic() - t0) * 1000)

        if r.status_code != 200:
            detail = r.text[:300]
            err = f"HTTP {r.status_code}: {detail}"
            if r.status_code in (408, 409, 425, 429) or r.status_code >= 500:
                raise TransientLLMError(err)
            raise FatalLLMError(err)

        try:
            data = r.json()
            msg = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise TransientLLMError(f"malformed response: {type(e).__name__}: {e}") from e

        content = msg.get("content") or ""
        usage = data.get("usage") or {}
        move = engine.parse_move(content)
        # Extract "say" from the content for commentary
        say = ""
        if s.commentary:
            m2 = SAY_RE.search(content)
            if m2:
                say = m2.group(1)
        usage["_ms"] = dt
        usage["_reasoning_tokens"] = (
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        )
        return move, (say if s.commentary else ""), usage

    # -- series loop --------------------------------------------------------
    async def _run_series(self):
        s = self.settings
        self.state.status = "LIVE"
        await self._broadcast_state()
        try:
            for g in range(1, s.games + 1):
                if self._stop.is_set():
                    break
                await self._pause.wait()
                if self._stop.is_set():
                    break
                self.state.game_no = g
                # Colours alternate: player 0 has white in odd games.
                white_idx = 0 if g % 2 == 1 else 1
                self.state.players[white_idx].color = "white"
                self.state.players[1 - white_idx].color = "black"
                await self._log("info",
                    f"Game {g}/{s.games}: {s.players[white_idx].label} (W) vs {s.players[1 - white_idx].label} (B)")
                await self._play_game(g, white_idx)
                await self._broadcast_state()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.state.status = "ERROR"
            self.state.error = f"{type(e).__name__}: {e}"
            await self._log("error", f"Series error: {e}")
            await self._broadcast_state()
            return
        if self.state.status != "ERROR":
            self.state.status = "IDLE" if self._stop.is_set() else "DONE"
        await self._broadcast_state()

    async def _play_game(self, game_no: int, white_idx: int):
        s = self.settings
        engine = create_engine(s.game_type)
        self.state.move_list = []
        self.state.last_move = ""
        self.state.last_from = ""
        self.state.last_to = ""
        self.state.check_square = ""
        self.state.last_say = ""
        self.state.last_player_idx = -1
        self.state.plies = 0
        self.state.board_fen = engine.board_state().get("fen", "")
        self.state.othello_grid = engine.board_state().get("othello_grid", "")
        await self._broadcast_state()

        prev_games = ""
        if s.include_previous and self.state.game_results:
            prev_games = "\n".join(
                f"Game {r['game']}: {r['white']} vs {r['black']} — {r['result']} ({r['moves']})"
                for r in self.state.game_results
            )

        move_no = 0
        while not engine.is_game_over() and self.state.plies < s.max_plies:
            if self._stop.is_set():
                return
            await self._pause.wait()
            if self._stop.is_set():
                return

            player_idx = white_idx if engine.current_player() == 0 else (1 - white_idx)
            color = engine.first_color if engine.current_player() == 0 else engine.second_color
            move_no += 1
            move = await self._request_move(player_idx, color, engine, game_no, move_no, prev_games)
            if move is None:
                if self._stop.is_set():
                    return
                await self._forfeit(game_no, white_idx, player_idx)
                return
            await self._apply_move(engine, move, player_idx, move_no)
            await self._sleep(s.speed_ms / 1000.0)

        await self._adjudicate(engine, game_no, white_idx)

    async def _request_move(self, player_idx: int, color: str, engine: GameEngine,
                            game_no: int, move_no: int, prev_games: str) -> Optional[str]:
        """Ask the model until it produces a legal move, or the budget runs out.

        Bad answers and network failures have separate budgets: a flaky endpoint
        should not cost you the game, but a model that keeps hallucinating moves
        should.
        """
        s = self.settings
        label = s.players[player_idx].label or s.players[player_idx].model
        bad_answers = 0
        net_failures = 0
        feedback = ""

        while not self._stop.is_set():
            await self._pause.wait()
            if self._stop.is_set():
                return None
            attempt = bad_answers + net_failures + 1
            await self._log("thinking", f"{label}: thinking… (move {move_no}, attempt {attempt})")
            try:
                raw, say, usage = await self._llm_move(
                    player_idx, color, engine, game_no, move_no, prev_games, feedback)
            except TransientLLMError as e:
                net_failures += 1
                await self._log("warn", f"{label}: {str(e)[:160]} (network attempt {net_failures})")
                if s.network_retries and net_failures >= s.network_retries:
                    await self._log("error", f"{label}: giving up after {net_failures} network failures.")
                    return None
                await self._sleep(min(2 ** min(net_failures, 5), 30))
                continue
            except FatalLLMError as e:
                bad_answers += 1
                await self._log("error", f"{label}: {str(e)[:200]}")
                if bad_answers >= s.retries:
                    return None
                await self._sleep(min(2 ** bad_answers, 30))
                continue

            # Check if the move is legal without applying it.
            # engine.apply() both validates and pushes, so we test by checking
            # against the legal move list (case-insensitive for robustness).
            legal = engine.legal_moves()
            legal_lower = {m.lower(): m for m in legal}
            matched = legal_lower.get(raw.strip().lower())
            if matched is None:
                bad_answers += 1
                self.state.players[player_idx].illegal += 1
                raw_short = raw[:80].replace("\n", " ") if raw else "(empty)"
                await self._log("warn",
                    f"{label}: illegal/unparseable move '{raw_short}' "
                    f"(attempt {bad_answers}). Legal: {', '.join(legal[:8])}…")
                feedback = (
                    f'Your previous answer "{raw}" is not a legal move in this position. '
                    f"Reply with exactly one move from this list: {', '.join(legal)}"
                )
                await self._broadcast_stats()
                if bad_answers >= s.retries:
                    return None
                continue

            self._record_usage(player_idx, usage)
            self.state.last_say = say
            return matched
        return None

    def _record_usage(self, player_idx: int, usage: dict):
        ps = self.state.players[player_idx]
        cost = _as_float(usage.get("cost")) or 0.0
        ps.tokens += int(usage.get("total_tokens") or 0)
        ps.reasoning += int(usage.get("_reasoning_tokens") or 0)
        ps.cost += cost
        ps.last_ms = int(usage.get("_ms") or 0)
        ps._move_times.append(ps.last_ms)
        ps.avg_ms = int(sum(ps._move_times) / len(ps._move_times))
        self.state.total_cost += cost

    async def _apply_move(self, engine: GameEngine, move_str: str,
                          player_idx: int, move_no: int):
        engine.apply(move_str)
        san = engine.last_move_display()
        st = self.state
        ps = st.players[player_idx]
        ps.moves += 1
        st.last_move = san
        # Get from/to squares for UI highlighting (chess-specific; othello returns "")
        try:
            from_sq, to_sq = engine.last_move_squares()
        except (AttributeError, TypeError):
            from_sq, to_sq = "", ""
        st.last_from = from_sq
        st.last_to = to_sq
        # Check square (chess-specific; othello has no check)
        bstate = engine.board_state()
        st.check_square = bstate.get("check_square", "")
        st.last_player_idx = player_idx
        st.move_list.append(san)
        st.plies = len(st.move_list)
        st.board_fen = bstate.get("fen", "")
        st.othello_grid = bstate.get("othello_grid", "")
        await self._broadcast({
            "type": "move", "ply": st.plies, "san": san, "say": st.last_say,
            "player_idx": player_idx, "fen": st.board_fen, "move_no": move_no,
            "from": st.last_from, "to": st.last_to, "check_square": st.check_square,
            "game_type": engine.game_type, "board_state": bstate,
        })
        await self._log("move",
            f"{self.settings.players[player_idx].label}: {san}"
            + (f" — \"{st.last_say}\"" if st.last_say else ""))
        await self._broadcast_stats()

    async def _forfeit(self, game_no: int, white_idx: int, loser_idx: int):
        s = self.settings
        winner_idx = 1 - loser_idx
        self.state.players[winner_idx].wins += 1
        self.state.players[loser_idx].losses += 1
        result = "1-0" if winner_idx == white_idx else "0-1"
        self.state.game_results.append({
            "game": game_no, "white": s.players[white_idx].label,
            "black": s.players[1 - white_idx].label,
            "result": result, "reason": "forfeit",
            "moves": len(self.state.move_list),
        })
        await self._log("error",
            f"{s.players[loser_idx].label} forfeits game {game_no} — no legal move produced.")
        await self._update_score()

    async def _adjudicate(self, engine: GameEngine, game_no: int, white_idx: int):
        s = self.settings
        if engine.is_game_over():
            result, reason = engine.adjudicate()
        else:
            # Hit the ply limit
            result, reason = engine.adjudicate_ply_limit()

        if result == "1-0":
            self.state.players[white_idx].wins += 1
            self.state.players[1 - white_idx].losses += 1
        elif result == "0-1":
            self.state.players[1 - white_idx].wins += 1
            self.state.players[white_idx].losses += 1
        else:
            self.state.players[white_idx].draws += 1
            self.state.players[1 - white_idx].draws += 1

        self.state.game_results.append({
            "game": game_no, "white": s.players[white_idx].label,
            "black": s.players[1 - white_idx].label,
            "result": result, "reason": reason,
            "moves": len(self.state.move_list),
        })
        await self._update_score()
        await self._log("info",
            f"Game {game_no} result: {result} ({reason}, {len(self.state.move_list)} moves)")

    async def _update_score(self):
        p0, p1 = self.state.players[0], self.state.players[1]
        self.state.series_score = f"{_points(p0)}–{_points(p1)}"
        await self._broadcast_stats()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp_int(value, fallback: int, lo: int, hi: int) -> int:
    try:
        if value is None or isinstance(value, bool):
            return fallback
        return min(max(int(value), lo), hi)
    except (TypeError, ValueError):
        return fallback

def _as_float(value) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # reject NaN

def _points(p: PlayerStats) -> str:
    """Chess series score: a draw is half a point, so 2.5 renders as 2½."""
    whole, half = divmod(p.wins * 2 + p.draws, 2)
    if not half:
        return str(whole)
    return f"{whole}½" if whole else "½"

def _render(template: str, values: dict) -> str:
    """Substitute {placeholders} without str.format.

    Prompts and personas contain literal JSON braces — `{"move": "<SAN>"}` —
    which str.format chokes on, turning a user's custom prompt into a KeyError
    that costs them the game.
    """
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    return out

def _ps_dict(p: PlayerStats) -> dict:
    d = asdict(p)
    d.pop("_move_times", None)
    return d

def _settings_dict(s: Settings) -> dict:
    """Client-facing settings. Never includes an API key — only whether one is set."""
    players = []
    for p in s.players:
        d = asdict(p)
        d.pop("api_key", None)
        d["api_key_set"] = bool(p.api_key)
        players.append(d)
    return {
        "base_url": s.base_url,
        "api_key_set": bool(s.api_key),
        "players": players,
        "games": s.games, "max_plies": s.max_plies,
        "retries": s.retries, "network_retries": s.network_retries,
        "max_tokens": s.max_tokens, "commentary": s.commentary,
        "include_previous": s.include_previous, "speed_ms": s.speed_ms,
        "game_type": s.game_type,
        "prompt_template": s.prompt_template,
    }

def _board_ascii(board: chess.Board, white_pov: bool) -> str:
    lines = []
    ranks = range(8, 0, -1) if white_pov else range(1, 9)
    files = range(8) if white_pov else range(7, -1, -1)
    for r in ranks:
        row = []
        for f in files:
            piece = board.piece_at(chess.square(f, r - 1))
            row.append(piece.symbol() if piece else "·")
        lines.append(" ".join(row))
    return "\n".join(lines)

SAN_RE = re.compile(r'"move"\s*:\s*"([^"]+)"')
SAY_RE = re.compile(r'"say"\s*:\s*"([^"]*)"')
JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

def _parse_move(content: str, board: chess.Board) -> tuple[str, str]:
    """Pull (move, say) out of a model reply that is JSON, nearly JSON, or prose."""
    move, say = "", ""
    candidates = [content]
    fence = FENCE_RE.search(content)
    if fence:
        candidates.insert(0, fence.group(1))
    obj = JSON_OBJ_RE.search(content)
    if obj:
        candidates.insert(0, obj.group(0))

    for text in candidates:
        try:
            data = json.loads(text.strip())
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            move = str(data.get("move") or "")
            say = str(data.get("say") or "")
            if move:
                return move.strip(), say.strip()

    m = SAN_RE.search(content)
    if m:
        move = m.group(1)
    m2 = SAY_RE.search(content)
    if m2:
        say = m2.group(1)

    if not move:
        # Last resort: scan prose for a token that is actually a move in this
        # position, in either SAN or UCI. Matching against the real move list
        # (rather than a loose SAN-shaped regex) stops ordinary English words
        # like "bad" or "faced" from being read as moves. Prefer the last
        # match — a model that thinks out loud names its choice at the end.
        playable = set()
        for m in board.legal_moves:
            playable.add(board.san(m).rstrip("+#"))
            playable.add(m.uci())
        for tok in re.split(r"[\s,.;:()\[\]\"'*]+", content):
            bare = tok.strip().rstrip("!?+#")
            if bare and bare in playable:
                move = bare
    return move.strip(), say.strip()

def _resolve_move(board: chess.Board, raw: str) -> Optional[chess.Move]:
    """Turn whatever the model said into a legal Move, or None.

    Accepts SAN ("Nf3"), decorated SAN ("Nf3!?"), sloppy case ("nf3") and UCI
    ("g1f3") — models emit all four regardless of what the prompt asked for.
    """
    if not raw:
        return None
    s = raw.strip().strip('"').strip()
    for candidate in (s, s.rstrip("!?"), s.rstrip("!?+#")):
        if not candidate:
            continue
        try:
            return board.parse_san(candidate)
        except (ValueError, TypeError):
            pass
        try:
            move = board.parse_uci(candidate.lower())
            if move in board.legal_moves:
                return move
        except (ValueError, TypeError):
            pass
    # Case-insensitive SAN match as a final pass.
    want = s.rstrip("!?+#").lower()
    for move in board.legal_moves:
        if board.san(move).rstrip("+#").lower() == want:
            return move
    return None

def _material(board: chess.Board, color: chess.Color) -> int:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9}
    return sum(vals.get(p.piece_type, 0)
               for p in board.piece_map().values() if p.color == color)


# ---------------------------------------------------------------------------
# Endpoint validation
# ---------------------------------------------------------------------------
_HOST_KIND_CACHE: dict[str, bool] = {}

def _endpoint(base_url: str, path: str) -> str:
    """Validate a user-supplied endpoint and join a path onto it.

    The drawer lets you point a player at any base URL, and the server attaches
    a bearer token to whatever it calls. When the server is exposed beyond
    loopback, refuse internal targets so it cannot be used as a proxy to cloud
    metadata services or to things listening on the host.
    """
    parts = urlsplit(base_url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FatalLLMError(f"unsupported endpoint URL: {base_url!r}")
    if not ALLOW_INTERNAL_ENDPOINTS and _is_internal_host(parts.hostname):
        raise FatalLLMError(
            f"refusing to call internal address {parts.hostname} while bound to {HOST}; "
            "set GT_ALLOW_INTERNAL_ENDPOINTS=1 to override")
    return base_url.strip().rstrip("/") + path

def _is_internal_host(hostname: str) -> bool:
    if hostname in _HOST_KIND_CACHE:
        return _HOST_KIND_CACHE[hostname]
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False  # unresolvable — let the HTTP client produce the error
    internal = False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            internal = True
            break
    _HOST_KIND_CACHE[hostname] = internal
    return internal

async def _fetch_models(base_url: str, api_key: str) -> dict:
    if not api_key:
        return {"models": [], "error": "no api key configured for this endpoint"}
    try:
        url = _endpoint(base_url, "/models")
    except FatalLLMError as e:
        return {"models": [], "error": str(e)}
    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.get(url, headers={"Authorization": f"Bearer {api_key}"})
        if r.status_code != 200:
            return {"models": [], "error": f"HTTP {r.status_code}", "detail": r.text[:300]}
        data = r.json()
    except Exception as e:
        return {"models": [], "error": f"{type(e).__name__}: {e}"}
    ids = [m.get("id", "") for m in (data.get("data") or []) if m.get("id")]
    return {"models": sorted(ids)}


async def _refresh_provider_models():
    """Fetch live model lists for all providers at startup.
    Merges the live list with the hardcoded presets so the dropdown never
    loses a model even if the endpoint is temporarily unreachable."""
    async with httpx.AsyncClient(timeout=20.0) as c:
        for key, preset in PROVIDERS.items():
            base_url = preset["base_url"]
            key_env = preset["key_env"]
            api_key = os.environ.get(key_env, "")
            # OpenRouter's /models works without a key; others need one.
            if not api_key and key != "openrouter":
                print(f"[{key}] skipped (no api key in {key_env})", flush=True)
                continue
            try:
                headers = {}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                r = await c.get(f"{base_url.rstrip('/')}/models", headers=headers)
                if r.status_code != 200:
                    print(f"[{key}] refresh failed: HTTP {r.status_code}", flush=True)
                    continue
                data = r.json()
                ids = [m.get("id", "") for m in (data.get("data") or []) if m.get("id")]
                # Filter to free models for providers that have a free_filter
                ff = preset.get("free_filter")
                if key == "openrouter":
                    ids = sorted(m for m in ids if ":free" in m)
                elif ff:
                    ids = sorted(m for m in ids if ff(m))
                else:
                    ids = sorted(ids)
                # Merge with preset list so hardcoded models aren't lost
                merged = list(dict.fromkeys(ids + preset["models"]))
                preset["models"] = merged
                print(f"[{key}] loaded {len(merged)} models ({len(ids)} from API)", flush=True)
            except Exception as e:
                print(f"[{key}] refresh error: {e}", flush=True)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
engine = MindArenaEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await _refresh_provider_models()
    yield
    await engine.reset()
    await engine._http.aclose()

app = FastAPI(title="MindArena", lifespan=lifespan)

NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache", "Expires": "0"}

@app.get("/")
async def index():
    try:
        html = INDEX_HTML.read_bytes()
    except OSError as e:
        return JSONResponse({"error": f"cannot read {INDEX_HTML}: {e}"}, 500)
    return Response(html, media_type="text/html", headers=NO_CACHE)

@app.get("/api/models")
async def list_models():
    s = engine.settings
    return await _fetch_models(s.base_url, s.api_key)

@app.get("/api/providers")
async def list_providers():
    """Provider presets plus whether their env key is set — never the key itself."""
    return {
        k: {
            "base_url": v["base_url"],
            "label": v["label"],
            "key_set": bool(os.environ.get(v["key_env"])),
            "models": v["models"],
        }
        for k, v in PROVIDERS.items()
    }

@app.post("/api/models_for")
async def models_for(req: dict):
    """Models for a provider preset, or for an explicit base URL.

    POST rather than GET because the request may carry an API key, and a key in
    a query string ends up in browser history, referrers and access logs. Keys
    are never echoed back: a provider preset resolves its key from the server's
    own environment, so the browser never has to hold one.
    """
    provider = (req.get("provider") or "").strip()
    base_url = (req.get("base_url") or "").strip()
    api_key = (req.get("api_key") or "").strip()
    if provider:
        preset = PROVIDERS.get(provider)
        if not preset:
            return JSONResponse({"error": "unknown provider"}, 400)
        base_url = base_url or preset["base_url"]
        api_key = api_key or os.environ.get(preset["key_env"], "")
        # Providers with a free_filter or openrouter return only free models
        # from the preset list (already filtered at startup).
        ff = preset.get("free_filter")
        if (provider == "openrouter" or ff) and preset["models"]:
            return {"models": preset["models"]}
    elif not base_url:
        return JSONResponse({"error": "provider or base_url required"}, 400)
    # A missing key or an unreachable endpoint is reported in the body, not as
    # an HTTP error: the question "which models are available?" was answered.
    return await _fetch_models(base_url, api_key or engine.settings.api_key)

@app.get("/api/state")
async def get_state():
    return engine.full_snapshot()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await engine.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            if raw.lower() == "ping":
                await ws.send_text("pong")
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            action = msg.get("action")
            if action == "start":
                await engine.start(msg.get("settings"))
            elif action == "pause":
                await engine.pause()
            elif action == "resume":
                await engine.resume()
            elif action == "reset":
                await engine.reset()
            elif action == "update_settings":
                engine._apply_settings(msg.get("settings") or {})
                await engine._broadcast_state()
    except WebSocketDisconnect:
        pass
    finally:
        engine.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
