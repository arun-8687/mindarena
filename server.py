"""
Grand Tensor Local — server-side chess engine + LLM orchestrator.

Backend calls Ollama Cloud (or any OpenAI-compatible endpoint) directly,
bypassing the browser CORS wall that blocks grandtensor.shantanugoel.com
from reaching ollama.com/v1.

Exposes:
  GET  /            -> monitor page
  GET  /api/models  -> list models from configured endpoint
  WS   /ws          -> live game stream (JSON state deltas)
  POST /api/control -> {action: start|pause|reset, settings?: {...}}
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import chess
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response


# ---------------------------------------------------------------------------
# Config — loaded from env / .env by launcher.py
# ---------------------------------------------------------------------------
ENDPOINT = os.environ.get("GT_BASE_URL", "https://ollama.com/v1")
_env_key_var = "OLLAMA" + "_API_" + "KEY"
API_KEY = os.environ.get(_env_key_var, os.environ.get("GT_API_KEY", ""))
DEFAULT_P0_MODEL = os.environ.get("GT_P0_MODEL", "kimi-k3")
DEFAULT_P1_MODEL = os.environ.get("GT_P1_MODEL", "deepseek-v4-pro")
DEFAULT_P0_LABEL = os.environ.get("GT_P0_LABEL", "Kimi K3")
DEFAULT_P1_LABEL = os.environ.get("GT_P1_LABEL", "DeepSeek V4 Pro")
HOST = os.environ.get("GT_HOST", "0.0.0.0")
PORT = int(os.environ.get("GT_PORT", "8791"))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class PlayerCfg:
    label: str = ""
    model: str = ""
    effort: str = "default"      # default | max | high | medium | low | none
    temperature: float = 0.2
    base_url: str = ""           # per-player endpoint override; "" = use global
    api_key: str = ""            # per-player api key override; "" = use global
    rating: int = 0              # display-only Elo badge; 0 = hide

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
    retries: int = 5
    network_retries: int = 0   # 0 = infinite
    max_tokens: int = 16000
    commentary: bool = True
    include_previous: bool = True
    speed_ms: int = 0          # delay between moves
    prompt_template: str = (
        "You are {player}, playing a game of chess as {color} against {opponent}.\n"
        "This is game {gameNumber} of {totalGames}.\n\n"
        "FEN: {fen}\n\n{board}\n\n"
        "Move number: {moveNumber}\n"
        "Last move: {lastMove}\n"
        "Check status: {inCheck}\n\n"
        "Moves so far: {moves}\n\n"
        "LEGAL MOVES ({legalMoveCount}): {legalMoves}\n\n"
        "Choose your move."
    )
    system_white: str = (
        "You are playing a game of chess as white.\n"
        'Respond with JSON: {"move": "<SAN>", "say": "<one short sentence of trash talk or reasoning, max 12 words>"}'
    )
    system_black: str = (
        "You are playing a game of chess as black.\n"
        'Respond with JSON: {"move": "<SAN>", "say": "<one short sentence of trash talk or reasoning, max 12 words>"}'
    )

@dataclass
class PlayerStats:
    label: str = ""
    model: str = ""
    color: str = ""
    effort: str = "default"
    rating: int = 39
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
    last_move: str = ""
    last_say: str = ""
    last_player_idx: int = -1
    plies: int = 0
    move_list: list[str] = field(default_factory=list)
    battle_log: list[dict] = field(default_factory=list)
    players: list[PlayerStats] = field(default_factory=lambda: [PlayerStats(), PlayerStats()])
    game_results: list[dict] = field(default_factory=list)
    total_cost: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class GrandTensorEngine:
    def __init__(self):
        self.settings = Settings()
        self.state = GameState(total_games=self.settings.games)
        self.ws_clients: set[WebSocket] = set()
        self._task: Optional[asyncio.Task] = None
        self._pause = asyncio.Event()
        self._stop = asyncio.Event()
        self._pause.set()  # not paused initially
        self._stop.set()   # not running initially
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0))
        self._reset_stats()

    def _reset_stats(self, hard: bool = True):
        s = self.settings
        prev_log = [] if hard else list(self.state.battle_log)
        prev_results = [] if hard else list(self.state.game_results)
        prev_score = "0–0" if hard else self.state.series_score
        prev_status = "IDLE" if hard else self.state.status
        self.state = GameState(
            total_games=s.games,
            battle_log=prev_log,
            game_results=prev_results,
            series_score=prev_score,
            status=prev_status,
        )
        for i, p in enumerate(s.players):
            ps = PlayerStats(
                label=p.label, model=p.model,
                color="white" if i == 0 else "black",
                effort=p.effort,
                rating=p.rating,
            )
            if len(self.state.players) > i:
                self.state.players[i] = ps
            else:
                self.state.players.append(ps)
        self.state.players = self.state.players[:2]

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
            "last_move": st.last_move,
            "last_say": st.last_say,
            "last_player_idx": st.last_player_idx,
            "plies": st.plies,
            "moves": st.move_list,
            "battle_log": st.battle_log[-200:],
            "players": [_ps_dict(p) for p in st.players],
            "game_results": st.game_results,
            "total_cost": st.total_cost,
            "error": st.error,
            "settings": _settings_dict(self.settings),
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
        import datetime
        self.state.battle_log.append({
            "t": datetime.datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "msg": msg,
        })
        # cap at 400
        if len(self.state.battle_log) > 400:
            self.state.battle_log = self.state.battle_log[-400:]
        await self._broadcast({"type": "log", "level": level, "msg": msg})

    async def _broadcast_state(self):
        await self._broadcast(self.full_snapshot())

    # -- control ------------------------------------------------------------
    async def start(self, new_settings: Optional[dict] = None):
        if new_settings:
            self._apply_settings(new_settings)
        # If a series is currently running, ignore the start (use reset first)
        if self._task and not self._task.done():
            return  # already running
        # Clean up any completed task reference
        self._task = None
        # fresh series — hard reset stats, keep settings
        self._reset_stats(hard=True)
        self._stop.clear()
        self._pause.set()
        self.state.status = "LIVE"
        self.state.error = None
        self._task = asyncio.create_task(self._run_series())
        await self._broadcast_state()

    async def pause(self):
        self._pause.clear()

    async def resume(self):
        self._pause.set()

    async def reset(self):
        self._stop.set()
        self._pause.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None
        self._reset_stats()
        self.state.status = "IDLE"
        await self._broadcast_state()

    def _apply_settings(self, d: dict):
        s = self.settings
        s.base_url = d.get("base_url", s.base_url)
        if d.get("api_key"):
            s.api_key = d["api_key"]
        s.games = int(d.get("games", s.games))
        s.max_plies = int(d.get("max_plies", s.max_plies))
        s.retries = int(d.get("retries", s.retries))
        s.network_retries = int(d.get("network_retries", s.network_retries))
        s.max_tokens = int(d.get("max_tokens", s.max_tokens))
        s.commentary = bool(d.get("commentary", s.commentary))
        s.include_previous = bool(d.get("include_previous", s.include_previous))
        s.speed_ms = int(d.get("speed_ms", s.speed_ms))
        if "prompt_template" in d and d["prompt_template"]:
            s.prompt_template = d["prompt_template"]
        if "system_white" in d and d["system_white"]:
            s.system_white = d["system_white"]
        if "system_black" in d and d["system_black"]:
            s.system_black = d["system_black"]
        players = d.get("players", [])
        for i, pd in enumerate(players[:2]):
            if pd.get("label") is not None:
                s.players[i].label = pd.get("label") or s.players[i].label
            if pd.get("model"):
                s.players[i].model = pd["model"]
            if pd.get("effort"):
                s.players[i].effort = pd["effort"]
            if pd.get("temperature") is not None and not (isinstance(pd.get("temperature"), (int, float)) and pd.get("temperature") != pd.get("temperature")):
                try:
                    t = float(pd.get("temperature"))
                    if not (t != t):  # not NaN
                        s.players[i].temperature = t
                except (TypeError, ValueError):
                    pass
            if pd.get("base_url"):
                s.players[i].base_url = pd["base_url"]
            if pd.get("api_key"):
                s.players[i].api_key = pd.get("api_key", s.players[i].api_key)
            if pd.get("rating") is not None:
                s.players[i].rating = int(pd.get("rating"))
        self._reset_stats(hard=False)

    # -- LLM call -----------------------------------------------------------
    async def _llm_move(self, player_idx: int, board: chess.Board,
                        game_no: int, move_no: int, prev_games: str) -> tuple[str, str, dict]:
        """Return (san_move, say, usage). Raises on failure."""
        s = self.settings
        p = s.players[player_idx]
        color = "white" if player_idx == 0 else "black"
        legal = [board.san(m) for m in board.legal_moves]
        prompt = s.prompt_template.format(
            player=p.label, color=color, opponent=s.players[1-player_idx].label,
            gameNumber=game_no, totalGames=s.games,
            fen=board.fen(),
            board=_board_ascii(board, player_idx == 0),
            moveNumber=move_no,
            lastMove=(self.state.last_move or "—"),
            inCheck="yes" if board.is_check() else "no",
            moves=" ".join(self.state.move_list) or "—",
            previousGames=prev_games or "—",
            legalMoveCount=len(legal),
            legalMoves=" ".join(legal),
        )
        system = s.system_white if player_idx == 0 else s.system_black
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]
        body = {
            "model": p.model,
            "messages": messages,
            "temperature": p.temperature,
            "max_tokens": s.max_tokens,
        }
        # reasoning effort hint — pass as extra field most OpenAI-compatible APIs ignore
        if p.effort and p.effort != "default":
            body["reasoning_effort"] = p.effort

        headers = {"Authorization": f"Bearer {p.api_key or s.api_key}",
                   "Content-Type": "application/json"}
        url = (p.base_url or s.base_url).rstrip("/") + "/chat/completions"

        t0 = time.monotonic()
        async with self._http.stream("POST", url, json=body, headers=headers) as r:
            if r.status_code != 200:
                err = (await r.aread()).decode(errors="replace")[:300]
                raise RuntimeError(f"HTTP {r.status_code}: {err}")
            raw = await r.aread()
        dt = int((time.monotonic() - t0) * 1000)
        data = json.loads(raw)
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        usage = data.get("usage", {}) or {}
        move, say = _parse_move(content)
        usage["_ms"] = dt
        usage["_reasoning_tokens"] = (
            usage.get("completion_tokens_details", {}) or {}
        ).get("reasoning_tokens", 0) or 0
        usage["_content"] = content[:400]
        return move, say, usage

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
                # colors alternate: even games swap
                white_idx = 0 if g % 2 == 1 else 1
                # update player colors so the UI attributes moves/say correctly
                self.state.players[white_idx].color = "white"
                self.state.players[1-white_idx].color = "black"
                await self._log("info",
                    f"Game {g}/{s.games}: {s.players[white_idx].label} (W) vs {s.players[1-white_idx].label} (B)")
                await self._play_game(g, white_idx)
                await self._broadcast_state()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.state.status = "ERROR"
            self.state.error = f"{type(e).__name__}: {e}"
            await self._log("error", f"Series error: {e}")
            await self._broadcast_state()
            return
        if self.state.status != "ERROR":
            self.state.status = "DONE"
        await self._broadcast_state()

    async def _play_game(self, game_no: int, white_idx: int):
        s = self.settings
        board = chess.Board()
        self.state.move_list = []
        self.state.last_move = ""
        self.state.last_say = ""
        self.state.last_player_idx = -1
        self.state.plies = 0
        self.state.board_fen = board.fen()
        await self._broadcast_state()

        prev_games = ""
        if s.include_previous and self.state.game_results:
            prev_games = "\n".join(
                f"Game {r['game']}: {r['white']} vs {r['black']} — {r['result']} ({r['moves']})"
                for r in self.state.game_results
            )

        move_no = 0
        while not board.is_game_over(claim_draw=True) and self.state.plies < s.max_plies:
            if self._stop.is_set():
                return
            await self._pause.wait()
            if self._stop.is_set():
                return

            player_idx = white_idx if board.turn == chess.WHITE else (1 - white_idx)
            move_no += 1
            attempt = 0
            success = False
            while attempt < s.retries and not self._stop.is_set():
                attempt += 1
                try:
                    await self._log("thinking",
                        f"{s.players[player_idx].model}: thinking… (move {move_no}, attempt {attempt})")
                    san, say, usage = await self._llm_move(player_idx, board, game_no, move_no, prev_games)
                    # validate
                    legal = [board.san(m) for m in board.legal_moves]
                    mv = _match_san(san, legal)
                    if mv is None:
                        ps = self.state.players[player_idx]
                        ps.illegal += 1
                        await self._log("warn",
                            f"{s.players[player_idx].model}: illegal/unparseable move {san!r} (attempt {attempt}). Legal: {', '.join(legal[:8])}…")
                        await self._broadcast_state()
                        continue
                    # apply
                    board.push_san(mv)
                    ps = self.state.players[player_idx]
                    ps.moves += 1
                    ps.tokens += usage.get("total_tokens", 0) or 0
                    ps.reasoning += usage.get("_reasoning_tokens", 0) or 0
                    ps.cost += float(usage.get("cost") or 0.0)
                    ps.last_ms = usage.get("_ms", 0)
                    ps._move_times.append(ps.last_ms)
                    ps.avg_ms = int(sum(ps._move_times) / len(ps._move_times))
                    self.state.total_cost += float(usage.get("cost") or 0.0)
                    self.state.last_move = mv
                    self.state.last_say = say
                    self.state.last_player_idx = player_idx
                    self.state.move_list.append(mv)
                    self.state.plies = len(self.state.move_list)
                    self.state.board_fen = board.fen()
                    await self._broadcast({"type":"move",
                        "ply": self.state.plies, "san": mv, "say": say,
                        "player_idx": player_idx, "fen": board.fen(),
                        "move_no": move_no})
                    await self._log("move",
                        f"{s.players[player_idx].label}: {mv}" + (f" — \"{say}\"" if say else ""))
                    await self._broadcast_state()
                    success = True
                    break
                except Exception as e:
                    await self._log("warn",
                        f"{s.players[player_idx].model}: {type(e).__name__}: {str(e)[:160]} (attempt {attempt})")
                    await self._broadcast_state()
                    if attempt >= s.retries:
                        break
                    await asyncio.sleep(min(2 ** attempt, 30))
            if not success:
                # forfeit
                winner_idx = 1 - player_idx
                self.state.players[winner_idx].wins += 1
                self.state.players[player_idx].losses += 1
                result = "1-0" if winner_idx == white_idx else "0-1"
                self.state.game_results.append({
                    "game": game_no, "white": s.players[white_idx].label,
                    "black": s.players[1-white_idx].label,
                    "result": result, "reason": "forfeit",
                    "moves": len(self.state.move_list),
                })
                await self._log("error",
                    f"{s.players[player_idx].model} forfeits game {game_no} after {attempt} failed attempts.")
                await self._update_score()
                return
            if s.speed_ms:
                await asyncio.sleep(s.speed_ms / 1000.0)

        # game ended — adjudicate
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            # hit ply limit — adjudicate on material
            wm = _material(board, chess.WHITE)
            bm = _material(board, chess.BLACK)
            if wm - bm >= 5:
                result = "1-0"
            elif bm - wm >= 5:
                result = "0-1"
            else:
                result = "1/2-1/2"
        else:
            result = outcome.result()
        if result == "1-0":
            self.state.players[white_idx].wins += 1
            self.state.players[1-white_idx].losses += 1
        elif result == "0-1":
            self.state.players[1-white_idx].wins += 1
            self.state.players[white_idx].losses += 1
        else:
            self.state.players[white_idx].draws += 1
            self.state.players[1-white_idx].draws += 1
        self.state.game_results.append({
            "game": game_no, "white": s.players[white_idx].label,
            "black": s.players[1-white_idx].label,
            "result": result, "reason": (outcome.termination.name if outcome and outcome.termination else "adjudicated"),
            "moves": len(self.state.move_list),
        })
        await self._update_score()
        await self._log("info",
            f"Game {game_no} result: {result} ({len(self.state.move_list)} moves)")

    async def _update_score(self):
        w, d, l = self.state.players[0].wins, self.state.players[0].draws, self.state.players[0].losses
        self.state.series_score = f"{w}–{l}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ps_dict(p: PlayerStats) -> dict:
    d = asdict(p)
    d.pop("_move_times", None)
    return d

def _settings_dict(s: Settings) -> dict:
    return {
        "base_url": s.base_url,
        "api_key_set": bool(s.api_key),
        "players": [asdict(p) for p in s.players],
        "games": s.games, "max_plies": s.max_plies,
        "retries": s.retries, "network_retries": s.network_retries,
        "max_tokens": s.max_tokens, "commentary": s.commentary,
        "include_previous": s.include_previous, "speed_ms": s.speed_ms,
        "prompt_template": s.prompt_template,
        "system_white": s.system_white, "system_black": s.system_black,
    }

def _board_ascii(board: chess.Board, white_pov: bool) -> str:
    lines = []
    ranks = range(8, 0, -1) if white_pov else range(1, 9)
    for r in ranks:
        row = []
        for f in range(8) if white_pov else range(7, -1, -1):
            sq = chess.square(f, r - 1)
            piece = board.piece_at(sq)
            row.append(piece.symbol() if piece else "·")
        lines.append(" ".join(row))
    return "\n".join(lines)

SAN_RE = re.compile(r'"move"\s*:\s*"([^"]+)"')
SAY_RE = re.compile(r'"say"\s*:\s*"([^"]*)"')

def _parse_move(content: str) -> tuple[str, str]:
    # Try JSON parse first
    move, say = "", ""
    # extract move
    m = SAN_RE.search(content)
    if m:
        move = m.group(1)
    m2 = SAY_RE.search(content)
    if m2:
        say = m2.group(1)
    if not move:
        # fall back: first SAN-like token in the content
        for tok in re.findall(r'\b[OQKRBNa-h1-8+x#=O\-]+\b', content):
            if 2 <= len(tok) <= 8:
                move = tok
                break
    return move.strip(), say.strip()

def _match_san(san: str, legal: list[str]) -> Optional[str]:
    if not san:
        return None
    s = san.strip().rstrip("!?+#")
    # exact
    for l in legal:
        if l == s:
            return l
    # case-insensitive
    slow = s.lower()
    for l in legal:
        if l.lower() == slow:
            return l
    # strip trailing +/# from legal too
    for l in legal:
        if l.rstrip("+#") == s:
            return l
    # try as UCI
    try:
        mv = chess.Move.from_uci(s)
        for l in legal:
            if board_san_from_uci(l, s):
                return l
    except Exception:
        pass
    return None

def board_san_from_uci(san: str, uci: str) -> bool:
    return False  # placeholder; _match_san handles by exact/caseless

def _material(board: chess.Board, color: chess.Color) -> int:
    vals = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3,
            chess.ROOK:5, chess.QUEEN:9}
    return sum(vals.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == color)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
engine = GrandTensorEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine._http.aclose()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def index():
    html = open("/home/arun/grand-tensor-local/static/index.html", "rb").read()
    return Response(html, media_type="text/html",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                             "Pragma": "no-cache", "Expires": "0"})

@app.get("/api/models")
async def list_models():
    s = engine.settings
    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.get(s.base_url.rstrip("/") + "/models",
                            headers={"Authorization": f"Bearer {s.api_key}"})
            if r.status_code != 200:
                return JSONResponse({"error": f"HTTP {r.status_code}", "detail": r.text[:300]}, 500)
            data = r.json()
            ids = [m.get("id", "") for m in data.get("data", [])]
            return {"models": sorted(ids)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

# Provider presets — known endpoints + env key names
PROVIDERS = {
    "ollama": {"base_url": "https://ollama.com/v1", "key_env": "OLLAMA_API_KEY",
               "label": "Ollama Cloud", "models": [
        "deepseek-v4-pro", "deepseek-v4-flash:0731", "deepseek-v4-flash:preview",
        "glm-5.2", "glm-5.1", "kimi-k3", "kimi-k2.7-code", "kimi-k2.6",
        "minimax-m3", "minimax-m2.7", "qwen3.5:397b", "gpt-oss:120b", "gpt-oss:20b",
        "gemma4:31b", "mistral-large-3:675b", "nemotron-3-ultra", "nemotron-3-super",
    ]},
    "opencode_go": {"base_url": "https://opencode.ai/zen/go/v1", "key_env": "OPENCODE_GO_API_KEY",
                    "label": "OpenCode Zen GO", "models": ["gpt-5.6-luna"]},
}

@app.get("/api/providers")
async def list_providers():
    # Return provider presets with whether the env key is set (not the key itself)
    out = {}
    for k, v in PROVIDERS.items():
        out[k] = {
            "base_url": v["base_url"],
            "label": v["label"],
            "key_set": bool(os.environ.get(v["key_env"])),
            "models": v["models"],
        }
    return out

@app.get("/api/provider_key")
async def provider_key(provider: str):
    """Resolve and return the actual env key for a provider (so the UI can fill it in)."""
    p = PROVIDERS.get(provider)
    if not p:
        return JSONResponse({"error": "unknown provider"}, 400)
    key = os.environ.get(p["key_env"], "")
    return {"api_key": key, "base_url": p["base_url"]}

@app.get("/api/models_for")
async def models_for(base_url: str, api_key: str = ""):
    """Fetch model list for a specific endpoint + key (per-player model dropdown)."""
    key = api_key or engine.settings.api_key
    if not key:
        return JSONResponse({"error": "no api key"}, 400)
    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.get(base_url.rstrip("/") + "/models",
                            headers={"Authorization": f"Bearer {key}"})
            if r.status_code != 200:
                return JSONResponse({"error": f"HTTP {r.status_code}", "detail": r.text[:300]}, 500)
            data = r.json()
            ids = [m.get("id", "") for m in data.get("data", [])]
            return {"models": sorted(ids)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

@app.get("/api/state")
async def get_state():
    return engine.full_snapshot()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await engine.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            if raw in ("ping", "Ping"):
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
                await engine._broadcast_state()
            elif action == "resume":
                await engine.resume()
                await engine._broadcast_state()
            elif action == "reset":
                await engine.reset()
            elif action == "update_settings":
                engine._apply_settings(msg.get("settings", {}))
                await engine._broadcast_state()
    except WebSocketDisconnect:
        engine.disconnect(ws)
    except Exception as e:
        engine.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")