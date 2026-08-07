"""End-to-end checks for the MindArena engine.

Runs real series against a fake OpenAI-compatible endpoint served on loopback,
so the engine's HTTP path, retry logic and adjudication are all exercised.

    python tests/test_arena.py
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
import server as S  # noqa: E402


# ---------------------------------------------------------------------------
# Fake endpoint
# ---------------------------------------------------------------------------
class FakeLLM:
    """Serves /v1/chat/completions, recording every request it is sent."""

    def __init__(self, mode: str = "legal"):
        self.mode = mode          # legal | garbage | uci | flaky
        self.requests: list[dict] = []
        self.fail_next = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                try:
                    self.reply()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the engine was reset mid-request; nothing to say

            def reply(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                if outer.fail_next > 0:
                    outer.fail_next -= 1
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"upstream is sulking")
                    return
                reply = outer.reply_for(body)
                payload = json.dumps({
                    "choices": [{"message": {"content": reply}}],
                    "usage": {"total_tokens": 100,
                              "completion_tokens_details": {"reasoning_tokens": 40}},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.handle_error = lambda *a: None
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    @staticmethod
    def board_from(body: dict) -> chess.Board:
        prompt = body["messages"][1]["content"]
        fen = prompt.split("FEN: ", 1)[1].split("\n", 1)[0].strip()
        return chess.Board(fen)

    def reply_for(self, body: dict) -> str:
        if self.mode == "garbage":
            return '{"move": "Qz9", "say": "unstoppable"}'
        board = self.board_from(body)
        move = random.choice(list(board.legal_moves))
        if self.mode == "uci":
            return f"I play {move.uci()}"          # bare UCI, no JSON at all
        return json.dumps({"move": board.san(move), "say": "your move, meatbag"})


def configure(engine: S.MindArenaEngine, fake: FakeLLM, **overrides):
    engine._apply_settings({
        "base_url": fake.base_url,
        "api_key": "test-key",
        "players": [
            {"label": "Alpha", "model": "alpha-1", "base_url": fake.base_url,
             "api_key": "test-key", "system": "You are Alpha playing as {color}."},
            {"label": "Beta", "model": "beta-1", "base_url": fake.base_url,
             "api_key": "test-key", "system": "You are Beta playing as {color}."},
        ],
        **overrides,
    })


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
CHECKS = []
def check(fn):
    CHECKS.append(fn)
    return fn


@check
async def colours_follow_the_board_not_the_player_slot():
    """Regression: even-numbered games told the white player it was black."""
    random.seed(7)
    with FakeLLM() as fake:
        eng = S.MindArenaEngine()
        configure(eng, fake, games=2, max_plies=8)
        await eng.start()
        await wait_done(eng)

        assert len(eng.state.game_results) == 2, eng.state.game_results
        mismatches = []
        for body in fake.requests:
            board = FakeLLM.board_from(body)
            claimed_user = "white" if "as white against" in body["messages"][1]["content"] else "black"
            claimed_sys = "white" if "as white" in body["messages"][0]["content"] else "black"
            actual = "white" if board.turn == chess.WHITE else "black"
            if claimed_user != actual or claimed_sys != actual:
                mismatches.append((board.fen(), actual, claimed_user, claimed_sys))
        assert not mismatches, f"{len(mismatches)} prompts stated the wrong colour: {mismatches[:2]}"


@check
async def personas_stay_with_their_player_across_colour_swaps():
    random.seed(11)
    with FakeLLM() as fake:
        eng = S.MindArenaEngine()
        configure(eng, fake, games=2, max_plies=4)
        await eng.start()
        await wait_done(eng)

        game2 = [b for b in fake.requests if "game 2 of 2" in b["messages"][1]["content"]]
        assert game2, "no game 2 requests recorded"
        for body in game2:
            system = body["messages"][0]["content"]
            model = body["model"]
            expected = "Alpha" if model == "alpha-1" else "Beta"
            assert system.startswith(f"You are {expected}"), (model, system)
        # In game 2 Alpha has black, and its persona must say so.
        alpha_g2 = [b for b in game2 if b["model"] == "alpha-1"]
        assert alpha_g2 and "as black" in alpha_g2[0]["messages"][0]["content"], \
            alpha_g2[0]["messages"][0]["content"]


@check
async def editing_settings_mid_series_keeps_the_game():
    """Regression: nudging the speed slider used to wipe the board and score."""
    random.seed(3)
    with FakeLLM() as fake:
        eng = S.MindArenaEngine()
        configure(eng, fake, games=1, max_plies=200, speed_ms=0)
        await eng.start()
        while eng.state.plies < 6:
            await asyncio.sleep(0.02)
        await settle(eng)
        before = (eng.state.plies, eng.state.board_fen, eng.state.game_no,
                  eng.state.players[0].moves, eng.state.players[0].tokens)
        eng._apply_settings({"speed_ms": 500})
        after = (eng.state.plies, eng.state.board_fen, eng.state.game_no,
                 eng.state.players[0].moves, eng.state.players[0].tokens)
        assert before == after, f"settings update mutated live state:\n{before}\n{after}"
        await eng.resume()
        assert eng.state.status == "LIVE", eng.state.status
        await teardown(eng)


@check
async def pause_resume_reports_status():
    random.seed(5)
    with FakeLLM() as fake:
        eng = S.MindArenaEngine()
        configure(eng, fake, games=1, max_plies=200)
        await eng.start()
        await asyncio.sleep(0.05)
        plies = await settle(eng)
        assert eng.state.status == "PAUSED", eng.state.status
        await asyncio.sleep(0.4)
        assert eng.state.plies == plies, "moves kept arriving while paused"
        await eng.resume()
        assert eng.state.status == "LIVE"
        await asyncio.sleep(0.4)
        assert eng.state.plies > plies, "resume did not restart the game"
        await eng.reset()
        assert eng.state.status == "IDLE"
        await teardown(eng)


@check
async def bare_uci_replies_are_accepted():
    """Models emit UCI whatever the prompt says; the old UCI path was a stub."""
    random.seed(13)
    with FakeLLM("uci") as fake:
        eng = S.MindArenaEngine()
        configure(eng, fake, games=1, max_plies=6, retries=2)
        await eng.start()
        await wait_done(eng)
        assert eng.state.plies == 6, f"expected 6 plies, got {eng.state.plies}"
        assert eng.state.players[0].illegal == 0, "UCI replies were rejected as illegal"


@check
async def persistent_nonsense_forfeits_and_gets_feedback():
    with FakeLLM("garbage") as fake:
        eng = S.MindArenaEngine()
        configure(eng, fake, games=1, retries=3)
        await eng.start()
        await wait_done(eng)

        assert eng.state.game_results, "no result recorded"
        result = eng.state.game_results[0]
        assert result["reason"] == "forfeit", result
        assert result["result"] == "0-1", result          # White (player 0) forfeited
        assert eng.state.players[0].illegal == 3, eng.state.players[0].illegal
        assert len(fake.requests) == 3, len(fake.requests)
        # Retries after the first must carry the rejection back to the model.
        assert len(fake.requests[1]["messages"]) == 3, fake.requests[1]["messages"]
        assert "not a legal move" in fake.requests[1]["messages"][2]["content"]


@check
async def transient_failures_do_not_cost_the_game():
    random.seed(17)
    with FakeLLM() as fake:
        eng = S.MindArenaEngine()
        configure(eng, fake, games=1, max_plies=2, retries=2, network_retries=0)
        fake.fail_next = 3        # more 503s than the bad-answer budget allows
        await eng.start()
        await wait_done(eng, timeout=60)
        assert eng.state.plies == 2, f"expected the game to survive, got {eng.state.plies} plies"
        assert not any(r["reason"] == "forfeit" for r in eng.state.game_results)


@check
def settings_snapshot_never_carries_a_key():
    eng = S.MindArenaEngine()
    eng._apply_settings({
        "api_key": "global-secret",
        "players": [{"api_key": "player-secret"}, {}],
    })
    blob = json.dumps(eng.full_snapshot())
    assert "global-secret" not in blob and "player-secret" not in blob, "snapshot leaked a key"
    assert eng.full_snapshot()["settings"]["players"][0]["api_key_set"] is True
    assert eng.settings.players[0].api_key == "player-secret", "key was not stored server-side"


@check
def move_parsing_is_not_fooled_by_english():
    board = chess.Board()
    move, _ = S._parse_move("That was a bad idea, he faced a decade of defeat.", board)
    assert move == "", f"prose was misread as the move {move!r}"

    move, _ = S._parse_move('```json\n{"move": "Nf3", "say": "hi"}\n```', board)
    assert move == "Nf3", move
    move, say = S._parse_move('  {"move":"e4","say":"pawn to king four"} ', board)
    assert (move, say) == ("e4", "pawn to king four"), (move, say)
    move, _ = S._parse_move("I'll go with Nc3 here.", board)
    assert move == "Nc3", move
    move, _ = S._parse_move("I play g1f3", board)          # bare UCI, no JSON
    assert move == "g1f3", move
    # A model thinking out loud names its actual choice last.
    move, _ = S._parse_move("I considered Nf3, but I'll play e4.", board)
    assert move == "e4", move


def _resolves(raw, expect):
    board = chess.Board()
    move = S._resolve_move(board, raw)
    assert move is not None and board.san(move) == expect, f"{raw!r} -> {move}"


@check
def move_resolution_accepts_what_models_actually_emit():
    _resolves("Nf3", "Nf3")
    _resolves("Nf3!?", "Nf3")
    _resolves("nf3", "Nf3")
    _resolves("g1f3", "Nf3")
    _resolves('"e4"', "e4")
    assert S._resolve_move(chess.Board(), "Qz9") is None
    assert S._resolve_move(chess.Board(), "") is None
    # A promotion in UCI, from a position where one is available.
    b = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
    assert b.san(S._resolve_move(b, "a7a8q")) == "a8=Q"


@check
def series_score_counts_draws_as_half():
    def score(w0, d0, l0, w1, d1, l1):
        st = S.GameState()
        st.players[0].wins, st.players[0].draws, st.players[0].losses = w0, d0, l0
        st.players[1].wins, st.players[1].draws, st.players[1].losses = w1, d1, l1
        return f"{S._points(st.players[0])}–{S._points(st.players[1])}"
    assert score(0, 0, 0, 0, 0, 0) == "0–0"
    assert score(0, 1, 0, 0, 1, 0) == "½–½", score(0, 1, 0, 0, 1, 0)
    assert score(1, 1, 0, 0, 1, 1) == "1½–½", score(1, 1, 0, 0, 1, 1)
    assert score(2, 0, 0, 0, 0, 2) == "2–0"


@check
def custom_prompts_with_json_braces_do_not_explode():
    """str.format used to raise KeyError on the JSON braces in a persona."""
    out = S._render('Play as {color}. Reply {"move": "<SAN>"} and nothing else.',
                    {"color": "black"})
    assert out == 'Play as black. Reply {"move": "<SAN>"} and nothing else.', out


@check
def index_is_served_from_beside_the_module():
    assert S.INDEX_HTML.is_file(), f"{S.INDEX_HTML} missing"
    assert "/home/arun" not in Path(S.__file__).read_text(), "hardcoded author path still present"


# ---------------------------------------------------------------------------
async def wait_done(engine: S.MindArenaEngine, timeout: float = 30.0):
    task = engine._task
    if task:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    await engine._http.aclose()


async def settle(engine: S.MindArenaEngine) -> int:
    """Pause, then let any in-flight move land. Returns the settled ply count."""
    await engine.pause()
    plies = -1
    while plies != engine.state.plies:
        plies = engine.state.plies
        await asyncio.sleep(0.15)
    return plies


async def teardown(engine: S.MindArenaEngine):
    await engine.reset()
    await engine._http.aclose()


def main() -> int:
    failed = 0
    for fn in CHECKS:
        name = fn.__name__.replace("_", " ")
        try:
            asyncio.run(fn()) if asyncio.iscoroutinefunction(fn) else fn()
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
