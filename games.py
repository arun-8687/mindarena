"""
Game engine abstraction for MindArena.

Each game implements the GameEngine protocol. The server's series loop
calls these methods to drive a match between two LLMs.

Supported games:
  - chess   (via python-chess)
  - othello (self-contained, no external dep)
"""

from __future__ import annotations

import re
import abc
import json
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Protocol / Abstract base
# ---------------------------------------------------------------------------

class GameEngine(abc.ABC):
    """A two-player, alternating-turn, perfect-information game.

    The engine is stateful — it holds the current position and advances it
    via ``apply``.  The server creates a fresh engine per game.
    """

    @property
    @abc.abstractmethod
    def game_type(self) -> str:
        """Short identifier, e.g. 'chess' or 'othello'."""

    @property
    def first_color(self) -> str:
        """Color label of the first mover (player 0). Default 'white'."""
        return "white"

    @property
    def second_color(self) -> str:
        """Color label of the second mover (player 1). Default 'black'."""
        return "black"

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset to the starting position."""

    @abc.abstractmethod
    def legal_moves(self) -> list[str]:
        """Return a list of legal-move strings (SAN for chess, cell refs for othello)."""

    @abc.abstractmethod
    def apply(self, move_str: str) -> bool:
        """Apply a move. Return True if it was legal and applied, False otherwise."""

    @abc.abstractmethod
    def is_game_over(self) -> bool:
        """True when the game has ended (win, loss, draw, or no legal moves)."""

    @abc.abstractmethod
    def winner(self) -> Optional[int]:
        """0 if player-0 (first player) won, 1 if player-1 won, None for draw."""

    @abc.abstractmethod
    def current_player(self) -> int:
        """0 or 1 — whose turn it is."""

    @abc.abstractmethod
    def board_ascii(self, pov: int = 0) -> str:
        """Textual board for the LLM prompt."""

    @abc.abstractmethod
    def board_state(self) -> dict:
        """JSON-serializable state for the WebSocket broadcast (FEN, grid, etc)."""

    @abc.abstractmethod
    def last_move_display(self) -> str:
        """Human-readable last move for the log/UI, e.g. 'e4' or 'D3'."""

    @abc.abstractmethod
    def parse_move(self, content: str) -> str:
        """Extract a move string from the LLM's raw text response."""

    @abc.abstractmethod
    def system_prompt(self, commentary: bool, player_color: str) -> str:
        """System prompt telling the LLM the rules and response format."""

    @abc.abstractmethod
    def prompt_template(self) -> str:
        """The user-message template with {placeholders}."""

    @abc.abstractmethod
    def prompt_values(self, player_label: str, opponent_label: str,
                      player_color: str, game_no: int, total_games: int,
                      move_no: int, last_move: str, move_list: list[str],
                      prev_games: str) -> dict:
        """Values dict to fill into the prompt template."""

    @abc.abstractmethod
    def score(self, p0_wins: int, p0_draws: int, p0_losses: int,
              p1_wins: int, p1_draws: int, p1_losses: int) -> str:
        """Format the series score string for the UI."""

    @abc.abstractmethod
    def adjudicate(self) -> tuple[str, str]:
        """If the game ended naturally, return (result_string, reason).
        e.g. ('1-0', 'checkmate') or ('1/2-1/2', 'stalemate').
        Only called when is_game_over() is True."""

    @abc.abstractmethod
    def adjudicate_ply_limit(self) -> tuple[str, str]:
        """When the ply limit is hit, decide a result. Return (result, reason)."""

    @staticmethod
    @abc.abstractmethod
    def grid_size() -> int:
        """Board dimension (8 for chess/othello). Used by UI for grid layout."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
MOVE_RE = re.compile(r'"move"\s*:\s*"([^"]*)"')
SAY_RE = re.compile(r'"say"\s*:\s*"([^"]*)"')


SAN_RE = re.compile(r'"move"\\s*:\\s*"([^"]+)"')


def parse_move_text(content: str, board: chess.Board) -> tuple[str, str]:
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

def resolve_move(board: chess.Board, raw: str) -> Optional[chess.Move]:
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

FENCE_BLOCK_RE = re.compile(r"```[ \t]*([a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.S)

def _strip_code_fences(text: str) -> str:
    """Pull source out of a reply, fenced or not.

    Models wrap code in markdown however firmly the prompt says not to, and some
    add prose around it. Prefer the largest fenced block; fall back to the raw
    text when there are no fences.
    """
    blocks = [body for _lang, body in FENCE_BLOCK_RE.findall(text)]
    if blocks:
        return max(blocks, key=len).strip("\n")
    stripped = text.strip()
    if stripped.startswith("```"):        # unterminated fence
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*[ \t]*\r?\n?", "", stripped)
        stripped = re.sub(r"```\s*$", "", stripped)
    return stripped.strip("\n")


def _extract_json_move(content: str) -> tuple[str, str]:
    """Pull (move, say) out of JSON or fenced JSON."""
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
    # Fallback: regex for "move":"..." and "say":"..."
    m = MOVE_RE.search(content)
    if m:
        move = m.group(1)
    m2 = SAY_RE.search(content)
    if m2:
        say = m2.group(1)
    return move.strip(), say.strip()


# ---------------------------------------------------------------------------
# Chess engine (wraps python-chess)
# ---------------------------------------------------------------------------

import chess

class ChessEngine(GameEngine):
    @property
    def game_type(self) -> str:
        return "chess"

    def __init__(self):
        self.board = chess.Board()
        self._last_san = ""

    def reset(self) -> None:
        self.board = chess.Board()
        self._last_san = ""

    def legal_moves(self) -> list[str]:
        return [self.board.san(m) for m in self.board.legal_moves]

    def apply(self, move_str: str) -> bool:
        try:
            move = self.board.parse_san(move_str)
        except (ValueError, chess.IllegalMoveError):
            # Try UCI
            try:
                move = self.board.parse_uci(move_str)
            except (ValueError, chess.IllegalMoveError):
                return False
        # SAN has to be produced from the position *before* the move, so capture
        # it as we push. Reading it back off move_stack afterwards raises,
        # because by then the move is no longer legal in the current position.
        self._last_san = self.board.san_and_push(move)
        return True

    def is_game_over(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def winner(self) -> Optional[int]:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return None
        if outcome.winner is None:
            return None
        # In chess, WHITE moves first. player 0 = the "white" player
        # but the server maps player_idx differently. We return 0 for
        # "first player (white)" win, 1 for "second player (black)" win.
        return 0 if outcome.winner == chess.WHITE else 1

    def current_player(self) -> int:
        return 0 if self.board.turn == chess.WHITE else 1

    def board_ascii(self, pov: int = 0) -> str:
        white_pov = (pov == 0)
        lines = []
        ranks = range(8, 0, -1) if white_pov else range(1, 9)
        files = range(8) if white_pov else range(7, -1, -1)
        for r in ranks:
            row = []
            for f in files:
                piece = self.board.piece_at(chess.square(f, r - 1))
                row.append(piece.symbol() if piece else "·")
            lines.append(" ".join(row))
        return "\n".join(lines)

    def board_state(self) -> dict:
        return {
            "fen": self.board.fen(),
            "check_square": (chess.square_name(self.board.king(self.board.turn))
                             if self.board.is_check() else ""),
        }

    def last_move_display(self) -> str:
        return self._last_san

    def last_move_squares(self) -> tuple[str, str]:
        """Return (from_square, to_square) for UI highlighting."""
        if self.board.move_stack:
            mv = self.board.move_stack[-1]
            return chess.square_name(mv.from_square), chess.square_name(mv.to_square)
        return "", ""

    def parse_move(self, content: str) -> str:
        """Turn a raw reply into a canonical SAN move, or "" if there isn't one.

        Matching against the moves actually legal in this position (rather than
        a SAN-shaped regex) is what keeps ordinary prose from being read as a
        move, and what lets a bare UCI reply like "g1f3" be accepted — models
        emit UCI regardless of what the prompt asked for.
        """
        raw, _say = parse_move_text(content, self.board)
        if not raw:
            return ""
        move = resolve_move(self.board, raw)
        return self.board.san(move) if move else raw

    def system_prompt(self, commentary: bool, player_color: str) -> str:
        if commentary:
            return (
                f"You are playing a game of chess as {player_color}.\n"
                'Respond with JSON: {"move": "<SAN>", "say": "<one short sentence of trash talk or reasoning, max 12 words>"}'
            )
        return (
            f"You are playing a game of chess as {player_color}.\n"
            'Respond with JSON: {"move": "<SAN>"}'
        )

    def prompt_template(self) -> str:
        return (
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

    def prompt_values(self, player_label, opponent_label, player_color,
                      game_no, total_games, move_no, last_move, move_list, prev_games) -> dict:
        legal = self.legal_moves()
        check_sq = (chess.square_name(self.board.king(self.board.turn))
                    if self.board.is_check() else "")
        return {
            "player": player_label,
            "color": player_color,
            "opponent": opponent_label,
            "gameNumber": game_no,
            "totalGames": total_games,
            "fen": self.board.fen(),
            "board": self.board_ascii(0 if player_color == "white" else 1),
            "moveNumber": move_no,
            "lastMove": last_move or "—",
            "inCheck": "yes" if self.board.is_check() else "no",
            "moves": " ".join(move_list) or "—",
            "previousGames": prev_games or "—",
            "legalMoveCount": len(legal),
            "legalMoves": " ".join(legal),
        }

    def score(self, p0_w, p0_d, p0_l, p1_w, p1_d, p1_l) -> str:
        def _pts(wins, draws):
            whole, half = divmod(wins * 2 + draws, 2)
            if not half:
                return str(whole)
            return f"{whole}½" if whole else "½"
        return f"{_pts(p0_w, p0_d)}–{_pts(p1_w, p1_d)}"

    def adjudicate(self) -> tuple[str, str]:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return ("1/2-1/2", "no-result")
        return (outcome.result(), outcome.termination.name)

    def adjudicate_ply_limit(self) -> tuple[str, str]:
        margin = _chess_material(self.board, chess.WHITE) - _chess_material(self.board, chess.BLACK)
        result = "1-0" if margin >= 5 else "0-1" if margin <= -5 else "1/2-1/2"
        return (result, "adjudicated")

    @staticmethod
    def grid_size() -> int:
        return 8


def _chess_material(board: chess.Board, color: chess.Color) -> int:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    total = 0
    for pt in chess.PIECE_TYPES:
        total += vals[pt] * len(board.pieces(pt, color))
    return total


# ---------------------------------------------------------------------------
# Othello engine (self-contained)
# ---------------------------------------------------------------------------

# Othello uses an 8x8 board. Cells are referred to as column-letter + row-number,
# e.g. "D3", "E5", etc. — same coordinate style as chess but columns A-H uppercase
# and rows 1-8. Black moves first in standard Othello.

EMPTY = 0
BLACK = 1  # player 0 (first player)
WHITE = 2  # player 1 (second player)

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class OthelloEngine(GameEngine):
    @property
    def game_type(self) -> str:
        return "othello"

    @property
    def first_color(self) -> str:
        return "black"

    @property
    def second_color(self) -> str:
        return "white"

    def __init__(self):
        self.grid = [[EMPTY] * 8 for _ in range(8)]
        self._current = 0  # 0 = Black (first), 1 = White
        self._last_move: Optional[tuple[int, int]] = None
        self.reset()

    def reset(self) -> None:
        self.grid = [[EMPTY] * 8 for _ in range(8)]
        # Standard Othello starting position: 4 pieces in the center
        self.grid[3][3] = WHITE
        self.grid[3][4] = BLACK
        self.grid[4][3] = BLACK
        self.grid[4][4] = WHITE
        self._current = 0  # Black moves first
        self._last_move = None

    def _piece(self, player: int) -> int:
        return BLACK if player == 0 else WHITE

    def _opponent(self, player: int) -> int:
        return 1 - player

    def _flips_at(self, row: int, col: int, player: int) -> list[tuple[int, int]]:
        """Return list of (r,c) cells that would be flipped if `player` plays at (row,col)."""
        if self.grid[row][col] != EMPTY:
            return []
        my_piece = self._piece(player)
        opp_piece = self._piece(self._opponent(player))
        flips = []
        for dr, dc in DIRS:
            r, c = row + dr, col + dc
            line = []
            while 0 <= r < 8 and 0 <= c < 8 and self.grid[r][c] == opp_piece:
                line.append((r, c))
                r += dr
                c += dc
            if line and 0 <= r < 8 and 0 <= c < 8 and self.grid[r][c] == my_piece:
                flips.extend(line)
        return flips

    def legal_moves(self) -> list[str]:
        moves = []
        for r in range(8):
            for c in range(8):
                if self.grid[r][c] == EMPTY and self._flips_at(r, c, self._current):
                    moves.append(f"{chr(ord('A') + c)}{r + 1}")
        return sorted(moves)

    def apply(self, move_str: str) -> bool:
        # Parse e.g. "D3" or "d3"
        move_str = move_str.strip().upper()
        if len(move_str) != 2:
            return False
        col_ch, row_ch = move_str[0], move_str[1]
        if col_ch < 'A' or col_ch > 'H':
            return False
        if row_ch < '1' or row_ch > '8':
            return False
        col = ord(col_ch) - ord('A')
        row = int(row_ch) - 1
        flips = self._flips_at(row, col, self._current)
        if not flips:
            return False
        self.grid[row][col] = self._piece(self._current)
        for r, c in flips:
            self.grid[r][c] = self._piece(self._current)
        self._last_move = (row, col)
        # Switch turn
        self._current = self._opponent(self._current)
        # If the next player has no moves, skip their turn
        if not self.legal_moves():
            self._current = self._opponent(self._current)
            # If the original player also has no moves, game is over
            if not self.legal_moves():
                pass  # is_game_over will return True
        return True

    def is_game_over(self) -> bool:
        # Game ends when neither player can move
        if self.legal_moves():
            return False
        # Current player has no moves; check if the other player can move
        self._current = self._opponent(self._current)
        has_moves = bool(self.legal_moves())
        self._current = self._opponent(self._current)  # restore
        return not has_moves

    def winner(self) -> Optional[int]:
        black = sum(1 for r in range(8) for c in range(8) if self.grid[r][c] == BLACK)
        white = sum(1 for r in range(8) for c in range(8) if self.grid[r][c] == WHITE)
        if black > white:
            return 0
        if white > black:
            return 1
        return None  # draw

    def current_player(self) -> int:
        return self._current

    def board_ascii(self, pov: int = 0) -> str:
        lines = ["   A B C D E F G H"]
        for r in range(8):
            row_str = f"{r + 1}  "
            for c in range(8):
                v = self.grid[r][c]
                row_str += ("B" if v == BLACK else "W" if v == WHITE else "·") + " "
            lines.append(row_str.rstrip())
        return "\n".join(lines)

    def board_state(self) -> dict:
        # Serialize as a flat string for the WS message
        # "B" = black, "W" = white, "." = empty, row by row from top
        cells = ""
        for r in range(8):
            for c in range(8):
                v = self.grid[r][c]
                cells += "B" if v == BLACK else "W" if v == WHITE else "."
        last = ""
        if self._last_move:
            lr, lc = self._last_move
            last = f"{chr(ord('A') + lc)}{lr + 1}"
        return {"othello_grid": cells, "last_move": last}

    def last_move_display(self) -> str:
        if self._last_move:
            lr, lc = self._last_move
            return f"{chr(ord('A') + lc)}{lr + 1}"
        return ""

    def parse_move(self, content: str) -> str:
        move, _say = _extract_json_move(content)
        if move:
            return move
        # Try to find a cell reference like "D3" or "d3" in the text
        m = re.search(r'\b([A-Ha-h][1-8])\b', content)
        if m:
            return m.group(1).upper()
        return ""

    def system_prompt(self, commentary: bool, player_color: str) -> str:
        color_name = "Black" if player_color == "black" else "White"
        if commentary:
            return (
                f"You are playing a game of Othello (Reversi) as {color_name}.\n"
                "Rules: Place your disc on an empty cell so that you 'flank' one or more "
                "of your opponent's discs (in a straight line) and flip them to your color. "
                "The game ends when neither player can move. Most discs wins.\n"
                "Columns are A-H (left to right), rows are 1-8 (top to bottom).\n"
                'Respond with JSON: {"move": "<cell like D3>", "say": "<one short sentence of trash talk, max 12 words>"}'
            )
        return (
            f"You are playing a game of Othello (Reversi) as {color_name}.\n"
            "Rules: Place your disc on an empty cell so that you 'flank' one or more "
            "of your opponent's discs (in a straight line) and flip them to your color. "
            "The game ends when neither player can move. Most discs wins.\n"
            "Columns are A-H (left to right), rows are 1-8 (top to bottom).\n"
            'Respond with JSON: {"move": "<cell like D3>"}'
        )

    def prompt_template(self) -> str:
        return (
            "You are {player}, playing a game of Othello as {color} against {opponent}.\n"
            "This is game {gameNumber} of {totalGames}.\n"
            "Earlier games: {previousGames}\n\n"
            "Current board (B=Black, W=White, ·=empty):\n{board}\n\n"
            "Move number: {moveNumber}\n"
            "Last move: {lastMove}\n\n"
            "Your discs: {myCount}\n"
            "Opponent discs: {oppCount}\n\n"
            "LEGAL MOVES ({legalMoveCount}): {legalMoves}\n\n"
            "Choose your move."
        )

    def prompt_values(self, player_label, opponent_label, player_color,
                      game_no, total_games, move_no, last_move, move_list, prev_games) -> dict:
        legal = self.legal_moves()
        my_piece = self._piece(self._current)
        opp_piece = self._piece(self._opponent(self._current))
        my_count = sum(1 for r in range(8) for c in range(8) if self.grid[r][c] == my_piece)
        opp_count = sum(1 for r in range(8) for c in range(8) if self.grid[r][c] == opp_piece)
        return {
            "player": player_label,
            "color": player_color,
            "opponent": opponent_label,
            "gameNumber": game_no,
            "totalGames": total_games,
            "board": self.board_ascii(),
            "moveNumber": move_no,
            "lastMove": last_move or "—",
            "previousGames": prev_games or "—",
            "legalMoveCount": len(legal),
            "legalMoves": " ".join(legal),
            "myCount": my_count,
            "oppCount": opp_count,
        }

    def score(self, p0_w, p0_d, p0_l, p1_w, p1_d, p1_l) -> str:
        # Othello uses simple win counting
        return f"{p0_w}–{p1_w}"

    def adjudicate(self) -> tuple[str, str]:
        black = sum(1 for r in range(8) for c in range(8) if self.grid[r][c] == BLACK)
        white = sum(1 for r in range(8) for c in range(8) if self.grid[r][c] == WHITE)
        if black > white:
            return ("1-0", f"disc count {black}-{white}")
        if white > black:
            return ("0-1", f"disc count {white}-{black}")
        return ("1/2-1/2", f"disc count {black}-{white}")

    def adjudicate_ply_limit(self) -> tuple[str, str]:
        return self.adjudicate()  # Same logic — most discs wins

    @staticmethod
    def grid_size() -> int:
        return 8


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GAMES: dict[str, type[GameEngine]] = {
    "chess": ChessEngine,
    "othello": OthelloEngine,
}

def create_engine(game_type: str) -> GameEngine:
    cls = GAMES.get(game_type, ChessEngine)
    return cls()