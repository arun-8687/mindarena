"""Launcher — loads API keys from a .env file, then starts the server.

Looks for a .env in, in order of preference:
  $MINDARENA_ENV        (explicit path)
  ./.env                (next to this file)
  ~/.config/mindarena/.env
  ~/.hermes/.env        (legacy location)

Values already present in the real environment always win.
"""
from __future__ import annotations

import os
import pathlib
import runpy
import sys

HERE = pathlib.Path(__file__).resolve().parent
HOME = pathlib.Path.home()

CANDIDATES = [
    pathlib.Path(os.environ["MINDARENA_ENV"]) if os.environ.get("MINDARENA_ENV") else None,
    HERE / ".env",
    HOME / ".config" / "mindarena" / ".env",
    HOME / ".hermes" / ".env",
]

# Any of these being set is enough to run; we only warn when none are.
KEY_VARS = ("OLLAMA_API_KEY", "OPENCODE_GO_API_KEY", "GT_API_KEY")


def load_env(path: pathlib.Path) -> int:
    """Parse a minimal .env file. Returns the number of variables set."""
    count = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k and k not in os.environ:
            os.environ[k] = v
            count += 1
    return count


def main() -> None:
    for candidate in CANDIDATES:
        if candidate and candidate.is_file():
            n = load_env(candidate)
            print(f"loaded {n} variable(s) from {candidate}", file=sys.stderr)
            break

    if not any(os.environ.get(k) for k in KEY_VARS):
        print(f"WARNING: none of {', '.join(KEY_VARS)} are set — "
              "the arena will not be able to reach a model endpoint.", file=sys.stderr)

    sys.argv = [sys.argv[0]]
    runpy.run_path(str(HERE / "server.py"), run_name="__main__")


if __name__ == "__main__":
    main()
