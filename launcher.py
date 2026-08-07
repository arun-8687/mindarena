"""Launcher — loads Ollama Cloud key from ~/.hermes/.env, starts the server."""
import os, sys, pathlib, runpy

HOME = pathlib.Path.home()
ENV = HOME / ".hermes" / ".env"
if ENV.exists():
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# sanity
key = os.environ.get("OLLAMA" + "_API_" + "KEY")
if not key:
    print("WARNING: OLLAMA_API_KEY not found in env or .env", file=sys.stderr)

# import and run server
sys.argv = [sys.argv[0]]
runpy.run_path(str(pathlib.Path(__file__).parent / "server.py"), run_name="__main__")