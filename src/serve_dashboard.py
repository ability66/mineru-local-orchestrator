from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve generated compare dashboards over HTTP."
    )
    parser.add_argument("--root-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    handler = partial(SimpleHTTPRequestHandler, directory=str(root_dir))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {root_dir} at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
