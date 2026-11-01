import argparse
import os

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="vjms · local tiny-language-model studio")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", default=os.environ.get("VJMS_HOME", ".vjms"))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1–65535")
    from .app import create_app
    print(f"\nvjms · http://127.0.0.1:{args.port}\nData stays on this computer. Ctrl+C to stop.\n")
    uvicorn.run(create_app(args.data_dir), host="127.0.0.1", port=args.port, workers=1)
