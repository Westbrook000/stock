#!/usr/bin/env python3
"""启动本地分析服务: python3 scripts/serve.py [--port 8300]"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("NO_PROXY", "*")

import uvicorn  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    print(f"威科夫分析服务: http://{args.host}:{args.port}")
    uvicorn.run("wyckoff.server.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
