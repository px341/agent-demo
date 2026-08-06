"""命令行参数解析：集中管理 myagent 的启动参数。

后续要加新参数，只需在 build_parser() 里追加 add_argument(...) 即可，
cli.py 通过 parse_args() 拿到解析结果。
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myagent",
        description="AI Agent Terminal",
    )
    parser.add_argument(
        "--one_shot",
        action="store_true",
        help="只调用一次 LLM，提示词由用户输入",
    )
    # 后续要加的其他参数在这里 add_argument(...)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数；缺省使用 sys.argv[1:]。"""

    return build_parser().parse_args(argv)


def main() -> int:
    args = parse_args()
    print(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
