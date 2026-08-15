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
        "--cwd",
        default=".",
        metavar="DIR",
        help="设置工作目录（默认当前目录）",
    )
    parser.add_argument(
        "--memory_dir",
        default=None,
        metavar="DIR",
        help="记忆存档目录（默认 memories/，相对项目根）",
    )
    parser.add_argument(
        "--no_memory",
        action="store_true",
        help="不启用记忆：不存档会话原文、不注入跨会话记忆",
    )
    parser.add_argument(
        "--no_compose",
        action="store_true",
        help="不启用上下文压缩：不裁剪 tool 输出、不丢弃历史",
    )
    parser.add_argument(
        "--multi_agent",
        action="store_true",
        help="启用编排者-工人多 agent 模式（注入 delegate_agent 委派工具）",
    )
    parser.add_argument(
        "--worker_prompt_dir",
        default=None,
        metavar="DIR",
        help="工人提示词目录（默认与主 agent 相同 prompts/；角色模板 worker_<role>.md）",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        metavar="N",
        help="工人并发上限（默认 4）",
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
