"""工人提示词渲染：角色模板加载 + 任务注入 + 输出契约兜底。

编排者委派任务时，工人 agent 使用独立的 system prompt（角色模板），
而不是复用主 agent 的工具提示词。模板解析规则：

1. ``<worker_prompt_dir>/worker_<role>.md``（按角色定制，如 researcher）；
2. 缺失回退 ``<worker_prompt_dir>/worker.md``（通用工人模板）；
3. 再缺失回退默认工具系统提示词（``DEFAULT_SYSTEM_PROMPT``）。

模板内 ``{task}`` 占位符替换为子任务描述；若模板未自带
``## 输出契约`` 段，则末尾追加四条约（保证工人输出结构一致，
编排者侧只按 ``final_answer`` / 轨迹文本解析，不依赖前缀）。
"""
from __future__ import annotations

from pathlib import Path

from ..api_config import DEFAULT_SYSTEM_PROMPT

#: 通用工人模板文件名（角色模板缺失时的回退）。
DEFAULT_WORKER_TEMPLATE = "worker.md"

#: 角色模板文件名模式：worker_<role>.md。
ROLE_TEMPLATE_PATTERN = "worker_{role}.md"

#: 模板中任务占位符。
TASK_PLACEHOLDER = "{task}"

#: 输出契约段标题；模板自带该标题时不再追加。
CONTRACT_MARKER = "## 输出契约"

#: 追加的输出契约四条（模板未自带时兜底）。
CONTRACT_RULES = (
    "1. 必须先调用工具（至少一次）收集证据，再给出最终结论；\n"
    "2. 同一轮提交多个独立的只读工具调用，减少模型往返；工具按顺序执行；\n"
    "3. 完成后输出最终答案，格式固定：`DONE: <结论>`，结论要完整、直接、有依据；\n"
    "4. 你只向编排者汇报结果，禁止反问、寒暄、解释过程或输出与任务无关的内容。\n"
)


def worker_template_path(prompt_dir: Path | str, role: str | None) -> Path:
    """返回角色模板路径（可能不存在，由调用方决定回退）。"""
    if role:
        return Path(prompt_dir) / ROLE_TEMPLATE_PATTERN.format(role=role)
    return Path(prompt_dir) / DEFAULT_WORKER_TEMPLATE


def render_worker_prompt(
    prompt_dir: Path | str,
    task: str,
    role: str | None = None,
) -> str:
    """渲染工人 system prompt：角色模板 + 任务 + 输出契约兜底。"""
    prompt_dir = Path(prompt_dir)
    path = worker_template_path(prompt_dir, role)
    try:
        template = path.read_text(encoding="utf-8")
    except OSError:
        # 角色模板缺失 → 回退通用模板；通用也缺失 → 默认提示词 + 任务段。
        if role is not None:
            try:
                template = (
                    prompt_dir / DEFAULT_WORKER_TEMPLATE
                ).read_text(encoding="utf-8")
            except OSError:
                return DEFAULT_SYSTEM_PROMPT + _task_block(task) + _contract_block()
        else:
            return DEFAULT_SYSTEM_PROMPT + _task_block(task) + _contract_block()

    rendered = template.replace(TASK_PLACEHOLDER, task)
    if CONTRACT_MARKER not in template:
        rendered += _contract_block()
    return rendered


def _task_block(task: str) -> str:
    """回退默认提示词时的任务段（无角色模板可用时的最小可用形态）。"""
    return f"\n\n## 你的任务\n\n{task}"


def _contract_block() -> str:
    """输出契约段（模板未自带时兜底追加）。"""
    return f"\n\n{CONTRACT_MARKER}\n\n{CONTRACT_RULES}"
