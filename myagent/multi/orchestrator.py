"""编排者-工人（orchestrator-worker）多 agent 装配。

设计要点：编排者就是一个普通 ``AgentLoop``，通过注入的
``delegate_agent`` 工具获得委派能力；工人是后台线程里临时创建的
另一个 ``AgentLoop``（共享同一 LLM client 与工作目录/工具执行器），
跑完一次子任务的完整 ReAct 后，把 ``AgentResponse`` 文本化为观察结果
返回给编排者继续推理。

上下文边界（与用户约定一致：**只共享工作区 + 任务描述**）：

- 工人使用独立的角色 system prompt（见 context.render_worker_prompt），
  任务描述以 ``{task}`` 注入，不携带编排者的会话历史与跨会话记忆；
- 工人的 ReAct 轨迹只在本地累积，不进入编排者的消息历史——
  编排者只见一条 ``delegate_agent`` 工具消息及其文本观察；
- 工人不带 memory / composer / approval（后台线程不能弹交互审批），
  写操作仍受底层 shell 黑名单与工具错误体系兜底。

委派并发：工人经 ``ThreadPoolExecutor`` 限流运行，同一批内多次
``delegate_agent`` 按模型下发顺序**顺序执行**（工人共享工作区，
并发写同一文件可能竞态，顺序是安全默认；池为异常隔离与后续并行铺路）。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..agent_config import AgentParams
from ..agent_loop import AgentLoop
from ..contracts import (
    AgentRequest,
    AgentResponse,
    LLMClient,
    StopReason,
    ToolExecutor,
)
from .context import render_worker_prompt
from .worker_tool import _clear_host, _set_host, register_worker_tool, worker_tools_schema

#: 工人推理轮数默认上限。
WORKER_MAX_TURNS = 12

#: 观察文本（DONE / FAILED）的最大字符数，超出截断。
OBSERVATION_MAX_CHARS = 4000

#: 工人轨迹单步最终答案的预览长度。
ANSWER_PREVIEW_CHARS = 120


@dataclass(slots=True)
class WorkerResult:
    """一次委派（工人运行）的结构化结果。

    编排者侧只消费 ``to_observation()`` 的文本；其余字段供日志/测试/审计。
    """

    #: 工人最终答案；非 FINAL_ANSWER 时为 None。
    final_answer: str | None

    #: 工人结束原因。
    stop_reason: StopReason

    #: 工人消耗的推理轮数。
    turns_used: int

    #: 工人分发给工具执行器的调用次数。
    tool_calls: int

    #: 工人错误文本（ERROR / TOOL_ERROR 时）。
    error: str | None = None

    #: 工人完整轨迹（节选进观察文本）。
    steps: list = field(default_factory=list)

    @classmethod
    def from_response(cls, resp: AgentResponse) -> "WorkerResult":
        return cls(
            final_answer=resp.final_answer,
            stop_reason=resp.stop_reason,
            turns_used=resp.turns_used,
            tool_calls=resp.tool_calls,
            steps=resp.steps,
            error=resp.error,
        )

    def to_observation(self, max_chars: int = OBSERVATION_MAX_CHARS) -> str:
        """渲染为编排者看到的观察文本。

        - 成功：``DONE: <答案>`` + 一行统计；
        - 失败：``FAILED: <原因>`` + 统计 + 工人轨迹节选。
        """
        stats = (
            f"[worker: turns={self.turns_used}, tools={self.tool_calls}, "
            f"stop={self.stop_reason.value}]"
        )
        if self.stop_reason is StopReason.FINAL_ANSWER and self.final_answer:
            answer = self.final_answer
            if len(answer) > max_chars:
                answer = answer[:max_chars] + "\n…（工人答案已截断）"
            return f"DONE: {answer}\n{stats}"
        reason = self.error or self.stop_reason.value
        trace = _summarize_steps(self.steps)
        body = f"FAILED: {reason}\n{stats}"
        if trace:
            body += "\n工人执行轨迹（节选）：\n" + trace
        if len(body) > max_chars:
            body = body[:max_chars] + "\n…（轨迹已截断）"
        return body


def _summarize_steps(steps: list) -> str:
    """把工人轨迹压缩为紧凑文本（每轮一行），供 FAILED 观察使用。"""
    lines: list[str] = []
    for step in steps:
        if getattr(step, "tool_calls", None):
            names = [
                (call or {}).get("function", {}).get("name", "?")
                for call in step.tool_calls
            ]
            lines.append(f"  turn{step.turn}: tools={','.join(names)}")
        elif getattr(step, "final_answer", None):
            preview = step.final_answer.replace("\n", " ")
            if len(preview) > ANSWER_PREVIEW_CHARS:
                preview = preview[:ANSWER_PREVIEW_CHARS] + "…"
            lines.append(f"  turn{step.turn}: final={preview}")
    return "\n".join(lines)


#: 工人 loop 工厂签名：任务 + 角色 + 轮数上限 → AgentLoop。
WorkerFactory = Callable[[str, str | None, int | None], AgentLoop]


class OrchestratorLoop:
    """包装一个 AgentLoop，注入 delegate_agent 委派能力。

    对外接口与 AgentLoop 一致（``run(AgentRequest) -> AgentResponse``），
    因此 cli 的 REPL 层无需感知多 agent。
    """

    def __init__(
        self,
        inner_loop: AgentLoop,
        *,
        worker_factory: WorkerFactory,
        max_workers: int = 4,
    ):
        self.inner = inner_loop
        self._worker_factory = worker_factory
        #: 工人执行池：限流并发 + 异常隔离（future.result 捕获线程内异常）。
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="worker"
        )

    def run(self, request: AgentRequest) -> AgentResponse:
        """代理到内层主循环；运行期间把自身挂到线程局部供 delegate_agent 取用。"""
        _set_host(self)
        try:
            return self.inner.run(request)
        finally:
            _clear_host()

    def delegate_worker(
        self,
        task: str,
        role: str | None = None,
        worker_turns: int | None = None,
    ) -> WorkerResult:
        """在后台线程运行一个工人 agent 完成子任务，阻塞等待结果。

        任何线程内异常都会转换为 ``WorkerResult(stop=ERROR)``，
        保证委派失败只表现为 FAILED 观察，不打断编排者。
        """
        future = self._pool.submit(self._run_worker, task, role, worker_turns)
        try:
            return future.result()
        except Exception as exc:
            return WorkerResult(
                final_answer=None,
                stop_reason=StopReason.ERROR,
                turns_used=0,
                tool_calls=0,
                error=f"工人运行异常：{exc}",
            )

    def _run_worker(
        self, task: str, role: str | None, worker_turns: int | None
    ) -> WorkerResult:
        loop = self._worker_factory(task, role, worker_turns)
        resp = loop.run(AgentRequest(user_input=task))
        return WorkerResult.from_response(resp)


def build_orchestrator_loop(
    *,
    agent_params: AgentParams,
    llm: LLMClient,
    tools: ToolExecutor,
    memory=None,
    composer=None,
    approval_gate=None,
    worker_prompt_dir: Path | str | None = None,
    max_workers: int = 4,
    worker_max_turns: int = WORKER_MAX_TURNS,
) -> OrchestratorLoop:
    """装配编排者主循环（cli --multi_agent 入口）。

    - 内层 loop：完整复用主 agent 装配（memory/composer/approval 照常注入）；
    - 工人 loop：独立 AgentParams（同 cwd、同 token 预算、轮数上限可调），
      环境模板从主 ``prompt_dir`` 解析，system prompt 由
      ``render_worker_prompt`` 按角色渲染覆盖；
    - 工人共享同一个 ToolExecutor（同一工作目录上下文）。
    """
    inner = AgentLoop(
        agent_params,
        llm=llm,
        tools=tools,
        memory=memory,
        composer=composer,
        approval_gate=approval_gate,
    )
    prompt_dir = Path(worker_prompt_dir) if worker_prompt_dir else Path(agent_params.prompt_dir)

    def worker_factory(task: str, role: str | None, worker_turns: int | None) -> AgentLoop:
        params = AgentParams(
            cwd=agent_params.cwd,
            prompt_dir=agent_params.prompt_dir,
            max_turns=worker_turns or worker_max_turns,
            max_input_tokens=agent_params.max_input_tokens,
            max_output_tokens=agent_params.max_output_tokens,
            tool_timeout=agent_params.tool_timeout,
        )
        loop = AgentLoop(
            params,
            llm=llm,
            tools=tools,
            # 工人不可见 delegate_agent：worker_tools_schema 已过滤。
            tools_schema=worker_tools_schema,
        )
        loop.system_prompt = render_worker_prompt(prompt_dir, task, role=role)
        return loop

    orchestrator = OrchestratorLoop(
        inner, worker_factory=worker_factory, max_workers=max_workers
    )
    # 注册 delegate_agent 工具（幂等；此后 to_openai_tools 每次渲染都包含它）。
    register_worker_tool()
    return orchestrator
