# myagent

一个基于 ReAct（思考→行动→观察）循环的小型 AI coding agent 终端。

```text
CLI 启动
  ↓
AgentParams + LLM Client（.env.llm → DeepSeek）
  ↓
AgentLoop（ReAct 主循环）
  ↓
parse_action → ToolCall / FinalAnswer / Retry
  ↓
ToolExecutor（tools.py 注册表）
  ↓
观察结果回灌 → 继续推理 / 最终答案
```

## 快速开始

```bash
# 先准备 .env.llm（参考 .env.example）
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

python -m myagent            # 交互式多轮 REPL
python -m myagent --one_shot # 一次性 ReAct 任务
python -m myagent --cwd DIR  # 指定工作目录
```

REPL 内建命令：`/exit`、`/quit`。

## 模块清单

`myagent/` 共 10 个 Python 文件（约 1350 行），分层如下：

| 模块 | 职责 |
|------|------|
| `__main__.py` | 入口，仅调用 `cli.main()` |
| `cli.py` | 启动装配；one_shot / REPL 两种模式；结果打印 |
| `argparse.py` | 命令行参数解析（集中管理，新参数只加 `add_argument`） |
| `agent_config.py` | `AgentParams` 参数与 `Message` 统一消息格式 |
| `api_config.py` | LLM 生成参数与默认 system prompt |
| `provider.py` | DeepSeek 客户端；透传 `reasoning_content` metadata，防多轮 400 |
| `contracts.py` | 主循环输入/输出契约 + 依赖注入 Protocol |
| `agent_loop.py` | ReAct 主循环；轨迹记录；`step_to_messages` 消息重建 |
| `actions.py` | 模型输出解析为 `ToolCall` / `FinalAnswer` / `Retry` |
| `tools.py` | 工具注册表与执行器（`read_file` / `list_files`，装饰器注册扩展） |
| `context_manager.py` | token 裁剪 + `.storage/` 跨会话历史持久化（解耦接缝） |
| `prompts/` | 系统提示词（`tools_system_prompt.md`、`planner_system_prompt.md`、`one_shot_system_prompt.md`） |

## 主循环流程

每次 `loop.run()`：

1. 拼接 `system + 历史 + 当前请求` 消息列表；
2. `llm.complete(messages)` 得到原始输出；
3. `parse_action` 解析为 Action：
   - `final` → 返回 `FINAL_ANSWER`；
   - `tool_call` → 通过注入的 `ToolExecutor` 执行（未注入则回灌错误观察），结果经 `step_to_messages` 重建为 `assistant + tool` 消息回灌；
   - `retry` / 非法 JSON → 把原因作为 `user` 反馈回灌，模型重试（不消耗工具计数）；
4. 超过 `max_turns` → 以 `MAX_TURNS` 收口；
5. LLM 异常 → 以 `ERROR` 收口。

## 关键设计

- **契约与依赖注入分离**：`contracts.py` 定义 `LLMClient`（必需）、`ToolExecutor` / `MemoryStore`（可选）三个 Protocol，主循环不感知具体实现；工具、记忆后续各自接入即可。
- **轮数与工具计数分离**：`attempts`（模型调用次数）与 `tool_calls`（实际工具调用次数）分别统计，格式错误的输出只消耗重试轮。
- **DeepSeek thinking mode 兼容**：`reasoning_content` 提取进 metadata，随 assistant 消息原样回传，否则真实端点返回 400。
- **多轮历史由调用方持有**：`AgentLoop` 每次只处理一个请求，不修改调用方传入的历史；REPL 层用 `step_to_messages` 从轨迹重建历史。

## 现状与待办

- 工具仅注册了 `read_file`、`list_files`，尚未接入 CLI（`cli.py` 建 `AgentLoop` 时未注入 tools）；
- `context_manager.py`（token 裁剪 + 历史持久化）与 `MemoryStore` 记忆为解耦接缝，未接入主链；
- 详细流程图见 `architecture.md`。