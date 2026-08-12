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
python -m myagent --cwd DIR  # 指定工作目录
python -m myagent --no_memory # 不启用记忆（不存档、不注入跨会话记忆）
python -m myagent --no_compose # 不启用上下文压缩（不裁剪 tool 输出、不丢弃历史）
```

REPL 内建命令：`/exit`、`/quit`。

## 模块清单

`myagent/` 共 10 个 Python 文件（约 1350 行），分层如下：

| 模块 | 职责 |
|------|------|
| `__main__.py` | 入口，仅调用 `cli.main()` |
| `cli.py` | 启动装配；REPL 多轮模式；结果打印 |
| `argparse.py` | 命令行参数解析（集中管理，新参数只加 `add_argument`） |
| `agent_config.py` | `AgentParams` 参数与 `Message` 统一消息格式 |
| `api_config.py` | LLM 生成参数与默认 system prompt |
| `provider.py` | DeepSeek 客户端；透传 `reasoning_content` metadata，防多轮 400 |
| `contracts.py` | 主循环输入/输出契约 + 依赖注入 Protocol |
| `agent_loop.py` | ReAct 主循环；轨迹记录；`step_to_messages` 消息重建；注入记忆段 |
| `actions.py` | 模型输出解析为 `ToolCall` / `FinalAnswer` / `Retry` |
| `tools/` | 工具注册表与执行器（`read_file` / `list_files` 等 10 个，装饰器注册扩展） |
| `memory/` | 记忆系统：会话 jsonl 存档、脱敏、水位线摘要、跨会话注入（`MemoryManager`） |
| `context/` | 上下文压缩：三部分预算管理（system+memory / 本轮 session / 用户输入） |
| `prompts/` | 系统提示词（工具 / 环境 / 摘要提取 / 摘要聚合） |

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

- **契约与依赖注入分离**：`contracts.py` 定义 `LLMClient`（必需）、`ToolExecutor` / `MemoryStore` / `ContextComposer`（可选）Protocol，主循环不感知具体实现；工具、记忆、上下文压缩各自接入。
- **上下文压缩（context/）**：每次请求的输入分三部分预算——用户输入不可压缩；tool 输出单条超限裁剪为开头+结尾各半、总量超限丢弃最早结果（保留最新一对）；整体仍超则丢弃最早非 tool 消息。tool 消息与其 assistant 声明成对处理，绝不留下孤儿消息，预算充足时零改动。
- **轮数与工具计数分离**：`attempts`（模型调用次数）与 `tool_calls`（实际工具调用次数）分别统计，格式错误的输出只消耗重试轮。
- **DeepSeek thinking mode 兼容**：`reasoning_content` 提取进 metadata，随 assistant 消息原样回传，否则真实端点返回 400。
- **多轮历史由调用方持有**：`AgentLoop` 每次只处理一个请求，不修改调用方传入的历史；REPL 层用 `step_to_messages` 从轨迹重建历史。

## 现状与待办

- 工具（10 个：文件/目录读写改删）已接入 CLI，read 免询问、write/delete 走审批闸门；
- 记忆系统已接入：启动时自动汇总历史会话（`sweep()`，mtime 水位线幂等 + 失败重试），REPL 每轮存档会话原文（脱敏），退出标记会话；`memories/summary.md` 作为「跨会话记忆」注入 system prompt（默认 4000 字符截断）。`--no_memory` 可禁用；
- 上下文压缩已接入：默认启用，按预算压缩输入（tool 单条裁剪 + 总量丢弃 + 非 tool 丢弃）；`--no_compose` 可禁用。`max_input_tokens` 由此生效；
- REPL 历史有上限（默认 300 条消息），超出后配对安全裁剪，保证不产生孤儿消息。