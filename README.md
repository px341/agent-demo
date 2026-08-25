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
python -m myagent --multi_agent # 编排者-工人多 agent 模式（可配 --worker_prompt_dir / --max_workers）
```

REPL 内建命令：`/exit`、`/quit`。

## 模块清单

`myagent/` 共 12 个顶层 Python 文件 + tools/context/memory/multi 四个子包（约 3800 行），分层如下：

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
| `memory/` | 记忆系统：会话 jsonl 存档、脱敏、水位线摘要、关键词索引按相关性检索注入（`MemoryManager`） |
| `context/` | 上下文压缩：三部分预算管理（system+memory / 本轮 session / 用户输入） |
| `multi/` | 多 agent：编排者-工人模式（`OrchestratorLoop` / `delegate_agent` 工具 / 工人提示词渲染） |
| `prompts/` | 系统提示词（工具 / 环境 / 摘要提取 / 摘要聚合 / 工人 worker.md / 角色 worker_<role>.md） |

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
- 记忆系统已接入：启动时自动汇总历史会话（`sweep()`，mtime 水位线幂等 + 失败重试），REPL 每轮存档会话原文（脱敏），退出标记会话。单会话摘要带 `keywords` 关键词索引（旧摘要词法兜底补索引，零额外 LLM 成本）；注入时按当前请求相关性检索（`context_block(query)`，纯词法匹配），命中则优先注入相关会话摘要、全局 `summary.md` 兜底，无命中回落全量聚合（默认 4000 字符预算）。`--no_memory` 可禁用；
- 上下文压缩已接入：默认启用，按预算压缩输入（tool 单条裁剪 + 总量丢弃 + 非 tool 丢弃）；`--no_compose` 可禁用。`max_input_tokens` 由此生效；
- REPL 历史有上限（默认 300 条消息），超出后配对安全裁剪，保证不产生孤儿消息。

## 多 agent（编排者-工人，`--multi_agent`）

核心思路：**编排者就是一个普通 `AgentLoop`**，只是额外注册了 `delegate_agent`
工具；工人是后台线程里临时创建的另一个 `AgentLoop`（共享同一 LLM client 与
工作目录/工具执行器），跑完一次子任务的完整 ReAct 后，把结果文本化为观察
返回给编排者继续推理。主循环本身零改动。

```text
编排者 AgentLoop（完整装配：记忆/压缩/审批照常）
  └─ delegate_agent 工具 → 后台线程 → 工人 AgentLoop（独立角色 prompt）
        └─ 工人自己跑 ReAct（read_file / run_shell …）
  ← DONE: <答案>  /  FAILED: <原因> 文本观察
  └─ 编排者基于观察继续推理
```

- **上下文边界（只共享工作区 + 任务描述）**：工人用独立角色 prompt
  （`prompts/worker.md`，或按 `role` 加载 `worker_<role>.md`，任务经 `{task}`
  注入），不带编排者的会话历史与跨会话记忆；工人的轨迹不进入编排者历史；
- **写者归一（解决并发写冲突）**：工人**只有只读工具**——schema 层只渲染
  `risk="read"` 工具（`worker_tools_schema`），executor 层用独立的白名单
  `ToolExecutor`（`tool_names=read_only_tool_names()`）双保险，模型即使
  幻想调用写工具也会被拒绝（`ValidationError` 回灌，可恢复）。需要改动时
  工人把建议整理为 unified diff 附在 `DONE:` 结论里，由编排者统一
  `git_apply_patch` 裁决落盘——多个工人永远不写同一文件，冲突交给 git。
  **边界说明**：工人侧"永不写"是硬保证；编排者侧**不承诺"收尾才写"**——
  它是单线程串行 + 审批闸门 + git 兜底，收到 patch 可随时 apply，也可
  边委派边改（有意的灵活性，无并发写冲突）。
- **工人失败隔离**：工人超轮数 / 出错只表现为 `FAILED:` 观察（含原因与轨迹
  节选），不打断编排者；
- **防递归委派**：工人 schema 不含 `delegate_agent`，嵌套委派被禁止；
- **免询问**：`delegate_agent` 标 `risk="read"`（委派本身不改工作区），
  工人不带审批闸门（后台线程不能弹交互确认），写操作由只读白名单杜绝；
- **并发**：委派经 `ThreadPoolExecutor(max_workers)` 限流；工人纯只读、
  无共享可变状态，同一批委派先顺序执行（为后续并行铺路，池只约束
  同时存活的工人数）。

参数：`--multi_agent`（启用）、`--worker_prompt_dir`（工人提示词目录，默认与
主 agent 同 `prompts/`）、`--max_workers`（并发上限，默认 4）。