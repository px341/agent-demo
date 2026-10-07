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

python -m pip install -e .
python -m myagent            # 交互式多轮 REPL
python -m myagent --cwd DIR  # 指定工作目录
python -m myagent --no_memory # 不启用记忆（不存档、不注入跨会话记忆）
python -m myagent --no_compose # 不启用上下文压缩（不裁剪 tool 输出、不丢弃历史）
python -m myagent --multi_agent # 编排者-工人多 agent 模式（可配 --worker_prompt_dir / --max_workers）
python -m myagent --one-shot "修复这个项目的失败测试" --auto-approve --no_memory
```

REPL 内建命令：`/exit`、`/quit`。

`--one-shot` 执行单个任务后退出，不读取标准输入。`--auto-approve` 会自动批准
工作目录内的读写/删除工具，只应配合临时目录、容器或其他隔离工作区使用。

## SWE-bench Verified Mini（默认评测环境）

默认使用社区的
[SWE-bench Verified Mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini)
test split，共 50 个固定实例，Mini 数据版本锁定为
`b316c349947c29963fce3f4a65967c9807a4b673`。默认只检查环境，后续显式运行时
默认选择原始顺序的前 3 个实例，不会自动跑完 50 题。

### 1. 安装和准备数据

在 Ubuntu-24.04 的项目目录中执行：

```bash
python3 -m venv agentvenv
source agentvenv/bin/activate
python -m pip install -e ".[benchmark]"
bash scripts/swebench-smoke.sh prepare
```

`prepare` 下载 Mini 数据及官方 Verified 数据，按 Mini 的 50 个实例 ID 补齐
当前 `swebench` 评分器需要的 `image`、`eval_script`、`log_parser`、`eval_type`
字段，并核对仓库、基础提交和题目文本。生成导出仅保留 `instance_id`、`repo`、
`base_commit`、`problem_statement`，放在 `.swebench-work/data/`；含答案与隐藏测试
信息的评分数据单独放在 `.swebench-work/scoring/`。准备数据不会调用模型、克隆题目仓库、
拉取 Docker 镜像或启动评分容器。

Docker Desktop 需要在 Settings → Resources → WSL Integration 中启用
Ubuntu-24.04，详见 [Docker 官方说明](https://docs.docker.com/desktop/features/wsl/)。
参考 `.env.example` 创建本地 `.env.llm`，填写自己的模型配置；预算模式目前支持
官方 `api.deepseek.com` 的 `deepseek-flash` / `deepseek-v4-flash`。
`MYAGENT_PYTHON` 可指定其他 Python 环境，默认是 `agentvenv/bin/python`。

### 2. 只检查，不运行评测

```bash
bash scripts/swebench-smoke.sh
# 或明确指定 check
bash scripts/swebench-smoke.sh check
```

检查 50 个实例的数据校验和、评分器字段、分词器与 Docker 连接，打印后续选定
实例和预算。此命令不会调用模型、拉取镜像或执行评分。

### 3. 先验证容器隔离，再运行模型

```bash
# 从官方环境镜像提取运行时与依赖，清除源码、Git、缓存和评分残留，并导入新镜像
bash scripts/swebench-smoke.sh prepare-environments --limit 3
# 不调用模型：逐题检查镜像、宿主访问隔离、编辑、管道与现有测试
bash scripts/swebench-smoke.sh isolation-check --limit 3
```

每题的所有工具都在独立 Docker 容器内运行，无宿主目录挂载，网络为 `none`，
不提供 Docker socket 或 API 密钥。模型请求由宿主控制器发送。基础源码按
Git 的已跟踪文件清单传入容器，再建立只有一个基础提交的 Git 仓库。容器内 Bash 支持
管道和重定向；Python、文件读写及 Git 工具都在相同容器中执行。详见
[隔离与重测说明](docs/benchmark-isolation.md)。

以下是后续运行命令；环境准备不会自动执行它们：

```bash
# 生成并评分默认的 3 题，整批保守估算预算为 2 元
bash scripts/swebench-smoke.sh all --run-id verified-mini-smoke-001

# 或分步生成、评分
bash scripts/swebench-smoke.sh generate --run-id verified-mini-smoke-002
bash scripts/swebench-smoke.sh eval --run-id verified-mini-smoke-002
```

默认限制：最多 3 个实例，每题最多 80 轮，每次最多输出 16384 tokens，低思考
强度，SDK 自动重试关闭。整批共享 `--budget-cny 2`，按已核对的
[Flash 官方高峰价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)
保守估算输入、输出费用；发送下一次请求前预留其费用，无法覆盖时停止。
该额度不是平台账单的硬限额，价格变化时需要更新代码里的费率。默认每题
1800 秒、整批 7200 秒、工具命令 60 秒；可显式调整时间上限。

`finish_reason=length`、空响应及不完整的工具参数均不会计为完成；截断工具
调用不执行。超时终止命令进程组，并重启容器以清理其他后代进程；最多允许
2 次恢复，之后停止该题。空补丁记为 `undelivered_empty_patch`。

可用 `--limit 1` 缩小范围，或用可重复的 `--instance-id` 选择具体的 Mini
实例。调整规模与预算必须显式传参，预算不足时不会自动提高额度或重跑。
每题完成或中止后立即保存预测；未尝试的题记录在汇总中，之后可用具体
`--instance-id` 和新的 run ID 补跑。预算中断不能视为完整的解题能力评测。

明确运行全量 50 题时，可使用：

```bash
bash scripts/swebench-smoke.sh all --limit 50 --budget-cny 150 --batch-seconds 21600 --run-id verified-mini-full-001
```

全量示例显式设置 6 小时生成上限；默认的 2 小时上限更适合小批试跑。

这里的 150 元是按所有输入缓存未命中高峰价计算的保守估算保护额度，不代表
实际费用；缓存命中和时段折扣可能显著降低实际扣费。

全量运行前也需先用 `--limit 50` 准备环境并通过隔离检查；生成入口会重新检查
选定题目的隔离与现有测试，失败时不发送模型请求。正式得分需用统一配置
重新生成全部 50 题。此前完整 clone 或宿主工具执行产生的结果不能作为该
配置的正式成绩。官方公开镜像拉取遇到 WSL 的 Docker 凭据助手故障时，
评分入口会尝试匿名拉取该公开镜像。

每次运行使用新 run ID，结果保存到 `.swebench-work/runs/<run-id>/`：

| 文件 | 内容 |
|---|---|
| `dataset.jsonl` | 仅含生成字段的选题快照 |
| `predictions.jsonl` | 逐题保存的模型补丁 |
| `usage.jsonl` | 逐次模型调用的 tokens、耗时和保守费用估算 |
| `generation-summary.json` | 每题交付状态、补丁哈希、预测文件哈希、耗时、费用及未尝试实例 |
| `evaluation/dataset.jsonl` | 生成结束并校验预测文件哈希后才创建的评分快照 |
| `evaluation-timing.json` | 官方评分阶段用时和退出码 |
| `<model>.<run-id>.json` | 官方汇总评分；详细测试日志在项目 `logs/` 中 |

解决率按本次选定题数统计；少量实例的结果应标注样本范围，不能当作完整
Verified Mini 的 50 题得分。数据、预测、日志、虚拟环境和 `.env.llm` 都被 Git 忽略。

### 最近一次 Mini 全量报告

2026-10-07 的 50 题运行结果为 **41/50（82%）**：8 题非空补丁未通过，1 题为空补丁，评分错误为 0。
配置、耗时、调用统计、逐题结果及官方 JSON 见
[完整报告](docs/benchmarks/2026-10-07-verified-mini/README.md)。

### 其他 SWE-bench 数据集

通用 `python -m myagent.swebench` 和 `python -m myagent.swebench_eval` 入口仍可
手动用于 Lite 等 JSON/JSONL 数据集。通用生成入口同样要求 `--environments`
提供经过检查的 Docker 环境；它不包含 Mini 的整批费用预算保护。补丁捕获
包含已跟踪变更及未被 Git 忽略的新文件，不修改原有暂存区。
Windows 使用 `myagent.swebench_eval` 可保持容器脚本的 LF 换行。

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
工作目录，使用独立只读工具执行器），跑完一次子任务的完整 ReAct 后，把结果
文本化为观察返回给编排者继续推理。主循环负责限流并行委派与串行写入。

```text
编排者 AgentLoop（完整装配：记忆/压缩/审批照常）
  └─ delegate_agent 工具 → 后台线程 → 工人 AgentLoop（独立角色 prompt）
        └─ 工人自己跑 ReAct（read_file / list_files …）
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
  **写入屏障**：编排者等待前一组工人全部结束后，才串行执行写入；写入完成
  后才启动后续委派。可在多组委派之间修改文件，无需等整个任务最终收尾。
- **工人失败隔离**：工人超轮数 / 出错只表现为 `FAILED:` 观察（含原因与轨迹
  节选），不打断编排者；
- **防递归委派**：工人 schema 不含 `delegate_agent`，嵌套委派被禁止；
- **免询问**：`delegate_agent` 标 `risk="read"`（委派本身不改工作区），
  工人不带审批闸门（后台线程不能弹交互确认），写操作由只读白名单杜绝；
- **并发**：同一轮连续的 `delegate_agent` 按 `max_workers` 分组并行，
  每组全部完成后启动下一组，结果按原调用顺序回灌。普通工具保持串行，
  写入、删除、shell 和 patch 操作全部由编排者执行。
- **超时与收尾**：调用上下文显式传入工具调度和超时线程，工人不继承委派权限。
  工具超时会报错，并等待在途线程结束再返回，避免与后续写入重叠；
  Python 线程不能强制取消，因此这不是工人运行时间的硬上限。CLI 退出会关闭工人池。

参数：`--multi_agent`（启用）、`--worker_prompt_dir`（工人提示词目录，默认与
主 agent 同 `prompts/`）、`--max_workers`（并发上限，默认 4）。
