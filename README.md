我会按“入口与装配、运行时、工具系统、状态持久化、上下文与记忆、模型适配、评估”来梳理 `pico/`，并特别说明 `checkpoint.py` 在恢复链路里的位置。先把全部模块和公开结构盘一遍。

模块清单已经完整：共 24 个 Python 文件，约 6750 行。整体不是“每个文件各自跑”，而是 `cli → Pico/AgentLoop → Context/Provider → ToolExecutor/Tools → Session/Run/Checkpoint` 这条主链；我正在核对每层的实际职责边界，避免只按文件名猜。

核心边界已经核对完了：`runtime.py` 是门面和共享状态中心，真正的循环已拆到 `agent_loop.py`；`checkpoint.py` 不保存文件副本，而是保存“任务进度 + 关键文件哈希 + 运行身份”，这一点很关键。接下来我把逐文件说明和整体数据流整理成一份可直接当代码导览看的结构。

`pico/` 可以理解成一个“小型 coding agent 运行平台”：CLI 负责启动，`Pico` 负责装配状态，`AgentLoop` 驱动模型循环，工具层负责实际读写代码，记忆与 checkpoint 负责跨轮恢复。

```text
CLI 启动
  ↓
Workspace + 配置 + Provider
  ↓
Pico Runtime
  ↓
ContextManager 组装 Prompt
  ↓
ModelClient.complete()
  ↓
解析为 tool / retry / final
  ↓
ToolExecutor → Tools
  ↓
更新 History / Memory / Checkpoint
  ↓
Session + Run artifacts 落盘
```

## 入口与装配

- [pico/__main__.py](/Users/user/pico/pico/__main__.py:1)

  `python -m pico` 的入口，只调用 `cli.main()`，没有业务逻辑。

- [pico/__init__.py](/Users/user/pico/pico/__init__.py:1)

  包的公共 API 门面。把 `Pico`、`WorkspaceContext`、模型客户端和 CLI 构建函数导出，让外部可以写：

  ```python
  from pico import Pico, FakeModelClient
  ```

- [pico/cli.py](/Users/user/pico/pico/cli.py:1)

  命令行总入口，主要负责：

  - 解析 `--cwd`、`--provider`、`--model`、`--resume` 等参数
  - 根据优先级选择模型和 provider
  - 构建 `WorkspaceContext`
  - 加载目标仓库的 `.env`
  - 创建或恢复 `SessionStore`
  - 装配 `Pico`
  - 进入一次性调用或 REPL
  - 处理 `/help`、`/memory`、`/session`、`/reset`

  它是“配置和对象装配层”，不直接运行 agent 循环。

- [pico/config.py](/Users/user/pico/pico/config.py:1)

  项目级配置读取器：

  - 向父目录查找 `.env`
  - 解析 `NAME=value` 和 `export NAME=value`
  - 把配置放入当前进程环境变量
  - 提供新旧环境变量的回退读取

  它不是完整 dotenv 实现，不支持变量展开，例如：

  ```env
  B=$A
  ```

  这里会把 `$A` 当作普通字符串。

## 运行时核心

- [pico/runtime.py](/Users/user/pico/pico/runtime.py:1)

  整个系统的核心门面，`Pico` 类持有几乎所有运行时依赖：

  - model client
  - workspace
  - session、memory、checkpoint
  - tool registry、tool executor
  - context manager
  - feature flags
  - 审批策略和只读状态
  - trace/report 状态

  主要职责包括：

  - 创建和恢复 agent
  - 构建、刷新稳定 prompt prefix
  - 更新工作记忆和持久记忆
  - 解析模型返回的 `<tool>`、`<final>`
  - 创建受限的只读子 agent
  - 审批危险工具
  - 限制文件路径不能逃逸出仓库
  - 脱敏 trace/report
  - 将 `ask()` 转交给 `AgentLoop`

  它本身已经接近“God Object”：状态管理、解析、安全、记忆和委派接口都汇集于此，不过主循环已经单独拆出。

- [pico/agent_loop.py](/Users/user/pico/pico/agent_loop.py:10)

  真正驱动一次 `ask()` 的控制循环。

  每次用户请求的流程是：

  1. 把用户请求写入 history 和 task summary
  2. 创建 `TaskState` 和 run 目录
  3. 重新构建 prompt
  4. 请求模型
  5. 将返回内容解析成 `tool`、`retry` 或 `final`
  6. 执行工具并记录结果
  7. 每次工具执行后创建 checkpoint
  8. 得到最终回答后写 report
  9. 达到步数或重试上限时正常收口

  这里区分了：

  - `attempts`：模型调用次数
  - `tool_steps`：实际工具调用次数

  因此格式错误的模型输出只消耗重试次数，不消耗工具步数。

- [pico/task_state.py](/Users/user/pico/pico/task_state.py:1)

  描述“某一次用户请求”的状态机，而不是整个会话。

  保存的信息包括：

  - `run_id`、`task_id`
  - 请求内容
  - 当前状态
  - 停止原因
  - 模型尝试次数
  - 工具调用次数和最后一个工具
  - checkpoint ID
  - 最终回答

  支持的停止原因包括步数上限、重试上限、模型错误、工具超时、审批拒绝等。

## Checkpoint 与恢复

- [pico/checkpoint.py](/Users/user/pico/pico/checkpoint.py:1)

  这个文件处理的是“任务现场恢复”，不是 Git checkpoint，也不保存代码文件副本。

  一个 checkpoint 主要包含：

  - 当前目标
  - 已完成和排除项
  - 当前阻塞原因
  - 建议下一步
  - 最近处理的关键文件
  - 关键文件内容哈希
  - 运行时身份
  - 父 checkpoint ID

  运行时身份包括：

  - 工作区根目录
  - 模型和 client 类型
  - 审批策略
  - 是否只读
  - 最大步数/token
  - feature flags
  - 工具集合签名
  - workspace fingerprint

  恢复状态分为：

  - `no-checkpoint`：没有 checkpoint
  - `full-valid`：文件和运行身份均匹配
  - `partial-stale`：关键文件内容已变化
  - `workspace-mismatch`：模型、工具、目录、配置等运行身份变化
  - `schema-mismatch`：checkpoint 数据版本不兼容

  checkpoint 会在以下时机创建：

  - 工具执行后
  - prompt 发生上下文压缩时
  - 检测到文件过期或运行环境变化时
  - 正常完成时
  - 达到步骤/重试上限时

  需要注意：

  - 它只对 `memory.recent_files` 中的文件做内容哈希。
  - `completed` 当前主要保存最终回答，不是结构化的任务清单。
  - 当“文件过期”和“运行身份变化”同时发生时，状态优先显示为 `partial-stale`。
  - checkpoint 数据仍保存在 session JSON 里，不是独立文件。

## 工作区与 Prompt

- [pico/workspace.py](/Users/user/pico/pico/workspace.py:1)

  生成仓库的轻量快照，提供给模型作为初始背景。

  收集：

  - `cwd`
  - Git 仓库根目录
  - 当前分支和默认分支
  - `git status`
  - 最近五次提交
  - 根目录或目标目录中的 `AGENTS.md`、`README.md`、`pyproject.toml`、`package.json`

  `fingerprint()` 用这些信息生成指纹，以判断稳定 prompt prefix 是否需要更新。

  此处也解释了之前的 `--cwd` 行为：如果指定的是 Git 仓库子目录，真正的工具根目录仍然是整个 Git 仓库根目录。

- [pico/prompt_prefix.py](/Users/user/pico/pico/prompt_prefix.py:1)

  构建相对稳定的 prompt 前缀，其中包含：

  - agent 工作规则
  - 工作区快照
  - 可用工具及参数结构
  - 模型应遵守的 `<tool>`/`<final>`输出格式

  同时生成：

  - prefix hash
  - workspace fingerprint
  - tool signature
  - 构建时间

  prefix hash 也用作支持缓存的 provider 的 `prompt_cache_key`。

- [pico/context_manager.py](/Users/user/pico/pico/context_manager.py:1)

  负责把所有上下文拼成最终 prompt，并控制字符预算。

  Prompt 顺序是：

  ```text
  稳定 prefix
  工作记忆
  相关记忆
  历史记录
  当前用户请求
  ```

  默认总预算约 12000 字符。超出预算后依次压缩：

  1. 相关记忆
  2. 历史记录
  3. 工作记忆
  4. prefix

  当前用户请求不会被裁剪。

  它还会：

  - 保留最近六条历史
  - 折叠重复的旧文件读取
  - 用文件摘要替换旧的完整读取结果
  - 压缩旧 shell 输出
  - 记录每一部分压缩前后的字符数

## 记忆系统

- [pico/features/__init__.py](/Users/user/pico/pico/features/__init__.py:1)

  `features` 子包标识文件。目前没有额外导出逻辑。

- [pico/features/memory.py](/Users/user/pico/pico/features/memory.py:1)

  实现分层记忆，分成三类：

  1. 工作记忆

     当前任务摘要、最近文件、文件摘要。

  2. 情节记忆

     最近工具和文件读取产生的短笔记，保存在 session 中。

  3. 持久记忆

     稳定的项目约定、关键决策、依赖事实和用户偏好，保存在：

     ```text
     .pico/memory/
     ```

  其他职责：

  - 规范化旧版 memory 数据
  - 将绝对路径转换为仓库相对路径
  - 计算文件哈希，自动淘汰过期摘要
  - 根据标签、关键词和时间召回相关记忆
  - 去重、限制数量
  - 提升持久记忆并处理新旧事实覆盖

  召回目前是透明的关键词匹配，没有使用 embedding。

## 工具系统

- [pico/tool_context.py](/Users/user/pico/pico/tool_context.py:1)

  工具能看到的最小运行上下文。

  只暴露：

  - 工作区根目录
  - 安全路径解析函数
  - 过滤后的 shell 环境
  - 委派深度
  - 创建子 agent 的回调

  这是依赖收窄层，避免 `tools.py` 直接依赖整个 `Pico` 对象。

- [pico/tools.py](/Users/user/pico/pico/tools.py:1)

  定义 agent 的能力白名单和具体实现。

  当前工具：

  - `list_files`
  - `read_file`
  - `search`
  - `run_shell`
  - `write_file`
  - `patch_file`
  - `delegate`

  它负责：

  - 工具 schema、风险等级和描述
  - 参数校验
  - 工具示例
  - 文件读写和搜索
  - shell 命令执行
  - 注册受深度限制的 `delegate`

  `patch_file` 要求 `old_text` 精确且只出现一次，以避免模糊修改。

- [pico/tool_executor.py](/Users/user/pico/pico/tool_executor.py:1)

  工具调用的统一护栏。

  执行顺序是：

  ```text
  allowlist
    → 工具是否存在
    → 参数校验
    → 重复调用检测
    → 风险审批
    → 工作区快照
    → 执行
    → 比较改动
    → 更新记忆
    → 返回结构化元数据
  ```

  它会区分：

  - `ok`
  - `error`
  - `rejected`
  - `partial_success`

  例如 shell 返回非零状态，但已经修改了文件，会标记为 `partial_success`，提醒下一轮先检查 diff。

  值得注意的是，文件工具有严格的仓库路径限制；但 `run_shell` 使用 `shell=True` 执行任意命令，主要依靠风险审批和过滤环境变量保护，它并不是操作系统级沙箱。

## 模型 Provider

- [pico/providers/__init__.py](/Users/user/pico/pico/providers/__init__.py:1)

  Provider 子包的公共导出文件。

- [pico/providers/clients.py](/Users/user/pico/pico/providers/clients.py:1)

  将不同模型接口统一成：

  ```python
  complete(prompt, max_new_tokens, ...)
  ```

  包含：

  - `FakeModelClient`：测试和 benchmark 使用的脚本化模型
  - `OllamaModelClient`：调用 `/api/generate`
  - `OpenAICompatibleModelClient`：调用 `/v1/responses`
  - `AnthropicCompatibleModelClient`：调用 `/v1/messages`

  还负责：

  - HTTP 错误处理和重试
  - 普通 JSON/SSE 响应解析
  - 提取 token usage
  - 提取 prompt cache 命中信息
  - 隐藏不同 provider 的协议差异

  当前只有部分 OpenAI-compatible 后端启用了 prompt cache；Anthropic-compatible 和 Ollama 路径没有接入缓存语义。

## 持久化与审计

- [pico/session_store.py](/Users/user/pico/pico/session_store.py:1)

  保存可恢复会话：

  ```text
  .pico/sessions/<session_id>.json
  ```

  包含 history、memory、checkpoint 等状态，并支持查找最新 session。

- [pico/run_store.py](/Users/user/pico/pico/run_store.py:1)

  保存单次运行的审计工件：

  ```text
  .pico/runs/<run_id>/
    task_state.json
    trace.jsonl
    report.json
  ```

  三者分别表示：

  - `task_state.json`：当前任务状态
  - `trace.jsonl`：逐事件时间线
  - `report.json`：最终结果汇总

  JSON 文件使用临时文件替换实现原子写入；`trace.jsonl` 则是追加写入。

- [pico/security.py](/Users/user/pico/pico/security.py:1)

  负责环境变量与工件脱敏：

  - 根据 `API_KEY`、`TOKEN`、`SECRET`、`PASSWORD` 等名字检测敏感变量
  - 将实际值替换成 `<redacted>`
  - 递归清理字典、列表和字符串
  - 生成只包含允许变量的 shell 环境

  它主要防止密钥进入 trace/report，并不是完整的命令执行安全沙箱。

## 评估系统

- [pico/evaluation/__init__.py](/Users/user/pico/pico/evaluation/__init__.py:1)

  评估子包标识文件。

- [pico/evaluation/evaluator.py](/Users/user/pico/pico/evaluation/evaluator.py:1)

  固定 benchmark 执行器。

  负责：

  - 校验 benchmark JSON schema
  - 复制测试 fixture 到临时目录
  - 用 `FakeModelClient` 执行确定性任务
  - 应用任务初始化条件
  - 检查预期文件、内容、工具调用和停止原因
  - 汇总结果并生成 benchmark artifact

  它用于验证 agent harness 本身，而不是日常 CLI 运行。

- [pico/evaluation/metrics.py](/Users/user/pico/pico/evaluation/metrics.py:1)

  大规模实验和指标汇总模块，也是当前最大的单文件。

  包含：

  - benchmark artifact 聚合
  - run trace 指标统计
  - memory/context 功能消融实验
  - 压力测试
  - 安全场景实验
  - provider 对比实验
  - 真实模型实验
  - checkpoint/resume 恢复指标
  - Markdown 报告生成

  它属于研究和回归验证基础设施，不在正常 `pico` CLI 主链上。

## 总体评价

这套代码的分层方向是清楚的：

- `cli.py`：启动和装配
- `runtime.py`：共享状态与门面
- `agent_loop.py`：控制循环
- `context_manager.py`：模型输入
- `tool_executor.py`：安全执行边界
- `memory.py` + `checkpoint.py`：跨轮恢复
- `session_store.py` + `run_store.py`：持久化和审计
- `clients.py`：模型协议适配

目前最明显的结构问题是 `runtime.py` 仍承担较多职责，而 `evaluation/metrics.py` 已经非常庞大。后续如果继续扩展，可以优先把 `runtime.py` 中的模型输出解析、持久记忆提升、workspace diff 分别拆成独立模块。