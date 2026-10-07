# SWE-bench Verified Mini 全量评测报告（2026-10-07）

本轮全部 50 个固定实例均已尝试并提交预测。官方评分解决 **41/50 题，解决率 82%**；8 个非空补丁未通过测试，1 个空补丁计为未解决。没有缺失预测、评分错误、基础设施失败或模糊失败。

[官方原始报告](official-report.json) 原样归档，未包含预测补丁、任务数据、隐藏测试或模型轨迹。

| 范围 | 总题数 | 解决 | 非空补丁未解决 | 空补丁 | 解决率 |
|---|---:|---:|---:|---:|---:|
| Django | 25 | 23 | 2 | 0 | 92% |
| Sphinx | 25 | 18 | 6 | 1 | 72% |
| 全量 | 50 | 41 | 8 | 1 | 82% |

## 运行范围与配置

- Run ID：`verified-mini-50-20261007-163115`。
- 数据集：`MariusHobbhahn/swe-bench-verified-mini`，test split，revision `b316c349947c29963fce3f4a65967c9807a4b673`；固定 50 题，Django 和 Sphinx 各 25 题。
- 配置与运行支持提交：`40613f1242c732cb337e6b8f031c3fe78a35553c`。
- 模型：`deepseek-flash`，官方 DeepSeek API；低思考强度，SDK 自动重试 0 次。
- 每题最多 80 轮，每次最多输出 16384 tokens；单题 1800 秒、命令 60 秒、API 请求 180 秒。
- 生成与评分均为单并发；生成结束并封存预测后才开始独立官方评分。
- 整批生成时限启动时为 7200 秒；进行第 34 题时，经用户要求延长为 21600 秒（6 小时）。保留所有已完成预测，其他解题参数与 150 元估算预算未改变。复现命令从启动时即显式设置 6 小时。
- 生成路径不注入历史会话记忆；模型工具在逐题离线 Docker 容器内执行。
- 评分器 `swebench=5.0.2`；Python 3.12.3、Docker 29.8.1、datasets 5.1.0、tiktoken 0.14.0。

## 时间、调用与费用口径

- 开始：2026-10-07 16:32:46；结束：2026-10-07 21:33:04，均为北京时间。任务总耗时约 300.30 分钟，包含首次准备 50 个环境、隔离检查、生成与评分。
- 生成计时：145.42 分钟；官方评分计时：21.60 分钟。
- 模型调用 2,065 次；记录输入 62,058,937 tokens、输出 1,021,322 tokens。
- 按运行代码的保守高峰费率（输入 2 元/百万 tokens、输出 8 元/百万 tokens）累计估算 **132.288450 元**，低于 150 元保护额度。该数把输入缓存命中也按未命中高峰价计算，**不是平台实际扣费**；本轮未归档实际账单。

## 隔离、封存与结果解释

- 50/50 个环境通过实际隔离验收和基础测试。模型工具容器网络为 `none`，无宿主目录挂载、Docker socket 或 API 密钥；仅复制基础提交的已跟踪源码。
- 生成输入只保留 `instance_id`、`repo`、`base_commit`、`problem_statement` 四字段。答案和隐藏测试仅由独立评分进程加载，评分输出没有回传到本轮模型解题路径。
- 所有题目重新生成，未复用历史运行的预测或结果。评估前校验整份预测、逐题补丁和生成数据 SHA-256；归档前再次通过 `verify_seal()`。
- 生成终止状态：`final_answer=43`、`max_turns=7`。达到轮数上限仍会保存当前补丁并评分，终止状态不等同于测试是否通过。
- 顶层退出码为 1，表示生成阶段存在达到轮数上限或空补丁的实例；官方评分阶段退出码为 0。全部 50 题已尝试，评分错误为 0。
- 评分完成后无未停止的测试容器；环境和评分镜像保留为本地缓存。

## 校验信息

- 生成数据 SHA-256：`bcf7db07f276dc1cc437ae9556c85e02e3acb1c49f66021b63fad6d006f2f8a7`。
- 预测文件 SHA-256：`0d82d84e1e38b1fde64ac7faf47069a70e9330557cbc50b82b4e1b39e19ad315`。
- 归档官方 JSON SHA-256：`1c9f7ee331795a7febebf8920f636edaed866b563217987d24a6597757f33c19`。
- 运行支持提交验证：315 项本地回归测试、2 项子测试通过；`git diff --check`、Shell 语法检查通过。

## 复现命令

在配置 `.env.llm` 并安装 benchmark 依赖后执行；每次使用新的 run ID：

```bash
bash scripts/swebench-smoke.sh prepare
bash scripts/swebench-smoke.sh prepare-environments --limit 50
bash scripts/swebench-smoke.sh all --limit 50 --budget-cny 150 \
  --batch-seconds 21600 --run-id verified-mini-full-reproduction-001
```

`all` 会在模型请求前自动检查这 50 题的隔离和基础测试。密钥、数据集、预测、逐题轨迹、用量明细、Docker 环境清单与旧测试结果均保留在本地忽略目录。

## 逐题结果

| 实例 | 官方结果 | 生成终止状态 | 轮数 | 补丁字符数 |
|---|---|---|---:|---:|
| `django__django-11790` | 通过 | `final_answer` | 15 | 1863 |
| `django__django-11815` | 通过 | `final_answer` | 17 | 4316 |
| `django__django-11848` | 通过 | `final_answer` | 16 | 899 |
| `django__django-11880` | 通过 | `final_answer` | 23 | 440 |
| `django__django-11885` | 通过 | `final_answer` | 73 | 1995 |
| `django__django-11951` | 通过 | `final_answer` | 17 | 870 |
| `django__django-11964` | 通过 | `final_answer` | 12 | 1457 |
| `django__django-11999` | 通过 | `final_answer` | 14 | 2094 |
| `django__django-12039` | 通过 | `final_answer` | 29 | 3957 |
| `django__django-12050` | 通过 | `final_answer` | 20 | 649 |
| `django__django-12143` | 通过 | `final_answer` | 28 | 2063 |
| `django__django-12155` | 通过 | `final_answer` | 20 | 700 |
| `django__django-12193` | 通过 | `final_answer` | 56 | 552 |
| `django__django-12209` | 通过 | `max_turns` | 80 | 2130 |
| `django__django-12262` | 通过 | `final_answer` | 20 | 1385 |
| `django__django-12273` | 未通过 | `final_answer` | 55 | 820 |
| `django__django-12276` | 通过 | `final_answer` | 24 | 1628 |
| `django__django-12304` | 通过 | `final_answer` | 18 | 758 |
| `django__django-12308` | 通过 | `final_answer` | 26 | 854 |
| `django__django-12325` | 未通过 | `final_answer` | 30 | 1109 |
| `django__django-12406` | 通过 | `max_turns` | 80 | 4160 |
| `django__django-12708` | 通过 | `final_answer` | 58 | 833 |
| `django__django-12713` | 通过 | `final_answer` | 28 | 2882 |
| `django__django-12774` | 通过 | `final_answer` | 26 | 940 |
| `django__django-9296` | 通过 | `final_answer` | 14 | 499 |
| `sphinx-doc__sphinx-10323` | 通过 | `final_answer` | 38 | 2145 |
| `sphinx-doc__sphinx-10435` | 通过 | `final_answer` | 26 | 874 |
| `sphinx-doc__sphinx-10466` | 通过 | `final_answer` | 49 | 482 |
| `sphinx-doc__sphinx-10673` | 通过 | `max_turns` | 80 | 3770 |
| `sphinx-doc__sphinx-11510` | 未通过 | `final_answer` | 68 | 2431 |
| `sphinx-doc__sphinx-7590` | 未通过 | `final_answer` | 70 | 3762 |
| `sphinx-doc__sphinx-7748` | 未通过 | `final_answer` | 70 | 3548 |
| `sphinx-doc__sphinx-7757` | 通过 | `final_answer` | 23 | 3059 |
| `sphinx-doc__sphinx-7985` | 未通过 | `max_turns` | 80 | 1361 |
| `sphinx-doc__sphinx-8035` | 通过 | `final_answer` | 68 | 3916 |
| `sphinx-doc__sphinx-8056` | 未通过 | `final_answer` | 73 | 1583 |
| `sphinx-doc__sphinx-8265` | 通过 | `final_answer` | 65 | 1806 |
| `sphinx-doc__sphinx-8269` | 通过 | `final_answer` | 29 | 625 |
| `sphinx-doc__sphinx-8475` | 通过 | `final_answer` | 12 | 2114 |
| `sphinx-doc__sphinx-8548` | 通过 | `final_answer` | 66 | 2277 |
| `sphinx-doc__sphinx-8551` | 通过 | `final_answer` | 56 | 2298 |
| `sphinx-doc__sphinx-8638` | 通过 | `max_turns` | 80 | 698 |
| `sphinx-doc__sphinx-8721` | 通过 | `final_answer` | 36 | 1595 |
| `sphinx-doc__sphinx-9229` | 未通过 | `max_turns` | 80 | 3200 |
| `sphinx-doc__sphinx-9230` | 通过 | `final_answer` | 37 | 2246 |
| `sphinx-doc__sphinx-9281` | 通过 | `final_answer` | 31 | 540 |
| `sphinx-doc__sphinx-9320` | 通过 | `final_answer` | 14 | 1348 |
| `sphinx-doc__sphinx-9367` | 通过 | `final_answer` | 20 | 1143 |
| `sphinx-doc__sphinx-9461` | 空补丁（未解决） | `max_turns` | 80 | 0 |
| `sphinx-doc__sphinx-9698` | 通过 | `final_answer` | 15 | 1316 |
