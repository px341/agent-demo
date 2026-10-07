# SWE-bench 生成与评分边界

生成工具执行在每题新建的 Docker 容器中。工具容器不挂载宿主目录、HF
缓存、历史预测、其他题目、Git 镜像或 Docker socket；使用 `network_mode=none`、
删除全部 capabilities、禁止权限提升，并限制内存、CPU 和进程数。源码经
Git 的已跟踪文件清单复制，保留符号链接，容器 Git 仓库只包含基础快照。
API 客户端和密钥留在宿主。

准备阶段可以联网获取公开数据、基础提交与环境镜像。生成阶段工具无网络。
`prepare-environments` 从官方评分镜像导出文件系统，仅保留系统与指定 Python
环境，丢弃原题仓库、Git、缓存、安装脚本、家目录、补丁和评分产物，再导入
独立镜像；不继承原镜像的环境变量、入口程序或挂载声明。生成前检查固定
镜像 ID、容器实际挂载与网络配置、空的原题目录，以及依赖环境中不存在可
导入的原题包；导入题目源码后再次确认模块来自 `/testbed`。

生成数据采用四字段白名单，未知字段也不会流入生成导出。评分数据保存在
独立路径。模型停止后先保存预测与逐题 SHA-256，再封存预测文件的 SHA-256。
评分入口拒绝未封存或已修改的预测文件。随后在独立评分子进程中加载隐藏
测试及评分配置，创建独立评分容器；评分输出没有回灌 agent 的接口。旧运行
不能追加生成或复用为本次正式结果。

## 无模型检查

```bash
bash scripts/swebench-smoke.sh prepare
bash scripts/swebench-smoke.sh prepare-environments --limit 1
bash scripts/swebench-smoke.sh isolation-check --limit 1
```

隔离检查在真实宿主临时目录放置随机答案缓存和 Git blob，尝试用绝对路径、
`../`、符号链接、Python `open()`、shell 与 `git --git-dir` 读取。检查宿主与
公共网络访问失败、无 socket 与密钥，同时验证正常文件编辑、管道和重定向，
验证已有源码可修改并还原、超时反馈及独立 session 的后代进程已被清理，最后运行仓库现有测试。
宿主保留旧泄漏样本时，还会尝试访问旧答案数据、实际 HF 缓存文件和外部
Git 镜像中的已知未来提交；只记录访问是否失败，内容不传给生成模型。
结果写到环境清单旁的 `.isolation.json`。

默认 Mini 的两个项目使用 Django `basic` 测试和 Sphinx 的 `tests/test_util.py`
作环境试跑。其他项目需要在环境清单中指定适合其基础版本的
`source_module`、`python`、`check_command` 和 `smoke_command`。环境检查失败应
修复准备配置，不能跳过检查继续付费。容器工具调度兼容旧题目的 Python 3.6。

## 小批试跑与全量

```bash
bash scripts/swebench-smoke.sh generate --limit 1 --budget-cny 2 --run-id container-smoke-001
bash scripts/swebench-smoke.sh eval --run-id container-smoke-001
# 校准完成且预算允许后，使用相同配置重新生成全部题目
bash scripts/swebench-smoke.sh prepare-environments --limit 50
bash scripts/swebench-smoke.sh all --limit 50 --budget-cny 150 --batch-seconds 21600 --run-id container-full-001
```

默认输出上限 16384 tokens、每题 80 轮、命令 60 秒、API 请求 180 秒、
每题 1800 秒、整批 7200 秒。`--request-timeout` 可校准 API 等待时间，仍受
题目和整批时间上限约束。预算依据保守高峰费率估算；API 请求前预留下一次的费用，自动重试
关闭，失败费用不确定时停止整批。调整预算、时间、轮数或输出上限都必须
显式传参。`usage.jsonl` 记录 tokens、finish reason、耗时与费用估算。

`length`、空响应及不完整工具调用会被拒绝；整批不完整工具调用均不执行。
超时先杀掉命令进程组，再重启容器以终止逃离该组的后代，最多恢复 2 次。
补丁收集包含新增的非忽略文件，使用独立索引，不依赖模型是否暂存或提交。
空补丁明确记录为未交付，不以模型的完成陈述替代代码交付。
