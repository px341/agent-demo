"""工具错误体系：分类 + 分级退出语义。

所有工具错误都抛出 :class:`ToolError` 子类，由主循环（AgentLoop）统一
处理退出语义：

- ``recoverable=True``（模型可自行修正）→ 转错误文本观察回灌，继续循环；
- ``recoverable=False``（环境 / 安全 / 未知）→ 终止本轮 ReAct 循环。

分类参考主流框架（OpenAI function calling 无标准错误枚举，错误以
tool 消息 content 文本形式返回，由模型解读；此处自定义分类并序列化
进观察文本）。

注意：``ToolPermissionError`` 名称避开 builtins.PermissionError，
但其 ``error_type`` 仍为 ``"PermissionError"``（与用户给出的分类对齐）。
"""
from __future__ import annotations


class ToolError(Exception):
    """工具错误基类。"""

    #: 错误类型名（用于序列化进观察文本与日志）。
    error_type: str = "ToolError"

    #: 是否可恢复：True → 回灌给模型继续；False → 终止 ReAct 循环。
    recoverable: bool = False

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"ToolError[{self.error_type}]: {self.message}"


class ValidationError(ToolError):
    """参数不合法 / 逻辑校验失败（模型可修正参数后重试）。"""

    error_type = "ValidationError"
    recoverable = True


class NotFoundError(ToolError):
    """资源不存在（路径错误，模型可换路径后重试）。"""

    error_type = "NotFoundError"
    recoverable = True


class ToolPermissionError(ToolError):
    """权限不足 / 越界（如路径逃逸、禁止操作工作目录本身）。"""

    error_type = "PermissionError"
    recoverable = False


class ApprovalDenied(ToolError):
    """用户拒绝审批（终止本轮任务）。"""

    error_type = "ApprovalDenied"
    recoverable = False


class ToolTimeoutError(ToolError):
    """工具执行超时。"""

    error_type = "TimeoutError"
    recoverable = False


class ExecutionError(ToolError):
    """工具执行抛出未知异常（兜底分类）。"""

    error_type = "ExecutionError"
    recoverable = False
