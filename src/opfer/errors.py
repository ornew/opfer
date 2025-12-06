from __future__ import annotations


class AgentException(Exception):
    pass


class ModelBehaviorError(AgentException):
    pass


class ModelIncompleteError(AgentException):
    reason: str
    message: str

    def __init__(self, reason: str, message: str):
        self.reason = reason
        self.message = message
        super().__init__(f"model incompleted: {reason} - {message}")


class ModelRefusalError(AgentException):
    reason: str
    message: str

    def __init__(self, reason: str, message: str):
        self.reason = reason
        self.message = message
        super().__init__(f"model refused: {reason} - {message}")


class RetryableError(AgentException):
    pass


class MaxTurnsExceeded(AgentException):
    pass


class MaxStepsExceeded(AgentException):
    pass


class UserError(AgentException):
    pass


class ToolError(AgentException):
    pass
