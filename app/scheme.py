from typing import Optional


class Message:
    """LLM对话中的消息"""

    def __init__(self, role, content):
        self.role: str = role
        self.ccontent: Optional[str] = content

    def to_dict(self) -> dict:
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content

        return message
