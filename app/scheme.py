from typing import Optional


class Message:
    """LLM对话中的消息"""

    role: str
    content: Optional[str]

    def to_dict(self) -> dict:
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content

        return message
