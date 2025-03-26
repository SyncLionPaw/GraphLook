from typing import List, Optional, Union
from .tools.tool import ToolCall


class Message:
    """LLM对话中的消息"""

    def __init__(self, role, content, tool_calls):
        self.role: str = role
        self.content: Optional[str] = content
        self.tool_calls: List[ToolCall] = tool_calls

    def to_dict(self) -> dict:
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [item for item in self.tool_calls]
        return message

    def __str__(self):
        return f"{self.role}-{self.content[:10]}"


class Memory:
    """就是消息数组 为了LLM的上下文而存在"""

    def __str__(self):
        ans = ""
        for m in self.messages:
            ans += str(m)
            ans += "\n"
        return ans

    def __init__(self, max_message: int):
        self.messages: List[Union[Message, dict]] = []
        self.max_message = max_message

    def add_message(self, m: Message):
        self.messages.append(m)
        if len(self.messages) > self.max_message:
            system_message = self.messages[0]
            self.messages = [system_message] + self.messages[-self.max_message :]

    def to_dict_list():
        pass

    def get_recent(self, n: int):
        return self.messages[-n:]
