from typing import List, Optional, Union
from openai import OpenAI, AsyncOpenAI
from app.scheme import Message


class LLMSettings:
    model: str
    base_url: str
    api_key: str
    temperature: float = 1.0
    max_tokens: int = 4096


class LLM:
    def __init__(self, llm_config: LLMSettings):
        self.model = llm_config.model
        self.base_url = llm_config.base_url
        self.api_key = llm_config.api_key
        self.temperature = llm_config.temperature

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def format_message(messages: List[Union[dict, Message]]) -> List[dict]:
        # 把消息转换成为OpenAI兼容的消息格式
        ans = []
        for m in messages:
            if not (isinstance(m, dict) or isinstance(m, Message)):
                raise ValueError("错误的消息类型 " + type(m))
            if isinstance(m, Message):
                m = m.to_dict()
            elif isinstance(m, dict):
                if "role" not in m:
                    raise ValueError("消息必须包含 role 字段")
                if not m.get("content"):
                    m["content"] = []
                elif isinstance(m["content"], str):
                    m["content"] = [{"type": "text", "text": m["content"]}]
                elif isinstance(m["content"], list):
                    m["content"] = [
                        (
                            {"type": "text", "text": item}
                            if isinstance(item, str)
                            else item
                        )
                        for item in m["content"]
                    ]
                if "content" in m:
                    ans.append(m)
        return ans

    async def ask(self, messages: List[Union[dict, Message]]):
        # 询问LLM，获取响应答案
        formatted_msgs = self.format_message(messages)
        params = {"model": self.model, "message": formatted_msgs}
        # 是否支持 stream？
        resp = await self.client.chat.completions.create(**params, stream=False)
        return resp
