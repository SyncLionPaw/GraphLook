import tomllib
from pathlib import Path
from typing import List, Union

from openai import AsyncOpenAI

from .scheme import Message


def get_project_root() -> Path:
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()


class LLMSettings:
    def __init__(
        self, model, base_url, api_key, temperature=1.0, max_tokens=4096, **params
    ):
        self.model: str = model
        self.base_url: str = base_url
        self.api_key: str = api_key
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens


class Config:
    @classmethod
    def load_config(cls):
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if not config_path.exists():
            raise FileNotFoundError("LLM配置文件不存在")

        with open(config_path, "rb") as f:
            raw_config: dict = tomllib.load(f)
        return raw_config

    @classmethod
    def get_llm_config(cls):
        llm_config = cls.load_config()["llm"]
        fields = ["model", "base_url", "api_key"]

        for field in fields:
            if field not in llm_config:
                raise KeyError(f"{field} not found")
        return LLMSettings(**llm_config)


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
                raise ValueError("错误的消息类型 " + str(type(m)))
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

    async def ask(self, messages: List[Union[dict, Message]], hook_after_resp=None):
        # 询问LLM，获取响应答案
        formatted_msgs = self.format_message(messages)
        params = {"model": self.model, "messages": formatted_msgs}
        # 是否支持 stream？
        resp = await self.client.chat.completions.create(**params, stream=False)
        if not resp.choices or not resp.choices[0].message.content:
            raise ValueError("LLM返回空或错误")
        return resp.choices[0].message.content
