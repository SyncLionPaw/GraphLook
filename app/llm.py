import json
import tomllib
from typing import List, Union

from openai import AsyncOpenAI

from .config import PROJECT_ROOT
from .scheme import Message


class LLMSettings:
    def __init__(
        self, model, base_url, api_key, temperature=1.0, max_tokens=4096, **params
    ):
        self.model: str = model
        self.base_url: str = base_url
        self.api_key: str = api_key
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.api_type: int = params.get("api_type", "openai")


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
        self.api_type = llm_config.api_type

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _init_client_(self):
        if self.client is not None:
            return self.client
        client = None
        match self.api_type:
            case "openai":
                client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            case "ollama":
                client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            case _:
                raise ValueError(f"不支持的 api 类型 {self.api_key}")
        self.client = client
        return client

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

    async def ask(
        self, messages: List[Union[dict, Message]], hook_after_resp=None, **kwargs
    ):
        # 询问LLM，获取响应答案
        formatted_msgs = self.format_message(messages)
        params = {"model": self.model, "messages": formatted_msgs}
        # 是否支持 stream？
        resp = await self.client.chat.completions.create(**params, stream=False)
        if not resp.choices or not resp.choices[0].message.content:
            raise ValueError("LLM返回空或错误")
        return resp.choices[0].message.content

    async def ask_tool(self, messages: List[Union[dict, Message]], **kwargs) -> tuple:
        formatted_msgs = self.format_message(messages)
        if not kwargs.get("tools", None):
            raise KeyError("tools missing")
        params = {
            "model": self.model,
            "messages": formatted_msgs,
            "tools": kwargs["tools"],
        }
        resp = await self.client.chat.completions.create(**params, stream=False)
        if not resp.choices or not resp.choices[0].message.content and not resp.choices[0].message.tool_calls:
            raise ValueError("LLM返回空或错误")
        tool_calls = resp.choices[0].message.tool_calls
        funcs_to_call = []
        if len(tool_calls) > 0:
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                funcs_to_call.append({"name": func_name, "arguments": func_args})
        return resp.choices[0].message.content, funcs_to_call


"""
ChatCompletionMessage(content='', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, 
tool_calls=[ChatCompletionMessageToolCall(id='call_03bc825691904cbfaab12f', 
function=Function(arguments='{"content": "春风拂柳枝，\\n碧波荡漾时。\\n山色远含翠，\\n花香近带脂。\\n天地一壶中，\\n日月双丸里。\\n古今多少事，\\n都付笑谈中。", 
"file_path": "a.txt"}', name='file_saver'), type='function', index=0)])
"""
