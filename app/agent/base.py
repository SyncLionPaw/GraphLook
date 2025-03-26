from typing import Dict, List
from app.scheme import Memory
from ..llm import LLM


class BaseAgent:
    """只具备简单问答功能的对话agent"""

    system_prompt = "你是一个智能助手，擅长使用十分简洁的语言来回答用户的问题。"

    def __init__(self, name, desc, llm):
        self.name: str = name
        self.desc: str = desc

        self.llm: LLM = llm

    async def run(self, mode="debug"):
        while True:
            q = input("请输入您的问题(quit 或者 exit退出):\nUser:")
            if q == "quit" or q == "exit":
                return
            resp = await self.llm.ask(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": q},
                ]
            )
            print("Agent:", resp, "\n")


class MemoryAgent(BaseAgent):
    """BaseAgent 不具备上下文能力, MemoryAgent 能够增加上下文信息"""

    def __init__(self, name, desc, llm, max_message):
        super().__init__(name, desc, llm)
        self.memory: Memory = Memory(max_message)
        self.memory.add_message({"role": "system", "content": self.system_prompt})

    def update_memory(self, content: str, role: str = "system"):
        item = {"role": role, "content": content}
        self.memory.add_message(item)

    def get_function_tools(self) -> List[Dict[str, dict]]:
        return []

    async def run(self, mode="debug"):
        while True:
            q = input("请输入您的问题(quit 或者 exit退出):\nUser:")
            if q == "quit" or q == "exit":
                return

            self.update_memory(q, "user")
            if mode == "debug":
                print(self.memory)

            resp = await self.llm.ask(self.memory.messages)  # 上下文的来源，token消耗
            print("Agent:", resp, "\n")
            self.update_memory(resp)


class CotAgent(BaseAgent):
    """具备 chain of thought 能力的思考 agent"""

    pass
