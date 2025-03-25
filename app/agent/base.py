from ..llm import LLM


class BaseAgent:
    def __init__(self, name, desc, llm):
        self.name: str = name
        self.desc: str = desc

        self.llm: LLM = llm

    async def run(self):
        while True:
            q = input("请输入您的问题(quit 或者 exit退出):")
            if q == "quit" or q == "exit":
                return
            resp = await self.llm.ask([{"role": "user", "content": q}])
            print(resp)
