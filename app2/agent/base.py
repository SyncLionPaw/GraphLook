import sys
from app.scheme import Memory


class BaseAgent:
    """可拓展的对话agent"""

    system_prompt = "你是一个智能助手，擅长用简单的语言回答用户的问题，满足用户的要求。"

    def __init__(self, name, desc, llm, max_message):
        self.name = name
        self.desc = desc
        self.llm = llm
        self.memory = Memory(max_message)
        self.memory.add_message({"role": "system", "content": self.system_prompt})

    async def run(self):
        context = {}
        while True:
            q = await self.get_user_input()
            context["q"] = q

            await self.hook_before_ask_llm(context)

            resp = await self.llm.ask(self.memory.messages)
            context["resp"] = resp

            await self.hook_after_ask_llm(context)

            await self.response_user(context=context)

    async def hook_before_ask_llm(self, context: dict):
        # update question into memory
        self.memory.add_message({"role": "system", "content": context["q"]})
        return

    async def hook_after_ask_llm(self, context: dict):
        return

    async def get_user_input(self):
        q = input("请输入您的问题(quit 或者 exit退出):\nUser:")
        if q == "quit" or q == "exit":
            await self.response_user({"resp": "bye!"})
            sys.exit(0)
        return q

    async def response_user(self, context: dict):
        resp = context["resp"]
        print("Agent:", resp, "\n")
