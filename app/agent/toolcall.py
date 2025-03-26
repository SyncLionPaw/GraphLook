import json
from typing import Dict, List, Optional

from app.agent.base import MemoryAgent

from app.tools.tool import BaseTool, FileSaver

TOOLCALL_PROMPT = (
    """你是一个擅长使用工具的agent，根据用户的需求，判断是否需要使用工具"""
)


DEFAULT_TOOLS = [FileSaver()]


class ToolCallAgent(MemoryAgent):
    """能够使用工具的agent"""

    system_prompt = TOOLCALL_PROMPT

    def __init__(
        self, name, desc, llm, max_message, tools: Optional[List[BaseTool]] = None
    ):
        super().__init__(name, desc, llm, max_message)
        self.tools: List[BaseTool] = DEFAULT_TOOLS
        if tools:
            self.tools += tools
        self.system_prompt = self.make_system_prompt()

        self.tools_map = {}
        for tool in self.tools:
            self.tools_map[tool.name] = tool


    def get_function_tools(self) -> List[Dict[str, dict]]:
        return [x.to_param() for x in self.tools]

    async def run(self, mode="debug"):
        while True:
            q = input("请输入您的问题(quit 或者 exit退出):\nUser:")
            if q == "quit" or q == "exit":
                return

            self.update_memory(q, "user")
            if mode == "debug":
                print(self.memory)

            functions = self.get_function_tools()
            print(functions)

            resp, funcs_to_call = await self.llm.ask_tool(self.memory.messages, tools=functions)  # 上下文的来源，token消耗

            for func in funcs_to_call:
                f: BaseTool = self.tools_map[func["name"].lower()]
                args = func["arguments"]
                if not f:
                    continue
                try:
                    tool_res = await f.execute(**args)
                    resp += "\n" + tool_res
                except Exception as e:
                    resp += str(e)

            print("Agent:", resp, "\n")
            self.update_memory(resp)

    def make_system_prompt(self) -> str:
        prompt = TOOLCALL_PROMPT
        for tool in self.tools:
            tool_desc = tool.desc
            tool_params = json.dumps(tool.params)
            prompt += "## 下面是工具的描述"
            prompt += tool_desc + tool_params
