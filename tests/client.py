import asyncio
from typing import Optional
from mcp import ClientSession, StdioServerParameters, stdio_client
from contextlib import AsyncExitStack
from openai import OpenAI


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
            base_url="http:127.0.0.1:11434/v1",
            api_key="ollama",
        )
        self.model="deepseek-r1:latest",


    def get_response(self, messages: list, tools: list):
        response = self.openai.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=messages,
            tools=tools,  # 前提必须支持 function_call
        )
        return response
    
    async def cleanup(self):
        await self.exit_stack.aclose()

    async def get_tools(self):
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,  # 工具描述
                    "parameters": tool.inputSchema,  # 工具输入模式
                },
            }
            for tool in response.tools
        ]
        return available_tools
    
    async def connect_to_server(self, server_script_path: str):
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )
        # 使用 stdio_client 创建与服务器的 stdio 传输
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()
        # 列出可用工具
        response = await self.session.list_tools()
        tools = response.tools
        print("\\n连接到服务器，工具列表:", [tool.name for tool in tools])


async def main():
    client = MCPClient()
    await client.connect_to_server("C:\\Users\\Administrator\\Desktop\\projects\\GraphLook\\tests\\addtool.py")

    q = "使用工具 帮我计算 23 + 45的结果"
    message = {
        "role": "user",
        "content": q
    }
    resp = client.get_response([message], client.get_tools())
    print(resp)
    await client.cleanup()


if __name__ == '__main__':
    asyncio.run(main())