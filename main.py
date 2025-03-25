import asyncio
from app.llm import LLM, LLMSettings
from app.agent.graphlook import BaseAgent


async def test_base_agent():
    llm_settings = LLMSettings(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-a00f8cd1ceeb46018027d05fcdf4e898",
    )
    llm = LLM(llm_config=llm_settings)
    bs_agent = BaseAgent("test1", "a simple agent", llm)
    await bs_agent.run()


if __name__ == "__main__":
    asyncio.run(test_base_agent())