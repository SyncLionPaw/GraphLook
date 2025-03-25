import asyncio
from app.llm import LLM, Config
from app.agent.base import BaseAgent, MemoryAgent


async def test_base_agent():
    llm_settings = Config.get_llm_config()

    llm = LLM(llm_config=llm_settings)
    bs_agent = BaseAgent("test1", "a simple agent", llm)
    await bs_agent.run()


async def test_memory_agent():
    llm_settings = Config.get_llm_config()

    llm = LLM(llm_config=llm_settings)
    bs_agent = MemoryAgent("test1", "a simple agent", llm, max_message=3)
    await bs_agent.run()


if __name__ == "__main__":
    # asyncio.run(test_base_agent())
    asyncio.run(test_memory_agent())