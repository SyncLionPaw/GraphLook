import asyncio
from app.llm import LLM, Config
from app.agent.base import BaseAgent, MemoryAgent
from app.agent.toolcall import ToolCallAgent
from app.agent.planning import PlanningAgent


async def test_base_agent():
    llm_settings = Config.get_llm_config()

    llm = LLM(llm_config=llm_settings)
    bs_agent = BaseAgent("test1", "a simple agent", llm)
    await bs_agent.run()


async def test_memory_agent():
    llm_settings = Config.get_llm_config()

    llm = LLM(llm_config=llm_settings)
    bs_agent = MemoryAgent("test2", "a simple agent", llm, max_message=3)
    await bs_agent.run()


async def test_toolcall_agent():
    llm_settings = Config.get_llm_config()

    llm = LLM(llm_config=llm_settings)
    tc_agent = ToolCallAgent("test3", "tool calls agent", llm, max_message=3)
    await tc_agent.run()


async def test_planning_agent():
    llm_settings = Config.get_llm_config()

    llm = LLM(llm_config=llm_settings)
    tc_agent = PlanningAgent("test4", "planning agent", llm, max_message=3)
    await tc_agent.run()


if __name__ == "__main__":
    # asyncio.run(test_base_agent())
    # asyncio.run(test_memory_agent())
    # asyncio.run(test_toolcall_agent())
    asyncio.run(test_planning_agent())
